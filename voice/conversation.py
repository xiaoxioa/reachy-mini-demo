# -*- coding: utf-8 -*-
"""ConversationHandler — 协议无关对话业务逻辑。

从 ChatCallback + RealtimeDialog 提取，通过 VoicePipeline 接口与后端通信。
工具分发、记忆管理、barge-in、transcript 录制均在此处理。
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from typing import Callable

from voice.pipeline.base import VoicePipeline
from voice.pipeline.events import EventType, PipelineEvent
from voice.config import (
    BYE_PHRASES, POINT_FRESH_S, SUMMARY_MODEL,
    CONV_SUMMARY_THRESHOLD,
)
from voice.state import State, log
import voice.state as _st_mod
from memory.safety import handle_clear_memory_intent, handle_confirm_clear


# ── transcript 泄漏标签 → 物理动作兜底 ──

_TAG_TO_ACTION = {
    "nod": "nod", "点头": "nod", "nodding": "nod",
    "shake": "shake_head", "shake_head": "shake_head", "摇头": "shake_head",
    "wiggle": "wiggle_antennas", "摆天线": "wiggle_antennas", "wave": "wiggle_antennas",
    "tilt": "tilt_head", "歪头": "tilt_head", "tilt_head": "tilt_head",
    "smile": "wiggle_antennas", "微笑": "wiggle_antennas",
    "look_left": "look_left", "look_right": "look_right",
    "look_up": "look_up", "look_down": "look_down",
}
_ACTION_TAG_RE = re.compile(
    r"</?(?:" + "|".join(re.escape(k) for k in _TAG_TO_ACTION) + r")[^>]*>"
    r"|[（(](?:" + "|".join(re.escape(k) for k in _TAG_TO_ACTION) + r")[)）]"
    r"|\*(?:" + "|".join(re.escape(k) for k in _TAG_TO_ACTION) + r")\*",
    re.IGNORECASE,
)


def _extract_tag_action(match_str: str) -> str | None:
    s = match_str.strip("<>/()（）* \t").lower()
    return _TAG_TO_ACTION.get(s)


class ConversationHandler:
    """协议无关的对话业务逻辑。

    使用::

        handler = ConversationHandler(st, pipeline, ...)
        pipeline.set_event_handler(handler.handle_event)
    """

    def __init__(
        self,
        st: State,
        pipeline: VoicePipeline,
        play_q: queue.Queue,
        motion_q: queue.Queue,
        snap_q: queue.Queue,
        clear_player_fn: Callable[[], None],
        memory_mgr,
        owner_mgr,
        id_recognizer,
        oai_client,
        instructions: str,
        tools: list,
        no_memory: bool = False,
        face_pipeline=None,
    ):
        self.st = st
        self.pipeline = pipeline
        self.play_q = play_q
        self.motion_q = motion_q
        self.snap_q = snap_q
        self._clear_player = clear_player_fn
        self.memory_mgr = memory_mgr
        self.owner_mgr = owner_mgr
        self.id_recognizer = id_recognizer
        self.oai_client = oai_client
        self.instructions = instructions
        self.tools = tools
        self.no_memory = no_memory
        self.face_pipeline = face_pipeline
        self.exit_i = 0

    # ── 主分发 ──

    def handle_event(self, event: PipelineEvent) -> None:
        _dispatch = {
            EventType.SESSION_READY: self._on_session_ready,
            EventType.USER_SPEECH_START: self._on_speech_start,
            EventType.USER_SPEECH_END: self._on_speech_end,
            EventType.USER_TRANSCRIPT: self._on_user_transcript,
            EventType.RESPONSE_START: self._on_response_start,
            EventType.RESPONSE_TEXT_DELTA: self._on_text_delta,
            EventType.RESPONSE_TEXT_DONE: self._on_text_done,
            EventType.RESPONSE_AUDIO_DELTA: self._on_audio_delta,
            EventType.RESPONSE_DONE: self._on_response_done,
            EventType.TOOL_CALL: self._on_tool_call,
            EventType.ERROR: self._on_error,
        }
        handler = _dispatch.get(event.type)
        if handler is None:
            return
        try:
            handler(event)
        except Exception as e:
            log(f"❌ handle_event({event.type.value}) 异常:{type(e).__name__}: {e}")

    # ── 事件处理 ──

    def _on_session_ready(self, event: PipelineEvent) -> None:
        log("✅ 会话配置生效")
        self.st.session_updated.set()

    def _on_speech_start(self, event: PipelineEvent) -> None:
        st = self.st
        now = time.monotonic()
        with st.lock:
            st.last_interaction_at = now
            st.user_speaking = True
            playing = (now < st.playback_end_estimate) or (not self.play_q.empty())
            in_flight = st.in_flight > 0
        log("🎤 检测到你开始说话…")
        if playing or in_flight:
            self._do_barge_in(in_flight)

    def _do_barge_in(self, in_flight: bool) -> None:
        st = self.st
        with st.lock:
            st.play_gen += 1
            st.drop_audio = True
            st.playback_end_estimate = time.monotonic()
        while True:
            try:
                self.play_q.get_nowait()
            except queue.Empty:
                break
        try:
            self._clear_player()
        except Exception as e:
            log(f"⚠ clear_player 失败:{type(e).__name__}: {e}")
        if in_flight:
            self.pipeline.cancel_response()
        if not st.no_expression:
            with st.lock:
                st.wake_cue = "barge"
                st.wake_cue_t = time.monotonic()
        log("⛔ 打断:已停止播放" + (",并取消在途回复" if in_flight else ""))

    def _on_speech_end(self, event: PipelineEvent) -> None:
        with self.st.lock:
            self.st.thinking = True
            self.st.user_speaking = False
        log("🤫 检测到你说完了,等模型回应…")

    def _on_user_transcript(self, event: PipelineEvent) -> None:
        st = self.st
        _transcript = (event.data.get("text") or "").strip()
        with st.lock:
            _log_pid = st.current_person_id or "_unknown"
            _log_name = st.current_person_name
            _injected_pid = st.identity_injected_pid
        log(f"📝 ASR结果:「{_transcript}」 当前人={_log_name}({_log_pid[:12]})"
            f" 已注入记忆pid={_injected_pid or '无'}")
        if not _transcript:
            return
        with st.lock:
            st.display_transcript_seq += 1
            st.display_transcript.append({
                "seq": st.display_transcript_seq,
                "ts": time.strftime("%H:%M:%S"),
                "role": "user", "text": _transcript,
                "pid": _log_pid, "name": _log_name,
            })
            if len(st.display_transcript) > 100:
                st.display_transcript = st.display_transcript[-80:]
        if not self.no_memory:
            with st.lock:
                st.conversation_log.setdefault(_log_pid, []).append(
                    ("user", _transcript))
                _check_log = st.conversation_log.get(_log_pid, [])
                _est_tok = sum(len(t) * 1.5 for _, t in _check_log)
            if (_est_tok > CONV_SUMMARY_THRESHOLD
                    and _log_pid != "_unknown" and self.memory_mgr):
                with st.lock:
                    _snap = list(st.conversation_log.get(_log_pid, []))
                    st.conversation_log[_log_pid] = []
                threading.Thread(target=self.save_summary,
                                 args=(_log_pid, _snap), daemon=True).start()
                log(f"📝 上下文过长，自动触发 consolidation"
                    f"({_log_pid[:12]}, ~{int(_est_tok)} tok)")

    def _on_response_start(self, event: PipelineEvent) -> None:
        st = self.st
        now = time.monotonic()
        with st.lock:
            st.in_flight += 1
            st.drop_audio = False
            st.resp_audio_count = 0
            st.fc_seen_this_resp = False
            st.last_interaction_at = now
            st.resp_snapshot_pid = st.current_person_id
            st.resp_snapshot_name = st.current_person_name
            _dt_seq = st.display_transcript_seq
            _rc_name = st.current_person_name
            _rc_pid = st.current_person_id
            _rc_injected = st.identity_injected
            _rc_injected_pid = st.identity_injected_pid
        if _st_mod._current_turn is not None:
            _st_mod._current_turn["dt_seq"] = _dt_seq
        log(f"💭 模型开始生成回复… 当前人={_rc_name}"
            f"({(_rc_pid or '')[:12]}) "
            f"injected={_rc_injected}(pid={_rc_injected_pid or '无'})")

    def _on_tool_call(self, event: PipelineEvent) -> None:
        st = self.st
        name = event.data.get("name", "")
        call_id = event.data.get("call_id", "")
        args_str = event.data.get("arguments", "{}")

        with st.lock:
            st.fc_seen_this_resp = True
            st.fc_gen = st.play_gen
            st.display_transcript_seq += 1
            st.display_transcript.append({
                "seq": st.display_transcript_seq,
                "ts": time.strftime("%H:%M:%S"),
                "role": "tool_call",
                "text": f"{name}({args_str})",
                "pid": st.resp_snapshot_pid or st.current_person_id or "_unknown",
                "name": st.resp_snapshot_name or st.current_person_name,
            })
        log(f"🤖 模型调用工具: {name}({args_str[:200]})")

        if name == "take_snapshot":
            with st.lock:
                maybe_pointing = (
                    (time.monotonic() - st.finger_ext_at) < POINT_FRESH_S)
                st.snapshot_pending += 1
            mode = "judge" if maybe_pointing else "scene"
            if maybe_pointing:
                log("👉 最近见过伸指 → 先原地看图判断是否真在指(两段式)")
            self.snap_q.put({
                "call_id": call_id, "gen": st.fc_gen, "mode": mode})

        elif name == "end_session":
            phrase = BYE_PHRASES[self.exit_i % len(BYE_PHRASES)]
            self.exit_i += 1
            output = {
                "success": True,
                "say": (f"对话结束。用中文只说这一句简短告别:"
                        f"「{phrase}」,别追问、别挽留、别加别的。"),
            }
            self._submit_tool_result("end_session", call_id, output)
            with st.lock:
                st.exit_request = True
            log(f"👋 收到结束意图 → 告别「{phrase}」+ 回待命")

        elif name == "identify_pointed_object":
            with st.lock:
                st.snapshot_pending += 1
            log("👉 收到指向请求 → 先原地看图判断(两段式)")
            self.snap_q.put({
                "call_id": call_id, "gen": st.fc_gen, "mode": "judge"})

        elif name in ("remember_fact", "forget_fact"):
            self._handle_memory_tool(name, call_id, args_str)

        elif name == "clear_memory":
            try:
                args_dict = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args_dict = {}
            result = handle_clear_memory_intent(
                st, args_dict, self.pipeline, self.id_recognizer)
            self._submit_tool_result("clear_memory", call_id,
                                     {"result": result})
            log(f"🔒 clear_memory 启动: {result}")

        elif name == "confirm_clear":
            try:
                args_dict = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args_dict = {}
            result = handle_confirm_clear(
                st, args_dict, self.memory_mgr, self.id_recognizer)
            self._submit_tool_result("confirm_clear", call_id,
                                     {"result": result})
            log(f"🔒 confirm_clear: {result}")

        else:
            self.motion_q.put({"name": name, "call_id": call_id})
            self._submit_tool_result(
                name, call_id, {"success": True, "action": name})

    def _handle_memory_tool(self, name: str, call_id: str,
                            args_str: str) -> None:
        st = self.st
        with st.lock:
            pid = st.resp_snapshot_pid or st.current_person_id
        if pid is None:
            result = "当前没有识别到用户身份,无法存储记忆。"
        else:
            try:
                args_dict = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args_dict = {}
            result = self.memory_mgr.handle_tool_call(pid, name, args_dict)
            with st.lock:
                st.identity_injected = False
                st.identity_injected_pid = None
            if name == "remember_fact":
                self._handle_remember_name(pid, args_dict)
            elif name == "forget_fact":
                keyword = args_dict.get("keyword", "")
                if "名" in keyword or "name" in keyword.lower():
                    self.memory_mgr.set_name(pid, None)
                    if self.id_recognizer is not None:
                        self.id_recognizer.db.set_name(pid, None)
                    with st.lock:
                        st.current_person_name = None
        self._submit_tool_result(name, call_id, {"result": result})
        log(f"🧠 记忆工具 {name}: {result}")

    def _handle_remember_name(self, pid: str, args_dict: dict) -> None:
        new_name = args_dict.get("name")
        if not new_name:
            return
        self.memory_mgr.set_name(pid, new_name)
        if self.id_recognizer is not None:
            self.id_recognizer.db.set_name(pid, new_name)
        if self.face_pipeline is not None:
            try:
                if self.face_pipeline.store.confirm_identity(pid, new_name):
                    self.face_pipeline.save_gallery()
                    log(f"🏷 gallery 身份已确认并落盘: "
                        f"{new_name} ({pid[:12]})")
            except Exception as _e:
                log(f"⚠ gallery 命名失败:{type(_e).__name__}: {_e}")
        with self.st.lock:
            self.st.current_person_name = new_name
        if self.owner_mgr is not None and not self.owner_mgr.has_owner():
            if self.owner_mgr.try_claim(pid, new_name):
                log(f"👑 认主成功: {new_name} ({pid})")

    def _on_text_delta(self, event: PipelineEvent) -> None:
        print(event.data.get("delta", ""), end="", flush=True)

    def _on_text_done(self, event: PipelineEvent) -> None:
        st = self.st
        print(flush=True)
        _atext = (event.data.get("text") or "").strip()
        if _atext:
            for m in _ACTION_TAG_RE.finditer(_atext):
                act = _extract_tag_action(m.group())
                if act:
                    log(f"⚠ 标签泄漏兜底: '{m.group()}' → 触发 {act}")
                    self.motion_q.put({"name": act})
            _atext = _ACTION_TAG_RE.sub("", _atext).strip()
        if _atext:
            with st.lock:
                _log_pid = (st.resp_snapshot_pid
                            or st.current_person_id or "_unknown")
                _log_name = (st.resp_snapshot_name
                             or st.current_person_name)
                st.display_transcript_seq += 1
                st.display_transcript.append({
                    "seq": st.display_transcript_seq,
                    "ts": time.strftime("%H:%M:%S"),
                    "role": "assistant", "text": _atext,
                    "pid": _log_pid, "name": _log_name,
                })
                if len(st.display_transcript) > 100:
                    st.display_transcript = st.display_transcript[-80:]
            log(f"📝 模型回复:「{_atext[:100]}"
                f"{'…' if len(_atext) > 100 else ''}」"
                f" 归属={_log_name}({_log_pid[:12]})")
            if not self.no_memory:
                with st.lock:
                    st.conversation_log.setdefault(
                        _log_pid, []).append(("assistant", _atext))

    def _on_audio_delta(self, event: PipelineEvent) -> None:
        st = self.st
        with st.lock:
            if st.drop_audio:
                return
            gen = st.play_gen
            st.resp_audio_count += 1
            if st.thinking:
                st.thinking = False
        self.play_q.put((gen, event.data["pcm_16k"]))

    def _on_response_done(self, event: PipelineEvent) -> None:
        st = self.st
        now = time.monotonic()
        fire_rc = False
        with st.lock:
            st.in_flight = max(0, st.in_flight - 1)
            st.resp_snapshot_pid = None
            st.resp_snapshot_name = None
            st.last_interaction_at = now
            if (st.fc_seen_this_resp
                    and st.resp_audio_count == 0
                    and st.fc_gen == st.play_gen
                    and st.snapshot_pending == 0):
                fire_rc = True
        d = event.data.get("first_audio_delay")
        log(f"✅ 本轮回复完成"
            f"{f'(首音频延迟 {d:.0f}ms)' if d else ''}")
        if fire_rc:
            log("📤 trigger_response(仅调工具无语音) 自动触发")
            self.pipeline.trigger_response()

    def _on_error(self, event: PipelineEvent) -> None:
        log(f"❌ 服务端错误事件:{event.data}")

    # ── Helpers ──

    def _submit_tool_result(self, name: str, call_id: str,
                            output: dict) -> None:
        output_json = json.dumps(output, ensure_ascii=False)
        try:
            log(f"📤 submit_tool_result call_id={call_id[:8]}"
                f" tool={name} output={output_json[:200]}")
            self.pipeline.submit_tool_result(call_id, output_json)
            self._record_tool_output(name, call_id, output_json)
        except Exception as e:
            log(f"⚠ {name} 回 output 失败:{e}")

    def _record_tool_output(self, name: str, call_id: str,
                            output: str) -> None:
        st = self.st
        with st.lock:
            st.display_transcript_seq += 1
            st.display_transcript.append({
                "seq": st.display_transcript_seq,
                "ts": time.strftime("%H:%M:%S"),
                "role": "tool_output",
                "text": f"{name} → {output[:120]}",
                "call_id": call_id,
            })
            if len(st.display_transcript) > 100:
                st.display_transcript = st.display_transcript[-80:]

    # ── Session management ──

    def open_session(self, timeout: float = 10.0) -> bool:
        return self.pipeline.open_session(
            self.instructions, self.tools, timeout)

    def close_session(self) -> None:
        st = self.st
        with st.lock:
            _all_logs = dict(st.conversation_log)
            st.conversation_log.clear()
            st.identity_injected = False
            st.identity_injected_pid = None
            st.current_person_id = None
            st.current_person_name = None
            st.current_is_owner = False
            st.user_speaking = False
            if st.clear_workflow is not None:
                st.clear_workflow = None
                st.clear_lock = False
        self.pipeline.close_session()
        if self.memory_mgr and not self.no_memory:
            for _pid, _log in _all_logs.items():
                if _pid != "_unknown" and len(_log) >= 2:
                    threading.Thread(target=self.save_summary,
                                     args=(_pid, _log),
                                     daemon=True).start()

    def update_memory(self, pid: str, pname: str | None) -> bool:
        st = self.st
        mem_prompt = (self.memory_mgr.get_prompt(pid, person_name=pname)
                      if self.memory_mgr else None)
        new_instr = self.instructions + (
            "\n\n" + mem_prompt if mem_prompt else "")
        try:
            log(f"📤 update_instructions(记忆注入) pid={pid} name={pname}")
            self.pipeline.update_instructions(new_instr, self.tools)
            log(f"🧠 记忆已注入 session instructions "
                f"({pname or pid[:12]})")
            with st.lock:
                st.identity_injected = True
                st.identity_injected_pid = pid
                st.dbg_memory_prompt = mem_prompt
                st.dbg_session_instructions = new_instr
                _buffered = list(st.audio_gate_buffer)
                st.audio_gate_buffer.clear()
                st.audio_gate_closed = False
            if _buffered:
                for chunk in _buffered:
                    try:
                        self.pipeline.feed_audio(chunk)
                    except Exception:
                        break
                log(f"🔓 音频闸门开启，flush {len(_buffered)} 帧缓存")
            return True
        except Exception as e:
            log(f"⚠ 记忆 update_instructions 失败:{e}")
            return False

    def restart_for_switch(self, old_pid: str | None, new_pid: str,
                           new_pname: str | None) -> bool:
        st = self.st
        log(f"🔄 身份切换重启: "
            f"{old_pid and old_pid[:8]}→{new_pid[:8]} ({new_pname})")
        if old_pid and self.memory_mgr and not self.no_memory:
            with st.lock:
                _old_log = list(st.conversation_log.get(old_pid, []))
                st.conversation_log.pop(old_pid, None)
            if len(_old_log) >= 2:
                threading.Thread(target=self.save_summary,
                                 args=(old_pid, _old_log),
                                 daemon=True).start()
        self.pipeline.close_session()
        with st.lock:
            st.in_flight = 0
            st.resp_audio_count = 0
            st.fc_seen_this_resp = False
            st.drop_audio = False
            st.pending_identity_restart = False
        if self.open_session():
            self.update_memory(new_pid, new_pname)
            log(f"✅ 会话重启完成，已注入 "
                f"{new_pname or new_pid[:12]} 的记忆")
            return True
        log("⚠ 会话重启失败(open_session 超时)")
        return False

    def save_summary(self, pid: str, conv_log: list) -> None:
        try:
            text = "\n".join(
                f"{'用户' if r == 'user' else '小艺'}: {t}"
                for r, t in conv_log if t and t.strip())
            if len(text) < 20:
                return
            current_facts = self.memory_mgr.get_facts(pid)
            current_name = self.memory_mgr.get_name(pid)
            facts_str = (json.dumps(current_facts, ensure_ascii=False)
                         if current_facts else "{}")
            prompt = (
                "你是记忆管理助手。仔细阅读对话，提取关于用户的所有个人信息。\n\n"
                f"已有记忆：{facts_str}\n\n"
                f"对话内容：\n{text[-4000:]}\n\n"
                "任务：\n"
                "1. facts = KV 格式合并已有记忆 + 对话中发现的新信息。"
                "key 是类别，value 是内容。\n"
                "   - 重点提取：爱好、喜好、职业、年龄、习惯、观点、提到的人/事物\n"
                "   - 用户说'我喜欢X' → {\"喜欢的东西\": \"X\"}\n"
                "   - 如果新信息与旧 fact 矛盾，保留新的 value，同 key 覆盖\n"
                "   - 不要把名字放进 facts（名字在独立字段管理）\n"
                "2. summary = 一句话介绍这位用户。\n"
                "   - 体现对用户的整体理解，总结facts和对话而不是列举属性\n"
                "   - 仅做描述，例如：'喜欢打篮球和看科幻小说'\n"
                "3. episode = 客观总结这次对话的摘要事件\n"
                "   - topic: 具体说聊了什么，不要太笼统\n"
                "   - highlights: 关于用户的关键信息点（每条是完整短句）\n\n"
                "只输出JSON：\n"
                '{"name":"用户名字(对话中提到则更新,否则为null)",'
                '"summary":"一句话认知描述",'
                '"facts":{"类别1":"内容1","类别2":"内容2"},'
                '"episode":{"topic":"具体话题","highlights":["要点1"],'
                '"mood":"engaged/casual/emotional/tense"}}'
            )
            resp = self.oai_client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",
                     "content": "请根据上述信息生成记忆JSON。"},
                ],
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            new_facts = result.get("facts", current_facts)
            if isinstance(new_facts, list):
                new_facts = {f"备注{i+1}": f
                             for i, f in enumerate(new_facts)}
            new_name = result.get("name")
            if new_name is None and current_name:
                new_name = current_name
            new_summary = result.get("summary")
            episode = result.get("episode")
            self.memory_mgr.consolidate_facts(
                pid, new_facts, new_name, new_summary)
            if episode:
                self.memory_mgr.save_episode(pid, episode)
            log(f"📝 记忆 consolidation 完成 ({pid[:12]}): "
                f"{len(new_facts)} facts, "
                f"summary={new_summary[:30] if new_summary else 'none'}, "
                f"episode="
                f"{episode.get('topic', '')[:30] if episode else 'none'}")
            with self.st.lock:
                if self.st.current_person_id == pid:
                    self.st.identity_injected = False
        except Exception as e:
            log(f"⚠ 记忆 consolidation 失败({pid[:12]}):{e}")
