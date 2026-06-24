# -*- coding: utf-8 -*-
"""主人认定模块 — 首次深度交互自动绑定 + 转让。

认主规则:
  1. 开箱后，第一个被 remember_fact(name=xxx) 的人自动成为 owner
  2. owner 可以说"把你送给xxx" → 转让所有权(需 LLM function_call transfer_ownership)
  3. owner.json 持久化到 data/owner.json

权限:
  - owner 可以删除任何人的记忆(clear_memory / forget_fact 对任意 pid)
  - 非 owner 只能删除自己的记忆
"""

import json
import os
import time
from typing import Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OWNER_PATH = os.path.join(_REPO, "data", "owner.json")


class OwnerManager:
    def __init__(self, path: str = _OWNER_PATH):
        self._path = path
        self._data: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self._path) and os.path.getsize(self._path) > 0:
            with open(self._path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._path)

    @property
    def owner_pid(self) -> Optional[str]:
        return self._data.get("person_id")

    @property
    def owner_name(self) -> Optional[str]:
        return self._data.get("name")

    def has_owner(self) -> bool:
        return bool(self._data.get("person_id"))

    def is_owner(self, person_id: str) -> bool:
        return self._data.get("person_id") == person_id

    def try_claim(self, person_id: str, name: str) -> bool:
        """首次认主: 仅当尚无 owner 时绑定。返回是否成功。"""
        if self.has_owner():
            return False
        self._data = {
            "person_id": person_id,
            "name": name,
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()
        return True

    def transfer(self, new_pid: str, new_name: str) -> str:
        """owner 转让所有权。调用方需先验证操作者是当前 owner。"""
        old = self._data.get("name", "unknown")
        self._data = {
            "person_id": new_pid,
            "name": new_name,
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "transferred_from": old,
        }
        self._save()
        return f"所有权已从 {old} 转让给 {new_name}"

    def can_delete_memory(self, actor_pid: str, target_pid: str) -> bool:
        """判断 actor 是否有权删除 target 的记忆。"""
        if actor_pid == target_pid:
            return True
        if self.is_owner(actor_pid):
            return True
        return False

    def update_owner_pid(self, old_pid: str, new_pid: str):
        """合并人脸后同步 owner pid。"""
        if self._data.get("person_id") == old_pid:
            self._data["person_id"] = new_pid
            self._save()
