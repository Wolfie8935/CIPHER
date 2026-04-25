"""
HTTP `Environment` adapter: bridges `CIPHEREnv` to `openenv_core` HTTP server types.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from openenv_core.env_server.interfaces import Environment
from openenv_core.env_server.types import Action, Observation, State

from cipher.env_wrapper import CIPHEREnv


@dataclass(kw_only=True)
class CIPHERAction(Action):
    text: str = ""


@dataclass(kw_only=True)
class CIPHERObservation(Observation):
    text: str = ""


class CIPHERHTTPEnvironment(Environment):
    """Serves CIPHER (stub-friendly default) over the OpenEnv Core HTTP contract."""

    def __init__(self) -> None:
        super().__init__()
        self._env = CIPHEREnv(max_steps=20, llm_mode="stub")
        self._state = State()

    @property
    def state(self) -> State:
        eid: Optional[Union[str, int]] = getattr(
            self._env, "_episode_count", self._state.step_count
        )
        if eid is not None:
            return State(episode_id=str(eid), step_count=1)
        return self._state

    def reset(self) -> CIPHERObservation:
        obs_text, info = self._env.reset()
        return CIPHERObservation(
            text=obs_text,
            done=False,
            reward=None,
            metadata=_as_metadata_dict(info),
        )

    def step(self, action: Action) -> CIPHERObservation:
        if isinstance(action, CIPHERAction):
            text = action.text or (
                str(action.metadata.get("text", "")) if action.metadata else ""
            )
        else:
            meta = getattr(action, "metadata", {}) or {}
            text = str(meta.get("text", ""))
        obs, reward, terminated, truncated, info = self._env.step(text)
        done = bool(terminated or truncated)
        return CIPHERObservation(
            text=obs,
            done=done,
            reward=float(reward) if reward is not None else None,
            metadata=_as_metadata_dict(info),
        )


def _as_metadata_dict(data: Any) -> Dict[str, Any]:
    """Ensure metadata is a JSON-friendly dict (HTTP response safe)."""
    if not isinstance(data, dict):
        return {"value": str(data)}
    out: Dict[str, Any] = {}
    for k, v in data.items():
        try:
            json.dumps(v, default=str)
            out[str(k)] = v
        except (TypeError, ValueError):
            out[str(k)] = str(v)
    return out
