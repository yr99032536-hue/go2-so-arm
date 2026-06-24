"""Safety helpers for SO-Arm action targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SOARM_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


DEFAULT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-110.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
    "gripper": (-10.0, 100.0),
}


@dataclass
class ActionSmoother:
    alpha: float = 0.25
    _last: np.ndarray | None = None

    def reset(self) -> None:
        self._last = None

    def update(self, target: np.ndarray) -> np.ndarray:
        target = target.astype(np.float32)
        if self._last is None:
            self._last = target.copy()
            return target
        self._last = (1.0 - self.alpha) * self._last + self.alpha * target
        return self._last.astype(np.float32)


def clamp_joint_targets_deg(target: np.ndarray) -> np.ndarray:
    clipped = np.nan_to_num(target.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    for i, joint_name in enumerate(SOARM_JOINT_ORDER):
        lo, hi = DEFAULT_LIMITS_DEG[joint_name]
        clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


def deg_to_rad(target_deg: np.ndarray) -> np.ndarray:
    return np.deg2rad(target_deg).astype(np.float32)


def rad_to_deg(joint_pos_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(joint_pos_rad).astype(np.float32)
