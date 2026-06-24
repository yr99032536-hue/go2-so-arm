"""Bridge a physical SO leader arm to the Isaac Sim SO-Arm ZMQ action port."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROBOT_MODELS_ROOT = THIS_DIR.parent
if str(ROBOT_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_MODELS_ROOT))

from soarm_nbv.safety import ActionSmoother, SOARM_JOINT_ORDER, clamp_joint_targets_deg
from soarm_nbv.zmq_bridge import ActionPublisher, SoArmAction, ZmqEndpointConfig


LEROBOT_SRC = Path("/home/iy/Isaac/lerobot/src")
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.teleoperators.so_leader import SO100Leader, SO101Leader  # noqa: E402
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: E402


class RuntimeOffsetConfig:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self._last_mtime_ns: int | None = None
        self._offsets_deg = np.zeros(len(SOARM_JOINT_ORDER), dtype=np.float32)

    @property
    def offsets_deg(self) -> np.ndarray:
        if not self.path:
            return self._offsets_deg
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self._last_mtime_ns is not None:
                self._last_mtime_ns = None
                self._offsets_deg = np.zeros(len(SOARM_JOINT_ORDER), dtype=np.float32)
                print(f">>> Runtime offsets cleared (missing file): {self.path}")
            return self._offsets_deg

        if stat.st_mtime_ns == self._last_mtime_ns:
            return self._offsets_deg

        payload = json.loads(self.path.read_text())
        new_offsets = np.zeros(len(SOARM_JOINT_ORDER), dtype=np.float32)
        if isinstance(payload, dict):
            for idx, joint_name in enumerate(SOARM_JOINT_ORDER):
                if joint_name in payload:
                    new_offsets[idx] = float(payload[joint_name])
        else:
            raise ValueError(f"Runtime offset file must be a JSON object: {self.path}")

        self._last_mtime_ns = stat.st_mtime_ns
        self._offsets_deg = new_offsets
        print(f">>> Runtime offsets loaded from {self.path}: {self._offsets_deg}")
        return self._offsets_deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO leader to Isaac Sim ZMQ teleoperation bridge.")
    parser.add_argument("--leader-port", required=True, help="Serial port of the physical SO leader arm.")
    parser.add_argument(
        "--leader-type",
        choices=("so100_leader", "so101_leader"),
        default="so101_leader",
        help="LeRobot teleoperator type.",
    )
    parser.add_argument("--leader-id", default="sim_leader", help="Calibration id used by LeRobot.")
    parser.add_argument(
        "--calibration-dir",
        default="",
        help="Optional LeRobot calibration directory override.",
    )
    parser.add_argument(
        "--action-port",
        type=int,
        default=5556,
        help="ZMQ action port consumed by nbv_v5.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Target publish rate.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Low-pass smoothing factor. Use 1.0 to disable smoothing.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip automatic calibration on connect and use the saved calibration as-is.",
    )
    parser.add_argument(
        "--runtime-offset-json",
        default="",
        help="Optional JSON file with per-joint live trim offsets in degrees.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print every N published actions. 0 disables per-action logging.",
    )
    return parser.parse_args()


def make_leader(args: argparse.Namespace):
    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else None
    config = SOLeaderTeleopConfig(
        port=args.leader_port,
        id=args.leader_id,
        use_degrees=True,
        calibration_dir=calibration_dir,
    )
    leader_cls = SO100Leader if args.leader_type == "so100_leader" else SO101Leader
    return leader_cls(config)


def action_dict_to_array(action_dict: dict[str, float]) -> np.ndarray:
    target = np.asarray([action_dict[f"{joint}.pos"] for joint in SOARM_JOINT_ORDER], dtype=np.float32)
    return clamp_joint_targets_deg(target)


def main() -> int:
    args = parse_args()
    period_s = 1.0 / max(args.fps, 1.0)
    smoother = ActionSmoother(alpha=float(np.clip(args.alpha, 0.0, 1.0)))
    leader = make_leader(args)
    publisher = ActionPublisher(ZmqEndpointConfig(action_port=args.action_port))
    runtime_offsets = RuntimeOffsetConfig(args.runtime_offset_json) if args.runtime_offset_json else None

    print(f">>> SO leader bridge starting on {args.leader_port}")
    print(f">>> Leader type: {args.leader_type}")
    print(f">>> Action PUB tcp://*:{args.action_port}")
    print(f">>> Joint order: {', '.join(SOARM_JOINT_ORDER)}")
    if runtime_offsets is not None:
        print(f">>> Runtime offset JSON: {runtime_offsets.path}")

    leader.connect(calibrate=not args.no_calibrate)
    step_count = 0
    try:
        while True:
            loop_start = time.perf_counter()
            target_deg = action_dict_to_array(leader.get_action())
            target_deg = smoother.update(target_deg)
            if runtime_offsets is not None:
                target_deg = clamp_joint_targets_deg(target_deg + runtime_offsets.offsets_deg)
            publisher.publish(SoArmAction(joint_target_deg=target_deg))
            if args.log_every > 0 and step_count % args.log_every == 0:
                print(f">>> leader target deg: {target_deg}")
            elapsed = time.perf_counter() - loop_start
            step_count += 1
            time.sleep(max(period_s - elapsed, 0.0))
    except KeyboardInterrupt:
        print(">>> SO leader bridge stopped")
        return 0
    finally:
        try:
            leader.disconnect()
        finally:
            publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
