import argparse
import os
import sys
import numpy as np
import torch
import zmq
import subprocess
import signal
import atexit
import time
from pathlib import Path

ROBOT_MODELS_ROOT = Path(__file__).resolve().parents[2]
if str(ROBOT_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_MODELS_ROOT))

# MCP 익스텐션 자동 활성화
mcp_ext_path = "/home/iy/Documents/isaac-sim-mcp"
if os.path.exists(mcp_ext_path) and "--kit_args" not in sys.argv:
    sys.argv.append("--kit_args")
    sys.argv.append(f"--ext-folder={mcp_ext_path} --enable=isaac.sim.mcp_extension")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 + SO-Arm 시뮬레이션 (기본 평면 맵)")
parser.add_argument("--enable_gr00t", action="store_true", help="Enable ZMQ bridge to a GR00T SO-Arm policy node.")
parser.add_argument("--gr00t_obs_port", type=int, default=5555, help="ZMQ PUB port for camera/state observations.")
parser.add_argument("--gr00t_action_port", type=int, default=5556, help="ZMQ SUB port for GR00T arm actions.")
parser.add_argument(
    "--show_camera_viewport",
    action="store_true",
    help="Show a visible wrist camera viewport. Camera observations remain active without this flag.",
)
parser.add_argument(
    "--gr00t_apply_actions",
    action="store_true",
    help="Apply received GR00T actions to SO-Arm joint targets. Without this, observations are published only.",
)
parser.add_argument('--leader_auto', action='store_true', help='Auto-start leader_teleop_bridge.py as background subprocess (teleoperation).')
parser.add_argument('--leader_port_dev', default='/dev/ttyACM0', help='Serial port of the physical SO leader arm.')
parser.add_argument('--leader_type', default='so101_leader', choices=('so100_leader', 'so101_leader'))
parser.add_argument('--leader_id', default='teleop_leader_v1', help='Calibration id (MUST match a file in calibration/teleoperators/so_leader/).')
parser.add_argument('--leader_fps', type=float, default=30.0)
parser.add_argument('--leader_alpha', type=float, default=0.35)
parser.add_argument('--leader_sim_alpha', type=float, default=1.0, help='Isaac-side smoothing alpha for leader teleop actions.')
parser.add_argument('--leader_action_log_every', type=int, default=20, help='Print every N received leader actions. 0 disables per-action logs.')
parser.add_argument('--leader_script', default='/home/iy/Isaac/Robotics/robot_models/soarm_nbv/leader_teleop_bridge.py')
parser.add_argument('--leader_python', default='/home/iy/miniconda3/envs/lerobot/bin/python')
parser.add_argument('--leader_log', default='/tmp/soarm_leader_teleop.log')
parser.add_argument('--runtime_offset_json', default='/tmp/soarm_runtime_offset.json')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.leader_auto:
    args_cli.gr00t_apply_actions = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.utils import configclass
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors import RayCasterCfg, patterns

from soarm_nbv.safety import ActionSmoother, clamp_joint_targets_deg, deg_to_rad
from soarm_nbv.zmq_bridge import ActionSubscriber, ZmqEndpointConfig


GO2_POLICY_PATH = "/home/iy/Isaac/IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-04-28_13-45-45/exported/policy.pt"
GO2_LEG_JOINT_ORDER = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
GO2_ACTION_SCALE = 0.25
GO2_POLICY_OBS_DIM = 247
GR00T_ARM_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GR00T_FULL_JOINT_ORDER = GR00T_ARM_JOINT_ORDER + ["gripper"]
GR00T_ACTION_CLIP_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
    "gripper": (-10.0, 100.0),
}
SO_ARM_MASS = 1.0e-5
SO_ARM_INERTIA = 1.0e-8
SO_ARM_LINKS = [
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "gripper_frame_link",
    "moving_jaw_so101_v1_link",
    "arm_camera_lens_frame",
]


def export_stage(path: str) -> None:
    if not path:
        return

    import omni.usd

    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().Export(path)
    print(f">>> Stage exported: {path}")


class WasdKeyboard(Se2Keyboard):
    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            "W": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        }


def setup_camera_viewports():
    from omni.kit.viewport.utility import create_viewport_window

    camera_views = [
        ("Wrist Camera", "/World/envs/env_0/Robot/gripper_link/wrist_camera"),
    ]
    viewports = []
    for title, camera_path in camera_views:
        window = create_viewport_window(title, width=420, height=320, visible=True)
        window.viewport_api.set_active_camera(camera_path)
        viewports.append(window)
    print(">>> Live Camera Viewports: ENABLED (Wrist Camera)")
    return viewports


class Gr00tZmqBridge:
    def __init__(self, obs_port: int, action_port: int):
        self.context = zmq.Context()
        self.obs_pub = self.context.socket(zmq.PUB)
        self.obs_pub.bind(f"tcp://*:{obs_port}")
        self.action_sub = self.context.socket(zmq.SUB)
        self.action_sub.connect(f"tcp://localhost:{action_port}")
        self.action_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.action_sub.setsockopt(zmq.RCVTIMEO, 0)
        self.obs_port = obs_port
        self.action_port = action_port
        print(f">>> GR00T ZMQ Bridge: OBS PUB tcp://*:{obs_port}, ACTION SUB tcp://localhost:{action_port}")

    def close(self):
        self.obs_pub.close(0)
        self.action_sub.close(0)
        self.context.term()

    def publish_observation(self, wrist_rgb: np.ndarray, joint_pos: np.ndarray):
        # GR00T SO_ARM Starter expects room+wrist streams. In minimal mode nbv_v4 has
        # only one real camera, so duplicate wrist into the room stream contract.
        room_rgb = wrist_rgb
        self.obs_pub.send_multipart(
            [
                np.array(room_rgb.shape, dtype=np.int32).tobytes(),
                room_rgb.tobytes(),
                np.array(wrist_rgb.shape, dtype=np.int32).tobytes(),
                wrist_rgb.tobytes(),
                joint_pos.astype(np.float32).tobytes(),
            ]
        )

    def receive_action(self) -> np.ndarray | None:
        try:
            action_raw = self.action_sub.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        action = np.frombuffer(action_raw, dtype=np.float32)
        if action.size < len(GR00T_FULL_JOINT_ORDER):
            return None
        return action[: len(GR00T_FULL_JOINT_ORDER)]


def clip_gr00t_action(action: np.ndarray) -> np.ndarray:
    clipped = action.astype(np.float32).copy()
    for i, joint_name in enumerate(GR00T_FULL_JOINT_ORDER):
        lo, hi = GR00T_ACTION_CLIP_DEG[joint_name]
        clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


def make_so_arm_weightless():
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdPhysics, Usd

    stage = omni.usd.get_context().get_stage()
    patched = []
    for link_name in SO_ARM_LINKS:
        prim_path = f"/World/envs/env_0/Robot/{link_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid_body_api.CreateDisableGravityAttr(True)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(SO_ARM_MASS)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(SO_ARM_INERTIA, SO_ARM_INERTIA, SO_ARM_INERTIA))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(child)
                collision_api.CreateCollisionEnabledAttr(False)
                
        patched.append(link_name)
    print(f">>> SO-Arm Dynamics & Collision: DISABLED ({len(patched)} links)")


@configclass
class NBVv4SceneCfg(InteractiveSceneCfg):
    # 기본 환경
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))

    warehouse = None

    # 로봇 (Go2 + SO-Arm), instanceable=False 로 GUI 조작 가능
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/iy/robot_models/assets/urdf/go2_with_so_arm.urdf",
            make_instanceable=False,
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=20.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*calf_joint": -1.5,
                "shoulder_pan": 0.0,
                "shoulder_lift": -0.5,
                "elbow_flex": 1.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        ),
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                stiffness=400.0,
                damping=20.0,
            ),
        },
    )

    # 카메라
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_camera",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=0.35,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.05, 0.04, 0.0),
            rot=(0.37993, -0.59637, 0.59637, -0.37993),
            convention="ros",
        ),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )




_leader_proc = None


def stop_existing_leader_processes(leader_script: Path) -> None:
    leader_script_path = str(leader_script)
    try:
        proc_list = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f">>> [leader_auto] process scan warning: {exc}")
        return

    current_pid = os.getpid()
    stopped = []
    for line in proc_list.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_text, command, arguments = parts
        if "python" not in command or leader_script_path not in arguments:
            continue
        pid = int(pid_text)
        if pid == current_pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            continue
    if stopped:
        print(f">>> [leader_auto] stopped stale leader_bridge process(es): {stopped}")
        time.sleep(1.0)


def start_leader_subprocess(args_ns) -> None:
    """leader_teleop_bridge.py를 백그라운드 subprocess로 시작 (텔레오퍼레이션 자동 연결).

    핵심 방어:
      1) --no-calibrate 강제 → calibrate()/input() 진입 차단 (EOF 원천 차단)
      2) --leader-id teleop_leader_v1 → 실제 캘리브 파일 매칭 (self.calibration={} 방지)
    """
    global _leader_proc
    if not getattr(args_ns, 'leader_auto', False):
        return
    leader_script = Path(args_ns.leader_script)
    if not leader_script.is_file():
        print(f">>> [leader_auto] leader script not found, skipping: {leader_script}")
        return
    stop_existing_leader_processes(leader_script)
    log_fh = open(args_ns.leader_log, 'ab', buffering=0)
    cmd = [
        args_ns.leader_python, '-u', str(leader_script),
        '--leader-port', args_ns.leader_port_dev,
        '--leader-type', args_ns.leader_type,
        '--leader-id', args_ns.leader_id,
        '--action-port', str(args_ns.gr00t_action_port),
        '--fps', str(args_ns.leader_fps),
        '--alpha', str(args_ns.leader_alpha),
        '--runtime-offset-json', args_ns.runtime_offset_json,
        '--no-calibrate',
    ]
    _leader_proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    atexit.register(stop_leader_subprocess)
    try:
        signal.signal(signal.SIGTERM, lambda *_: (stop_leader_subprocess(), sys.exit(0)))
    except (ValueError, OSError):
        pass
    print(f">>> [leader_auto] leader_bridge PID={_leader_proc.pid} started")
    print(f">>> [leader_auto] log: {args_ns.leader_log}")
    time.sleep(1.5)
    rc = _leader_proc.poll()
    if rc is not None:
        print(f">>> [leader_auto] WARNING: leader_bridge exited early (code={rc}). Check: tail -50 {args_ns.leader_log}")
        print(f">>> [leader_auto] If calibration missing: run calibrate_leader.sh first.")
        _leader_proc = None


def stop_leader_subprocess() -> None:
    """leader 자식 프로세스 정리 (graceful SIGTERM → 강제 kill)."""
    global _leader_proc
    proc = _leader_proc
    _leader_proc = None
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(">>> [leader_auto] leader_bridge stopped")
    except Exception as e:
        print(f">>> [leader_auto] stop warning: {e}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = NBVv4SceneCfg(num_envs=1, env_spacing=5.0)
    if not args_cli.enable_cameras:
        scene_cfg.wrist_camera = None
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    camera_viewports = setup_camera_viewports() if args_cli.enable_cameras and args_cli.show_camera_viewport else []

    robot = scene["robot"]
    height_scanner = scene.sensors.get("height_scanner")
    default_joint_pos = robot.data.default_joint_pos.clone()
    policy = torch.jit.load(GO2_POLICY_PATH, map_location=sim.device).eval()
    leg_joint_ids, leg_joint_names = robot.find_joints(GO2_LEG_JOINT_ORDER, preserve_order=True)
    if leg_joint_names != GO2_LEG_JOINT_ORDER:
        raise RuntimeError(f"Unexpected Go2 leg joint order: {leg_joint_names}")
    leg_joint_ids = torch.tensor(leg_joint_ids, device=sim.device, dtype=torch.long)
    gr00t_joint_ids, gr00t_joint_names = robot.find_joints(GR00T_FULL_JOINT_ORDER, preserve_order=True)
    if gr00t_joint_names != GR00T_FULL_JOINT_ORDER:
        raise RuntimeError(f"Unexpected SO-Arm joint order: {gr00t_joint_names}")
    gr00t_joint_ids = torch.tensor(gr00t_joint_ids, device=sim.device, dtype=torch.long)
    gripper_link_idx = robot.find_bodies("gripper_link")[0][0]
    default_leg_joint_pos = default_joint_pos[:, leg_joint_ids].clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    last_action = torch.zeros((1, len(GO2_LEG_JOINT_ORDER)), device=sim.device)
    policy_decimation = 4
    policy_step = 0
    joint_targets = default_joint_pos.clone()
    latest_gr00t_action_deg = None
    latest_leader_action_deg = None

    keyboard = WasdKeyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=0.8,
            v_y_sensitivity=0.6,
            omega_z_sensitivity=1.2,
            sim_device=sim.device,
        )
    )
    keyboard.add_callback("K", keyboard.reset)
    gr00t_bridge = Gr00tZmqBridge(args_cli.gr00t_obs_port, args_cli.gr00t_action_port) if args_cli.enable_gr00t else None
    leader_action_sub = (
        ActionSubscriber(ZmqEndpointConfig(action_port=args_cli.gr00t_action_port)) if args_cli.leader_auto else None
    )
    leader_smoother = ActionSmoother(alpha=float(np.clip(args_cli.leader_sim_alpha, 0.0, 1.0)))
    if args_cli.leader_auto:
        start_leader_subprocess(args_cli)
    robot.set_joint_position_target(joint_targets)
    scene.write_data_to_sim()
    sim.play()  # 자동 재생 (Play 버튼 없이 시뮬레이션 시작)
    print(">>> Simulation: PLAYING (auto-started)", flush=True)

    print("\n>>> Go2 + SO-Arm Simulation Started! <<<")
    print(">>> MCP Extension: ENABLED")
    print(">>> Map: 기본 평면 (GroundPlane)")
    print(f">>> Camera Sensors: {'ENABLED (Wrist)' if args_cli.enable_cameras else 'DISABLED'}")
    print(f">>> Camera Viewports: {'ENABLED (Wrist)' if args_cli.enable_cameras and args_cli.show_camera_viewport else 'DISABLED'}")
    print(">>> SO-Arm Dynamics & Collision: URDF DEFAULT")
    print(">>> Mustard bottle: REMOVED")
    print(">>> Robot instanceable: FALSE (GUI 조작 가능)\n")
    print(">>> Manual Control: ENABLED")
    print(">>> RL Locomotion Policy: ENABLED")
    print(f">>> GR00T Bridge: {'ENABLED' if args_cli.enable_gr00t else 'DISABLED'}")
    print(f">>> Leader Teleoperation: {'ENABLED' if args_cli.leader_auto else 'DISABLED'}")
    print(f">>> Leader Action Apply: {'ENABLED' if args_cli.leader_auto and args_cli.gr00t_apply_actions else 'DISABLED'}")
    print(f">>> GR00T Action Apply: {'ENABLED' if args_cli.gr00t_apply_actions else 'DISABLED'}")
    print(f">>> Policy: {GO2_POLICY_PATH}")
    print(f">>> Leg joints: {', '.join(leg_joint_names)}")
    print(f">>> Observation joints ({robot.num_joints}): {', '.join(robot.joint_names)}")
    print(">>> Keys: hold W/S forward/back, hold A/D strafe, hold Q/E yaw, K stop\n")

    loop_count = 0
    while simulation_app.is_running():
        # 디버그: sim 재생 여부 + 키보드 명령 수신 확인 (200루프마다)
        if loop_count % 200 == 0:
            _kb_dbg = keyboard.advance()
            print(
                f"[dbg] is_playing={sim.is_playing()} loop={loop_count} "
                f"kb_vel={_kb_dbg.cpu().numpy().tolist()} norm={_kb_dbg.norm().item():.2f}",
                flush=True,
            )
        loop_count += 1
        if sim.is_playing():
            if leader_action_sub is not None:
                leader_action = leader_action_sub.receive()
                if leader_action is not None:
                    target_deg = clamp_joint_targets_deg(leader_action.joint_target_deg)
                    target_deg = leader_smoother.update(target_deg)
                    latest_leader_action_deg = target_deg.copy()
                    if args_cli.gr00t_apply_actions:
                        target_rad = torch.as_tensor(deg_to_rad(target_deg), device=sim.device, dtype=joint_targets.dtype)
                        joint_targets[:, gr00t_joint_ids] = target_rad
                    if args_cli.leader_action_log_every > 0 and policy_step % args_cli.leader_action_log_every == 0:
                        print(f">>> leader action applied deg: {target_deg}")

            if policy_step % policy_decimation == 0:
                vel_cmd_b = keyboard.advance().view(1, 3)
                if height_scanner is not None:
                    height_scan = height_scanner.data.pos_w[:, 2].unsqueeze(1) - height_scanner.data.ray_hits_w[..., 2] - 0.5
                    height_scan = torch.nan_to_num(height_scan, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    height_scan = torch.zeros((robot.data.root_lin_vel_b.shape[0], 0), device=sim.device)
                policy_obs = torch.cat(
                    (
                        robot.data.root_lin_vel_b,
                        robot.data.root_ang_vel_b,
                        robot.data.projected_gravity_b,
                        vel_cmd_b,
                        robot.data.joint_pos - default_joint_pos,
                        robot.data.joint_vel - default_joint_vel,
                        last_action,
                        height_scan,
                    ),
                    dim=-1,
                )
                if policy_obs.shape[-1] < GO2_POLICY_OBS_DIM:
                    height_scan_padding = torch.zeros(
                        (policy_obs.shape[0], GO2_POLICY_OBS_DIM - policy_obs.shape[-1]),
                        device=policy_obs.device,
                        dtype=policy_obs.dtype,
                    )
                    policy_obs = torch.cat((policy_obs, height_scan_padding), dim=-1)
                with torch.inference_mode():
                    last_action = policy(policy_obs)
                joint_targets = default_joint_pos.clone()
                joint_targets[:, leg_joint_ids] = default_leg_joint_pos + GO2_ACTION_SCALE * last_action
                if gr00t_bridge is not None:
                    action = gr00t_bridge.receive_action()
                    if action is not None:
                        action = clip_gr00t_action(action)
                        latest_gr00t_action_deg = action.copy()
                        print(f">>> GR00T action received ({'applied' if args_cli.gr00t_apply_actions else 'observed'}): {action}")
                active_arm_target_deg = latest_leader_action_deg if latest_leader_action_deg is not None else latest_gr00t_action_deg
                if active_arm_target_deg is not None and args_cli.gr00t_apply_actions:
                    action_rad = torch.as_tensor(
                        deg_to_rad(active_arm_target_deg),
                        device=sim.device,
                        dtype=joint_targets.dtype,
                    )
                    joint_targets[:, gr00t_joint_ids] = action_rad

            robot.set_joint_position_target(joint_targets)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)
            if gr00t_bridge is not None and args_cli.enable_cameras:
                wrist_out = scene.sensors["wrist_camera"].data.output
                if "rgb" in wrist_out:
                    wrist_rgb = wrist_out["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                    joint_pos_np = np.rad2deg(
                        robot.data.joint_pos[0, gr00t_joint_ids].detach().cpu().numpy().astype(np.float32)
                    )
                    gr00t_bridge.publish_observation(wrist_rgb, joint_pos_np)
                    if latest_gr00t_action_deg is not None and args_cli.gr00t_apply_actions and policy_step % 20 == 0:
                        joint_err_deg = latest_gr00t_action_deg - joint_pos_np
                        gripper_pos_w = robot.data.body_pos_w[0, gripper_link_idx].detach().cpu().numpy().astype(np.float32)
                        print(f">>> SO-Arm current deg: {joint_pos_np}")
                        print(f">>> SO-Arm target-current deg err: {joint_err_deg}")
                        print(f">>> gripper_link world pos: {gripper_pos_w}")
            if latest_leader_action_deg is not None and args_cli.leader_auto and policy_step % 20 == 0:
                joint_pos_np = np.rad2deg(
                    robot.data.joint_pos[0, gr00t_joint_ids].detach().cpu().numpy().astype(np.float32)
                )
                joint_err_deg = latest_leader_action_deg - joint_pos_np
                gripper_pos_w = robot.data.body_pos_w[0, gripper_link_idx].detach().cpu().numpy().astype(np.float32)
                print(f">>> SO-Arm current deg: {joint_pos_np}")
                print(f">>> SO-Arm target-current deg err: {joint_err_deg}")
                print(f">>> gripper_link world pos: {gripper_pos_w}")
            policy_step += 1
        else:
            # 일시정지 상태일 때는 물리 엔진 강제 업데이트(write_data)를 하지 않고 UI만 갱신
            simulation_app.update()

    stop_leader_subprocess()
    simulation_app.close()
    if gr00t_bridge is not None:
        gr00t_bridge.close()
    if leader_action_sub is not None:
        leader_action_sub.close()
    del camera_viewports


if __name__ == "__main__":
    main()
