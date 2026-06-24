# Go2 SO-Arm Leader Teleoperation

Isaac Sim에서 Go2 보행 정책을 유지하면서 실제 SO-Arm 리더 암의 관절 목표를 시뮬레이션 SO-Arm에 적용하기 위한 초기 프로젝트 스냅샷입니다.

## 포함 파일

- `src/sim/go2_soarm.py`: Go2 + SO-Arm Isaac Sim 실행 스크립트
- `soarm_nbv/leader_teleop_bridge.py`: 실제 리더 암을 읽어 ZMQ action으로 보내는 브리지
- `soarm_nbv/zmq_bridge.py`: ZMQ 송수신 유틸리티
- `soarm_nbv/safety.py`: 관절 제한 및 단위 변환 유틸리티
- `soarm_nbv/start_teleop.sh`: 리더 암 텔레오퍼레이션 실행 스크립트
- `docs/teleoperation-notes.md`: 적용 내용과 원인 분석 기록

## 실행

기존 Isaac Sim 환경에서 아래처럼 실행합니다.

```bash
/home/iy/miniconda3/envs/isaacsim-5.1/bin/python -u src/sim/go2_soarm.py \
  --enable_cameras \
  --show_camera_viewport \
  --leader_auto
```

또는 기존 프로젝트 경로에서:

```bash
/home/iy/Isaac/Robotics/robot_models/soarm_nbv/start_teleop.sh
```

## 핵심 변경

- `--leader_auto` 실행 시 리더 암 action 적용을 자동 활성화합니다.
- 리더 암 브리지를 GR00T 브리지와 분리해서 별도 `ActionSubscriber`로 받습니다.
- 매 시뮬레이션 step마다 최신 리더 암 목표를 SO-Arm 관절 target에 즉시 반영합니다.
- RL 보행 정책이 leg target을 갱신한 뒤에도 arm target을 다시 덮어써서 팔 명령이 사라지지 않게 했습니다.
- 예전 브리지 프로세스가 남아 있으면 같은 ZMQ port를 잡아 텔레오퍼레이션이 먹히지 않으므로, 실행 전 stale leader bridge를 정리합니다.
- 손목 카메라는 SO-Arm mimic 데이터셋 생성 스크립트의 방식에 맞춰 gripper link 하위에 고정하고 wrist camera offset/config를 조정했습니다.

## 빠른 검증

```bash
python3 -m py_compile \
  src/sim/go2_soarm.py \
  soarm_nbv/leader_teleop_bridge.py \
  soarm_nbv/zmq_bridge.py \
  soarm_nbv/safety.py

bash -n soarm_nbv/start_teleop.sh
```

실행 로그에서 아래 메시지를 확인합니다.

```text
Leader Teleoperation: ENABLED
Leader Action Apply: ENABLED
RL Locomotion Policy: ENABLED
leader action applied deg
SO-Arm current deg
SO-Arm target-current deg err
```
