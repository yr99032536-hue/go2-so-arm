# Go2 SO-Arm 리더 암 텔레오퍼레이션 기록

## 문제

1. 손목 카메라가 초기 위치에 고정된 것처럼 보였다.
2. 실제 리더 암을 움직여도 Isaac Sim의 SO-Arm이 따라오지 않았다.
3. 같은 Isaac Sim 실행에서 강화 학습 보행 정책이 다시 꺼진 것처럼 로봇이 엎드리고 제대로 걷지 못했다.

## 적용한 방향

손목 카메라는 월드에 직접 붙인 카메라처럼 쓰지 않고, `gripper_link` 아래에 붙은 카메라 prim으로 구성했다. 그래서 로봇 팔 링크가 움직이면 카메라 pose도 링크를 따라 움직인다.

리더 암 텔레오퍼레이션은 GR00T action path와 분리했다. 실제 리더 암 브리지가 ZMQ로 보내는 관절 목표를 Isaac Sim 쪽에서 별도 subscriber로 받고, 그 값을 SO-Arm 관절 target에 계속 적용한다.

보행 정책은 그대로 켠다. 단, 보행 정책은 주기적으로 전체 joint target을 갱신하므로, leg target을 policy 결과로 갱신한 뒤 arm target을 리더 암 목표로 다시 적용한다.

## 원인 분석

텔레오퍼레이션이 안 된 직접 원인은 크게 두 가지였다.

1. `--leader_auto`가 실제 arm action apply까지 켜지 못해 리더 암 값이 target으로 들어가지 않았다.
2. 이전 `leader_teleop_bridge.py` 프로세스가 남아 같은 ZMQ port를 점유할 수 있었다. 이 경우 새 브리지가 뜬 것처럼 보여도 실제 action stream이 원하는 프로세스에서 나오지 않는다.

보행이 무너진 원인은 보행 정책 자체가 꺼졌기 때문이라기보다, 팔 target이 시뮬레이션에 강하게 들어오면서 로봇의 동역학 조건이 바뀐 쪽에 가깝다.

여기서 "팔 target이 강하게 들어왔다"는 말은 무게 옵션이 새로 켜졌다는 뜻이 아니다. 실제 리더 암의 현재 자세가 시뮬레이션 SO-Arm의 목표 자세로 계속 들어오면서, 팔이 원래 학습 때와 다른 자세를 유지하게 되었다는 뜻이다. 그 결과 무게 중심과 관성 조건이 달라지고, 기존 보행 정책이 익숙한 상태 분포에서 벗어날 수 있다.

정리하면, 원래 보행 정책은 팔의 특정 자세 범위 또는 학습 중 본 팔 자세 분포에 맞게 안정화되어 있었는데, 리더 암 자세가 그대로 들어오면서 팔 위치가 달라졌고 그 변화가 몸통 균형과 발 접촉 안정성에 영향을 준 것이다.

## 확인된 정상 로그

```text
Leader Teleoperation: ENABLED
Leader Action Apply: ENABLED
RL Locomotion Policy: ENABLED
leader action applied deg
SO-Arm current deg
SO-Arm target-current deg err
```

## 운영 메모

- 리더 암 텔레오퍼레이션 테스트 전에는 stale bridge process를 정리한다.
- 팔 목표가 들어오는 동안 보행이 불안정하면, 먼저 팔 자세를 학습 분포에 가까운 중립 자세로 맞춘다.
- 이후 안정화를 위해 arm target smoothing, arm motion limit, locomotion policy retraining 또는 domain randomization을 고려한다.
