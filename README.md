# EBiM Task 2 — Thermal Pad Placement

A scripted policy for EBiM Benchmark Task 2 (thermal-pad placement) in the
room scene. It resolves the randomized target slot from the evaluator's
loose-bbox stream and runs a pick–carry–place–sweep sequence with an accept
ladder, each completed attempt judged by the official `evaluate` service.
Packaged as a ROS 2 (Jazzy) Docker image that talks to the benchmark's Isaac
Sim scene over the `/isaac/*` topic contract.

## Prerequisites: scene launch

Copy the bridge data contract into the benchmark before launching:

```bash
cp config/bridge/fr3duo_mobile_task2_data_contract.yaml \
   <benchmark>/task2_isaacsim/assets/embodiments/fr3duo_mobile_task2/data_contract.yaml
```

Launch the official task2 room scene in recording mode with exactly these
flags:

```bash
cd <benchmark>/task2_isaacsim && /isaac-sim/python.sh scripts/scene_room.py \
  --record --headless \
  --no-spine-keyboard-control \
  --franka-root <benchmark>/task2_isaacsim \
  --embodiment fr3duo_mobile_task2 \
  --robot-x 2.1 --robot-y 3.05 --robot-z 0.0 --robot-yaw -90.0
```

Gate readiness on the bridge banner (`room bridge started`), then start the
container.

## Build & run

```bash
docker build -t ebim-task2-autonomy:latest .
docker run --rm --network host --ipc host ebim-task2-autonomy:latest
```

- The image's default CMD is the complete entry: `python3 -u -m ebim_task2.official_run`.
- Compose alternative: `docker compose run --rm policy python3 -u -m ebim_task2.official_run` (compose sets host network/IPC).
- Env: `ROS_DOMAIN_ID` must match the simulator's (default 0; pass `-e ROS_DOMAIN_ID=<n>` otherwise). `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` are baked into the image.
- `--attempts N` — maximum chain attempts (default 36; env `EBIM_RUN_ATTEMPTS`).
- `--budget-min M` — wall-clock budget in minutes; no new attempt starts past it (default 240; env `EBIM_RUN_BUDGET_MIN`).
- `--stop-iou X` — fixed accept threshold instead of the built-in ladder.

## Test

```bash
PYTHONPATH=src python3 -m pytest tests/
```

Six tests import `rclpy` and require the ROS 2 Jazzy environment; run the
suite inside the Docker image to include them.

## Model weights (optional VLA route)

The scripted policy is the default entry and needs no weights. The optional
VLA route loads a fine-tuned checkpoint published on HuggingFace:

```bash
hf download ByteMelodist/pi05_allslot_st3 --local-dir <dir>
```

Build the VLA runtime image (torch + lerobot):

```bash
docker build -f Dockerfile.train -t task2-trainer:latest .
docker build -f Dockerfile.vla   -t ebim-task2-vla:latest .
```

Pass `--build-arg TORCH_CUDA_CHANNEL=cu130` to the first build for sm_120
GPUs. Point `vla.checkpoint` in `config/task2.vla.yaml` at `<dir>` (or set
`EBIM_VLA_CHECKPOINT=<dir>`), then run
`python3 -m ebim_task2.runner --config /opt/ebim-task2/config/task2.vla.yaml --policy vla`.

## License

Apache-2.0 (see `LICENSE`).
