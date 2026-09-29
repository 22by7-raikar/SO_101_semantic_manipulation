# Reproducible development environment

This setup implements the software foundation for the supplied SO-101 project proposal: calibrated RGB-D geometry, language-conditioned cost fields, planning research, and LeRobot baselines. It does not implement the research planner or enable physical execution.

## Supported machines and profiles

| Profile | Where | Included |
| --- | --- | --- |
| `core` | Apple Silicon Mac or Ubuntu x86_64 | NumPy/SciPy, headless OpenCV with ArUco, plotting, trimesh, URDF loading, MuJoCo, tests and lint |
| `vision` | Apple Silicon Mac or Ubuntu x86_64 | Core plus PyTorch/torchvision, Transformers, Accelerate and safetensors |
| `robot` | Ubuntu x86_64 only | Vision plus LeRobot with Feetech and SmolVLA dependencies, RealSense, serial I/O and video decoding |

Both people use their own clone and `.venv`, including when logged into the same Ubuntu laptop. `uv.lock` fixes transitive versions as well as direct packages. Python is pinned to 3.11.15 and uv to 0.11.14. Do not combine these environments with an activated Conda environment or use `sudo pip`.

The Linux ML profile uses PyTorch **2.7.1 + CUDA 11.8**, torchvision **0.22.1**, and LeRobot **0.4.4**, with TorchCodec **0.4.0**. This is a deliberately fixed compatibility set, not a request to track latest LeRobot. Apple Silicon uses the PyPI PyTorch build for CPU/MPS development; CUDA workloads run on Ubuntu. Windows, Intel Mac, and Linux ARM are not covered by this lock.

## 1. Ubuntu system preparation (administrator, once)

Check the host before changing it:

```bash
cat /etc/os-release
uname -m
nvidia-smi
```

Ubuntu 20.04 is past standard support. Prefer a planned migration to Ubuntu 22.04/24.04 LTS after checking hardware compatibility, or arrange Ubuntu Pro/ESM while retaining 20.04. Do not upgrade the OS or NVIDIA driver during another person's job.

Install ordinary runtime tools:

```bash
sudo apt update
sudo apt install git curl ca-certificates tmux build-essential pkg-config ffmpeg libgl1 libglib2.0-0 libegl1 libusb-1.0-0
```

Build tools cover small native transitive dependencies such as Linux input support; the Python interpreter and headers come from uv.

The NVIDIA driver must work before Python can use CUDA. A maintained driver supporting CUDA 11.8 (520.61.05 or newer as a conservative target) is expected. Use Ubuntu's Additional Drivers workflow if the driver is missing; reboot with someone present and rerun `nvidia-smi`. Do not install a system CUDA toolkit just for these binary PyTorch wheels: they supply the CUDA runtime dependencies. Custom CUDA extensions would require a separate, matched compiler/toolkit setup.

FFmpeg shared libraries are required by TorchCodec; the pinned version supports FFmpeg major versions 4–7. Ubuntu 20.04's distribution FFmpeg is within that range. Installing a newer, unrelated TorchCodec can break its PyTorch ABI compatibility. `so101-doctor --hardware` checks the actual import on your host.

## 2. Install uv (each contributor, once per machine)

Download the pinned installer and inspect it before executing:

```bash
curl -LsSf https://astral.sh/uv/0.11.14/install.sh -o /tmp/install-so101-uv.sh
less /tmp/install-so101-uv.sh
sh /tmp/install-so101-uv.sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Press `q` to exit `less`. Follow the installer's shell startup instructions so `uv` remains on PATH. The same process works on the Mac and Ubuntu. `uv sync` obtains the pinned Python automatically if needed; it does not replace Ubuntu's system Python.

## 3. Install your profile

Inside your own clone:

```bash
# Mac or Ubuntu, geometry and planning first:
bash scripts/setup.sh core

# OR vision development (downloads additional ML packages):
bash scripts/setup.sh vision

# OR the complete GPU/hardware environment, on Ubuntu:
bash scripts/setup.sh robot
```

Choose one command. The setup script syncs exactly from the lockfile and checks the selected profile (including CUDA and hardware-library imports for `robot`). GPU packages take several GB of disk space and can take time to download. No model weights are downloaded by setup.

Then run:

```bash
uv run --no-sync pytest -q
uv run --no-sync so101-demo
```

The demo creates `outputs/demo/cost_field.npz` and `cost_field.png`, using synthetic depth and synthetic semantic scores. It works offline, does not open devices, and does not command motors. It proves package execution and the proposed interchange format, not perception quality or collision safety.

On the Ubuntu robot profile, also run:

```bash
uv run --no-sync so101-doctor --gpu --hardware
# With the RealSense physically attached:
uv run --no-sync so101-doctor --camera
```

Treat a failing diagnostic as an unresolved setup issue. No remote GPU or physical hardware was available while authoring this repository setup; those checks must pass on your laptop.

**Use `uv run --no-sync` after selecting a profile.** Plain `uv run` or a bare `uv sync` may synchronize back to the default core profile and remove optional packages. To change profiles, rerun the setup script. Alternatively activate `.venv` and run its executables directly:

```bash
source .venv/bin/activate
python -c 'import numpy; print(numpy.__version__)'
```

## 4. RealSense and SO-101 host permissions

Connect camera and motor USB interfaces to Ubuntu, not the Mac. The camera model and firmware must be checked on the actual device. Use a USB 3 data cable.

An administrator can grant serial access to the two real usernames:

```bash
sudo usermod -aG dialout akshay
sudo usermod -aG dialout teammate
```

Both users must log out and reconnect afterward. Verify with `id` and inspect stable port names with `ls -l /dev/serial/by-id/`. Avoid global `chmod 777` device permissions and do not run robot processes as root.

For RealSense USB access, obtain the rules from the SDK version corresponding to the Python wrapper:

```bash
curl -fL https://raw.githubusercontent.com/realsenseai/librealsense/v2.56.5/config/99-realsense-libusb.rules -o /tmp/99-realsense-libusb.rules
less /tmp/99-realsense-libusb.rules
sudo install -m 644 /tmp/99-realsense-libusb.rules /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Review any existing rules before replacing them. Reconnect the camera and run the camera diagnostic again. If streaming still fails, check SDK/firmware/kernel compatibility using the official RealSense instructions; a Python wheel alone does not verify device compatibility.

Use the [LeRobot 0.4.4 SO-101 instructions](https://huggingface.co/docs/lerobot/v0.4.4/so101) for motor setup and calibration. CLI tools are installed in `.venv/bin`; for example, `uv run --no-sync lerobot-find-port`. That tool is interactive and may ask you to disconnect/reconnect a cable. Do not run hardware setup while another person controls the robot.

LeRobot keyboard-driven teleoperation may require a local graphical session; SSH is not a substitute for local operator input. If a headless process fails importing `pynput`, use the appropriate local desktop session or a deliberately designed headless control path. Do not fake a display merely to bypass an operator-input requirement.

Copy `configs/lab.example.yaml` to `configs/lab.local.yaml` and fill measured calibration, serial paths and a verified model path. Null fields are intentional. The file is a shared configuration template for future modules, not automatically loaded by LeRobot.

## 5. Tools for daily work

- Use `tmux` for remote jobs; see [remote access](REMOTE_GPU_SETUP.md).
- Use a local editor or VS Code Remote SSH, selecting the remote clone's `.venv/bin/python`.
- Run `uv run --no-sync ruff check .`, `uv run --no-sync ruff format --check .`, and `uv run --no-sync pytest` before pushing.
- A GitHub Actions workflow checks the core environment on Linux and Apple Silicon. It cannot validate attached robots or CUDA hardware.
- Each contributor uses their own Git branch, Hugging Face credentials and model cache. Do not commit tokens, camera recordings or weights.

Optional notebooks on Ubuntu (preserve your chosen extras):

```bash
uv sync --locked --extra vision --extra hardware --extra notebooks
uv run --no-sync jupyter lab --ip=127.0.0.1 --no-browser --port=8888
```

From the Mac, in another terminal:

```bash
ssh -N -L 8888:127.0.0.1:8888 so101-gpu
```

Open the authenticated localhost URL printed by Jupyter. Use separate ports when both users run servers. For a Mac core notebook environment, use `uv sync --locked --extra notebooks` instead.

## 6. Scope and project choices

Start with saved RGB-D frames, measured calibration and geometry-only planning. The core includes SciPy optimization, URDF loading via yourdfpy and MuJoCo for simulation work. A robot model is not bundled: agree on a licensed SO-101 model, pin its source commit, verify joint order/limits/meshes and compare FK with the real arm before planning motion.

Transformers supplies model interfaces for Grounding DINO, SAM 2, SigLIP and DINOv2. Choose and pin a model ID and full revision commit when implementing the first perception module. Downloads may require a model license or account. No checkpoint is silently chosen or downloaded here.

PaliGemma/large VLMs, quantization, OMPL/TrajOpt and flow-matching extensions are deferred until the initial geometry pipeline works. OMPL is a planning library, not itself a trajectory optimizer. Avoid installing every candidate stack into the shared environment. Add a chosen dependency with a tested pin and regenerate the lock in a reviewed change.

LeRobot's ACT and SmolVLA dependencies are available in the robot profile. This does not establish that training fits in 6 GB. Start with batch size 1, measure peak VRAM, use frozen models/cached features, and keep one heavy GPU job active at a time. A model's weights, activations and optimizer states all consume memory.

A numerical optimizer and collision checker do not provide unconditional physical safety guarantees. Calibration error, depth holes, model mismatch and controller behavior still matter. Keep execution disabled until limits, collision checking, operator supervision and stop behavior are verified.

## 7. Updating dependencies together

Both contributors should use the same Git commit and lockfile. Normal setup uses `--locked`, which refuses a stale lock. To intentionally change dependencies, edit `pyproject.toml`, run `uv lock` with the pinned uv version, reinstall the relevant profiles, and repeat core and Ubuntu diagnostics. Commit the manifest and lockfile together. Avoid `pip install -U` or `uv lock --upgrade` as routine setup steps.

For reproducible experiments record Git SHA, model revision, camera calibration revision, dataset version, seed, package lock and NVIDIA driver version. OS packages and drivers are host prerequisites and are not captured by `uv.lock`.

## References

- [uv project dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [PyTorch published version combinations](https://pytorch.org/get-started/previous-versions/)
- [LeRobot 0.4.4 dependency manifest](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml)
- [TorchCodec 0.4.0 installation](https://github.com/pytorch/torchcodec/tree/v0.4.0)
- [RealSense SDK 2.56.5](https://github.com/realsenseai/librealsense/tree/v2.56.5)
