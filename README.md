# Vision-Based Robotic Hand Control

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-97CA00)](LICENSE)
![MediaPipe](https://img.shields.io/badge/Tracking-MediaPipe-0097A7)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Arduino Mega](https://img.shields.io/badge/Hardware-Arduino%20Mega-00878F?logo=arduino&logoColor=white)
![Local processing](https://img.shields.io/badge/Processing-Local-7952B3)

Real-time hand tracking and servo control for a robotic hand, using a webcam, MediaPipe hand landmarks, and an Arduino Mega. The current implementation estimates finger flexion and wrist orientation geometrically, maps calibrated poses to servo angles, and sends commands over serial.

This README describes the latest integrated implementation on `main`. This is a research prototype; accuracy, latency, and hardware reliability have not been established through a comparative benchmark.

## How it works

1. OpenCV captures frames from a webcam.
2. MediaPipe detects one hand and its landmarks.
3. A fitted palm plane provides a reference for finger angles, with an offset reference for the thumb.
4. Open-hand and fist calibration map those angles to mechanical servo ranges.
5. Threshold-based quantization and keepalive messages control serial updates.
6. Arduino firmware applies angle limits and limits motion to four degrees per loop.

The current Python controller uses raw angles rather than an EMA or One Euro filter. Recorded smoothing experiments are supplied separately.

## Project layout

| Folder | Contents |
| --- | --- |
| [`src/`](src/) | Current webcam and serial controller |
| [`firmware/`](firmware/) | Hand controller and standalone servo test sketches |
| [`config/`](config/) | Example calibration; local calibration is generated here |
| [`analysis/`](analysis/) | Smoothing experiment analysis scripts |
| [`data/experiments/`](data/experiments/) | Recorded raw and quantized trial data |
| [`results/smoothing/`](results/smoothing/) | Existing experiment figures and original ZIP |
| [`docs/`](docs/) | Hardware, protocol, and experiment notes |
| [`archive/`](archive/) | Reference versions and snapshot provenance |

## Installation

Use Python 3.10 and a webcam. The pinned MediaPipe version supplies the `mp.solutions.hands` API used by this code. The runtime pins follow the repository's later hand-tracking environment; this combination has not been revalidated on every operating system.

```bash
git clone https://github.com/Hugo132645/gesture_recognition.git
cd gesture_recognition
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## Run without hardware

From the repository root:

```bash
python src/vision_based_control.py --dry-run --camera-index 0
```

Add `--show-3d` for the landmark view. `--dry-run` disables serial output while retaining webcam tracking and calibration.

With the webcam window focused:

| Key | Action |
| --- | --- |
| `o` | Capture an open hand |
| `f` | Capture a fist and update calibration after an open pose |
| `v` | Toggle verification overlay |
| `r` | Reset in-memory calibration |
| `w` | Save calibration |
| `q` / Escape | Quit |

Capture both poses for your own hand, then press `w`. Calibration is stored at `config/servo_calib.json`, regardless of the working directory. The supplied `servo_calib.example.json` is a reference recording, not a universal calibration.

## Arduino integration

Open `firmware/hand_control/hand_control.ino` in Arduino IDE, select the Arduino Mega and its port, and upload using the Servo library. Review the [pin mapping and protocol](docs/hardware.md) before running with connected motors.

```bash
python src/vision_based_control.py --serial-port COM4 --baud 115200
```

Replace `COM4` with your actual device port, such as `/dev/ttyACM0` on Linux. The controller falls back to tracking without serial if the port cannot be opened; check the console rather than assuming hardware is connected.

**Current integration limitation:** the firmware waits for `<CALIB_DONE>` before moving the servos, but the Python controller does not emit that message. Capturing or saving calibration alone does not unlock the firmware. The original control behavior has been retained; completing this handshake is required for end-to-end hardware operation.

## Recording and analysis

```bash
python src/vision_based_control.py --dry-run --log outputs/session1
```

This writes paired `_raw.csv` and `_quant.csv` files. Logging occurs when the controller decides to send an update, rather than for every captured frame. See [experiment notes](docs/experiments.md) for analysing the included trials and interpreting the figures.

## Limitations

Tracking depends on visibility, lighting and camera placement. Only one hand is tracked. The firmware drives five fingers and one wrist servo; the seventh transmitted value is wrist pitch and is not applied to a servo. The included figures are historical experiment outputs and do not establish performance of the current raw-angle controller.

## License

[MIT](LICENSE).
