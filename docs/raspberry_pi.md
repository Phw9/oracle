# Raspberry Pi Runbook

## Assumptions

`../tmi`에서 별도 라즈베리파이 하드웨어 스펙 파일은 찾지 못했습니다. 그래서 기본값은 Raspberry Pi 4/5급에서 USB 카메라 또는 libcamera로 노출된 카메라를 쓰는 보수적 설정입니다. 기본 설치 경로는 `/home/willtek/work/oracle`입니다.

## Camera

OpenCV가 카메라를 직접 열 수 있으면 기본 명령으로 충분합니다.

```bash
oracle-report capture --camera-index 0 --output-dir runs/pi-test
```

CSI 카메라가 OpenCV에 직접 노출되지 않으면 `libcamera-vid` 또는 Picamera2로 프레임을 넘기는 어댑터를 추가해야 합니다. 현재 코어 하네스는 `FrameSource` 경계가 분리되어 있어 Picamera2 소스를 별도 클래스로 붙이기 쉽습니다.

## Recommended Runtime Defaults

- 프레임 크기: 640x480
- 카메라 FPS: 15
- 얼굴 탐지 스케일: 0.5, 실제 cascade 입력 320x240
- 얼굴 탐지 주기: 2프레임마다 1회
- 얼굴 최소 크기: 96px
- 얼굴 유지 시간: 2.0초
- LLM 서버: `llama-server` on `http://127.0.0.1:8080/v1`
- LLM 이미지 입력: 멀티모달 GGUF 서버에서만 사용
- 미리보기: 현장 디버깅 때만 켜고, 키오스크 운영에서는 끄기

## Install

```bash
mkdir -p /home/willtek/work
cd /home/willtek/work
git clone <repo-url> oracle
cd /home/willtek/work/oracle
sudo apt-get update
sudo apt-get install -y python3-opencv libatlas-base-dev
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
cp .env.example .env
```

모델 다운로드를 나중에 하려면 클론할 때 LFS smudge를 끕니다.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url> oracle
cd /home/willtek/work/oracle
ORACLE_SKIP_MODEL_DOWNLOAD=1 ORACLE_SKIP_TESTS=1 ./build.sh
```

이 모드는 Git LFS의 `models/model.gguf.part01`, `models/model.gguf.part02`
다운로드를 건너뜁니다. 앱을 로컬 LLM으로 실행하려면 나중에 `git lfs pull
--include "models/model.gguf.part*"`를 실행하거나 다른 장비에서 준비한
`models/model.gguf`를 복사해야 합니다.

## llama.cpp Server

텍스트 전용 모델:

```bash
llama-server -m /home/willtek/work/oracle/models/model.gguf --host 127.0.0.1 --port 8080 -c 4096
```

비전 모델을 쓸 때는 해당 모델의 llama.cpp 문서에 맞춰 projector/mmproj 옵션을 함께 지정합니다. 텍스트 전용 서버에서는 `.env`에 아래 값을 둡니다.

```env
ORACLE_LLM_SEND_IMAGE=0
```

## Service Example

```bash
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo cp systemd/oracle-report.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl enable oracle-report
sudo systemctl start llama-server
sudo systemctl start oracle-report
```

서비스 파일은 기본적으로 `/home/willtek/work/oracle`과 `User=willtek`을 기준으로 작성되어 있습니다. 다른 위치나 계정을 쓰면 `systemd/*.service`의 `WorkingDirectory`, `User`, 모델 경로를 바꿔야 합니다.
