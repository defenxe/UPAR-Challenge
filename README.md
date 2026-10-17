# UPAR ViT-L/14 학습 안내

이 안내에서는 QueryBalanced ViT-L/14 모델과 그 확장 모델인 Frozen MultiScale Local Attribute를 순서대로 학습합니다. 파이프라인이 공식 CLIP 가중치를 다운로드하고 검증한 뒤 사용 가능한 GPU 하나를 선택합니다. 첫 번째 학습에서 완료된 Epoch 2 Checkpoint와 메타데이터를 두 번째 모델에 자동으로 연결합니다.

## 빠른 시작

```bash
git clone https://github.com/defenxe/UPAR-Challenge.git
cd UPAR-Challenge

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
  torch==1.12.1+cu113 torchvision==0.13.1+cu113
python -m pip install -r IRRA-QueryBalanced-ViT-L14/requirements.txt \
  -r IRRA-QB-ViT-L14-FrozenMultiScaleLocalAttr/requirements.txt

# 데이터셋을 ./data/에 준비하거나 UPAR_DATA_ROOT에 데이터 경로를 지정하세요.
# 선택 사항: 특정 유휴 GPU를 사용하려면 UPAR_GPU_UUID=GPU-...를 지정하세요.

bash scripts/train_pipeline.sh download
bash scripts/train_pipeline.sh smoke
bash scripts/train_pipeline.sh querybalanced
bash scripts/train_pipeline.sh frozen
```

`querybalanced`는 학습 직전에 GPU와 데이터셋을 다시 검사합니다. 이 검사까지 포함해 전체 단계를 한 번에 실행하려면 `bash scripts/train_pipeline.sh all`을 사용하세요.

저장소 바깥에 데이터가 있다면 명령을 실행하기 전에 경로를 한 번 지정하세요.

```bash
export UPAR_DATA_ROOT=/path/to/UPAR-data
# 선택 사항: 생략하면 파이프라인이 유휴 GPU 하나를 자동 선택합니다.
export UPAR_GPU_UUID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

같은 값은 각 명령에 `--data-root PATH`, `--gpu UUID` 옵션으로 전달할 수도 있습니다.

## 요구 사항

- Linux, Bash, Python 3.7–3.10, CUDA 11.3과 호환되는 NVIDIA 드라이버, `nvidia-smi`가 필요합니다.
- 설치 명령은 검증된 CUDA wheel 조합인 PyTorch 1.12.1 / torchvision 0.13.1 (CUDA 11.3)을 고정합니다. 모델 패키지를 설치하기 전에 이 조합을 설치하세요. 테스트한 드라이버 470.182.03에서는 기본 패키지 해석으로 설치된 PyTorch 1.13.1+cu117이 CUDA 사용 가능 상태를 보고하면서도 GPU UUID로 선택하면 GPU를 하나도 노출하지 않았습니다. 고정한 조합은 같은 검사에서 통과했습니다. 이 조합은 [공식 PyTorch 이전 버전 안내](https://pytorch.org/get-started/previous-versions/)에 있습니다.
- 여유 메모리가 11,000 MiB 이상인 유휴 NVIDIA GPU 하나가 필요합니다. Frozen 모델은 GPU UUID, 여유 메모리, Batch 크기 32와 CUDA 사전 검사를 그대로 수행합니다.
- 나머지 Python 패키지는 각 모델의 `requirements.txt`에 기재되어 있습니다.
- QueryBalanced의 Epoch 1–10 Checkpoint에 약 20 GB가 필요합니다. Frozen의 Checkpoint와 로그를 저장할 공간도 추가로 확보하세요.

파이프라인은 UUID로 유휴 GPU 하나를 선택하고 해당 UUID를 CUDA와 학습 코드에 전달합니다. GPU를 직접 선택하려면 `UPAR_GPU_UUID`를 지정하거나 명령에 `--gpu GPU-...`를 추가하세요.

## 데이터셋 구조

기본 데이터 경로는 저장소 루트의 `data/`입니다. 다른 위치를 사용하려면 `UPAR_DATA_ROOT` 또는 `--data-root`를 지정하세요. 파이프라인은 학습 전에 CSV 파일, Attribute 열, Query 연결, 참조 이미지 파일을 검사합니다.

```text
data/
├── annotations/task2/train/
│   ├── queries.csv
│   ├── gt.csv
│   └── ids.csv
├── annotations/task2/val/
│   ├── queries.csv
│   ├── gt.csv
│   └── ids.csv
└── <gt.csv에서 참조하는 이미지>
```

`gt.csv`의 이미지 경로는 데이터 루트 기준 상대경로여야 합니다. 각 split에는 공식 40개 Attribute의 Query 및 정답 열이 있어야 하며, 참조되는 모든 이미지 파일이 존재해야 합니다.

## 모델 개요

**IRRA-QueryBalanced-ViT-L14**는 공식 OpenAI CLIP ViT-L/14에서 시작합니다. 기존 QueryBalanced sampler와 SDM+MAP Loss를 사용해 FP32로 60 Epoch, 유효 Batch 크기 32로 학습합니다. Optimizer, Scheduler, Sampling, Augmentation, Loss 설정은 `IRRA-QueryBalanced-ViT-L14/configs/train_fp32_micro4.json`에 그대로 있습니다. 완료된 Epoch 1–10의 모델 가중치를 저장합니다.

**IRRA-QB-ViT-L14-FrozenMultiScaleLocalAttr**는 완료된 QueryBalanced Epoch 2 모델을 불러옵니다. 기존 모델을 고정하고 multiscale/local Attribute 모듈만 FP32로 5 Epoch 학습합니다. Batch 크기 32, 기존 weighted BCE, Adam, cosine Scheduler 설정을 사용하며 설정은 `IRRA-QB-ViT-L14-FrozenMultiScaleLocalAttr/configs/frozen_localattr_e5.json`에 있습니다.

## 1단계: QueryBalanced 학습

공식 OpenAI CLIP ViT-L/14 파일을 다운로드하고 고정된 SHA-256을 검증합니다.

```bash
bash scripts/train_pipeline.sh download
```

전체 학습 전에 Smoke Test를 실행하세요. 의존성을 불러오고 두 split과 참조 이미지를 검사하며, 공식 가중치 Hash, 선택된 GPU UUID와 메모리를 확인한 뒤 CUDA에서 해당 GPU 하나만 보이는지 검사합니다. 학습 Checkpoint는 만들지 않습니다.

```bash
bash scripts/train_pipeline.sh smoke
```

QueryBalanced 학습을 시작합니다.

```bash
bash scripts/train_pipeline.sh querybalanced
```

기존 60 Epoch 설정을 사용해 학습을 실행합니다. 완료된 Run은 고유 ID와 함께 `IRRA-QueryBalanced-ViT-L14/experiments/` 아래에 기록됩니다. 기존 Run 디렉터리를 재사용하지 않습니다.

## 2단계: FrozenMultiScaleLocalAttr 학습

QueryBalanced 학습이 끝난 뒤 다음 명령을 실행합니다.

```bash
bash scripts/train_pipeline.sh frozen
```

파이프라인은 이 파이프라인으로 만들어진 완료된 QueryBalanced Run 중 가장 최근 Run을 선택합니다. 해당 Run의 `pipeline.json`, `config.json`, `status.json`, `train.log`, `metrics.jsonl`을 확인하고 Epoch 2 Checkpoint의 SHA-256을 계산합니다. 그런 다음 Run별 Frozen Config에 Checkpoint 정보와 GPU UUID를 기록합니다. 기존 CUDA Batch-32 사전 검사와 Checkpoint 출처 검증을 통과해야 Frozen 학습을 시작합니다.

## Checkpoint 및 결과

- 공식 CLIP 가중치: `IRRA-QueryBalanced-ViT-L14/weights/ViT-L-14.pt`
- QueryBalanced Run: `IRRA-QueryBalanced-ViT-L14/experiments/full_*/`. 완료된 Run에는 `epoch_001.pth`–`epoch_010.pth`, `config.json`, `status.json`, `train.log`, `metrics.jsonl`, `metrics.csv`, 출처 기록 `pipeline.json`이 있습니다.
- Frozen Run: `IRRA-QB-ViT-L14-FrozenMultiScaleLocalAttr/experiments/frozen_pipeline_*/`. 최상위에 생성된 Config와 사전 검사 보고서가 있으며 `run/` 아래에 `epoch_001.pth`–`epoch_005.pth`, 지표, 상태, Manifest와 학습 로그가 저장됩니다.
- Epoch 2의 원본 경로, SHA-256, QueryBalanced Run ID, 데이터 루트와 GPU UUID가 자동으로 기록됩니다. 모델 Checkpoint는 일반 `state_dict` 형식입니다.

가장 최근 Run의 상태를 확인합니다.

```bash
cat IRRA-QueryBalanced-ViT-L14/experiments/full_*/status.json
cat IRRA-QB-ViT-L14-FrozenMultiScaleLocalAttr/experiments/frozen_pipeline_*/run/status.json
```

## 문제 해결

- **데이터셋을 찾을 수 없음:** `data/`에 데이터셋을 두거나 `UPAR_DATA_ROOT`를 데이터 루트로 설정하세요. `annotations/task2/{train,val}/`와 모든 이미지 경로가 있는지 확인하세요.
- **GPU를 선택할 수 없음:** `nvidia-smi`를 실행해 상태를 확인하세요. 여유 메모리가 11,000 MiB 이상인 NVIDIA GPU를 확보하거나 `UPAR_GPU_UUID` / `--gpu`로 UUID를 지정하세요.
- **`nvidia-smi`에는 GPU가 보이지만 CUDA에는 0개로 표시됨:** 설치된 버전이 `torch 1.12.1+cu113`, `torchvision 0.13.1+cu113`인지 확인하세요. 빠른 시작의 고정 설치 명령을 다시 실행한 뒤 `bash scripts/train_pipeline.sh smoke`를 재실행하세요.
- **Frozen에서 원본 Run을 찾을 수 없음:** `bash scripts/train_pipeline.sh querybalanced`를 완료하고 해당 `status.json`에 `complete`, 60 Epoch가 기록됐는지 확인하세요.
- **CLIP Hash가 일치하지 않음:** 파이프라인은 기존 파일을 덮어쓰지 않습니다. 잘못된 `ViT-L-14.pt`를 다른 곳으로 옮긴 뒤 `bash scripts/train_pipeline.sh download`를 다시 실행하세요.
- **사전 검사 실패:** Frozen 학습은 시작되지 않습니다. 보고된 GPU, 데이터, Checkpoint 또는 CUDA 문제를 해결하고 `frozen` 명령을 다시 실행하세요. 재시도할 때마다 새로운 출력 디렉터리를 만듭니다.
- **학습 중단:** 완료된 Epoch의 Checkpoint와 로그는 고유 Run 디렉터리에 보존됩니다. QueryBalanced Trainer는 Optimizer/Scheduler 상태를 이용한 정확한 재개를 지원하지 않습니다.

모델별 README에서도 이 안내를 참조할 수 있습니다. 실험 설정은 기존 모델 Config와 학습 코드에 있으며, 파이프라인은 경로 확인, 입력 검증, GPU 선택, Run 메타데이터와 Checkpoint 연결을 담당합니다.
