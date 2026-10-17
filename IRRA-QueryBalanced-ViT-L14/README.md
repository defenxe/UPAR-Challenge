# IRRA-QueryBalanced-ViT-L14

`IRRA-QueryBalanced` 원본에서 필요한 코드만 독립 복사한 ViT-L/14 실험입니다. 다른 실험의 학습 코드, LODO/Ensemble, MultiScale Local Attribute, Ordinal을 사용하지 않습니다. 로컬의 공식 OpenAI foundation archive만 SHA 검증 후 독립 복사했습니다. 학습된 다른 모델 Weight를 사용하지 않습니다.

## 기존 대비 변경

- CLIP ViT-B/16 → 공식 OpenAI ViT-L/14. Vision 24 blocks / width1024 / heads16 / patch14; Text 12 blocks / width768 / heads12; joint projection768.
- 입력 H×W 384×128 → **378×126**, 기존과 같은 3:1 종횡비. Vision position grid16×16 →27×9, 기존 bilinear 방식 FP32 interpolation(`align_corners=False`), CLS 보존. CLS 포함244token.
- MAP은 원래 구현의 dynamic embedding size로 768차원 /12heads /4blocks / vocabulary49408. 랜덤 초기화는 기존과 동일한 MAP 학습 모듈뿐입니다. CLIP의 모든 backbone key/shape/value는 strict 로딩합니다.
- SDM/MAP 계산, 40개 Attribute 순서, 기존 Prompt/BPE/masking, QueryBalanced replacement group_count^(-0.5), train/val join, resize/flip/CLIP normalization, metric 함수는 기존 구현 그대로입니다. Micro-Batch로 GPU에 올리되 배치 구성·목적 함수는 유지합니다.
- 총60Epoch, seed1, Adam(AdamW 아님), LR1e-5, weight_decay4e-5, betas(.9,.999), eps1e-3, linear warm-up5Epoch/factor.1, cosine target0, Epoch scheduler step 모두 유지합니다. Adam foreach=False는 추가 workspace를 줄이며 같은 update를 수행합니다.
- FP32만 지원합니다. AMP/autocast/GradScaler/FP16/BF16은 사용하지 않고 TF32도 끕니다.
- 저장 정책은 완료 Epoch1~10에 **plain model.state_dict()만** 저장합니다. Epoch11~60은 train/validation/metrics를 계속하며 Weight 저장이 없습니다. Optimizer/scheduler/epoch metadata나 best/last/latest checkpoint를 저장하지 않습니다.

## Effective Batch 32와 Gradient Cache

Config에서 micro1×32, micro2×16, micro4×8을 선택할 수 있습니다. Gradient Checkpointing도 켜고 끌 수 있습니다.

DataLoader는 기존처럼 같은32쌍의 augmentation/attribute masking을 한 번 생성합니다. Micro-Batch별 no-grad raw global features를 캐시한 뒤 전체32×32 normalized similarity를 사용해 원래 bidirectional SDM을 계산합니다. Feature leaf gradient를 구하고 같은 입력/RNG로 Encoder를 재계산하여 image/text encoder 양쪽으로 gradient를 전달합니다. MAP micro loss는 **supervised BPE token 수/전체 supervised token 수**로 가중하여 원래 token-mean loss와 일치시킵니다. Effective Batch당 Adam1회, 완료 Epoch당 scheduler1회입니다.

단순 Micro-Batch SDM gradient accumulation 대안은 제공하지 않습니다. BatchNorm/nonzero Dropout/학습 가능한 temperature를 거부하여 잘못된 정합성 주장을 방지합니다. 실제 CLIP/MAP은 LayerNorm 및 attention dropout0입니다. RNG 보존 FP32 block checkpointing을 제공합니다.

원본 loader는 drop_last=False이므로 마지막5개 샘플도 유지합니다. 따라서 마지막 배치는 원본과 동일하게 SDM5×5입니다. 정상 배치는 모두32×32이며 Epoch당3053 optimizer updates입니다. 각 Epoch sampler와 loader worker generator도 seed+epoch−1로 reseed합니다.

## 데이터 및 공식 Weight

학습은97,669장/7,204attribute groups, Validation은33,407장/3,462queries입니다. 모든 기존 학습 도메인을 유지합니다. Train/Val sample ID 및 resolved path 중복0, 이미지 존재 전수 검사 및 annotation SHA를 기록합니다. Pixel-content duplicate 검사는 하지 않습니다. Test 정답은 사용하지 않습니다.

공식 foundation: `weights/ViT-L-14.pt`, SHA256:
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
원본 OpenAI 저장 tensor를 로딩할 때 FP32로 변환합니다. Shape mismatch/미등록 backbone key를 무시하지 않습니다. 공식 CLIP scalar logit_scale/input_resolution/context_length/vocab_size는 명시적으로 제외하고 원래 QueryBalanced 고정 temperature.02를 유지합니다.

## Mercury 테스트

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B tests/test_core.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -u tests/gpu_smoke.py \
  --config configs/train_fp32_micro4.json --report-name gpu_smoke_micro4
```

CPU tests는 작은 공유 image/text/MAP 모델에서 full-batch와 cache loss/gradient를 비교합니다. Effective32 및 partial5, micro1/2/4, checkpoint on/off12조합을 검증합니다. 가짜60Epoch train engine 테스트에서 training/validation60회와 Weight10개만 저장하는지 확인합니다.

GPU smoke는 실제 TITAN Xp 및 실제 QueryBalanced32쌍으로2회 FP32 Forward/Backward/Adam을 수행합니다. Weight 갱신, finite FP32 gradient, 전체 SDM32, peak allocated/reserved, allocation 안정성, 실제 작은 Val retrieval, 임시 plain state_dict save/reload를 검증합니다. 임시 smoke Weight는 폐기되며 Full Training은 다시 공식 pretrained에서 시작합니다.

근거 파일은 `reports/unit_tests.json`, `reports/gpu_smoke_micro1.json`, `reports/gpu_smoke_micro4.json`, `reports/data_audit.json`, `reports/implementation_report_ko.md`입니다. 작은 Val subset 점수를 전체 Validation 성능으로 해석하지 않습니다.

## 실제60Epoch 실행

```bash
bash run_full.sh --config configs/train_fp32_micro4.json \
  --gpu-uuid GPU-4b164f16-af97-6e48-dac8-1225482ffa03
```

Launcher는 해당 configuration/GPU의 unit+actual GPU tests 통과를 요구합니다. 지정 GPU의 다른 compute process 또는 점유 메모리가 있으면 거부하며, 어떤 프로세스도 종료하지 않습니다. CUDA_VISIBLE_DEVICES에는 확인한 UUID 하나만 사용해 컨테이너의 GPU ordinal 차이를 처리합니다. UUID별 flock과 전용 tmux socket `/tmp/irra-qb-vitl14-single.sock`을 사용합니다. 신규 timestamp+random run directory를 만들며 기존 run을 덮어쓰지 않습니다.

실행 직후 실제 run path/session/console log 및 attach 명령을 출력합니다. 모든 결과는 영구 NAS의 신규 프로젝트 `experiments/full_<timestamp>_<id>/` 아래 저장됩니다.

```bash
tmux -S /tmp/irra-qb-vitl14-single.sock list-sessions
tmux -S /tmp/irra-qb-vitl14-single.sock attach-session -t 출력된_세션명
```

새 Run만 다음처럼 직접 실행할 수도 있습니다.

```bash
python -B -u train_upar.py --output "$PWD/experiments/새로운_Run명" \
  --config configs/train_fp32_micro4.json \
  --pretrained "$PWD/weights/ViT-L-14.pt" --device cuda \
  --gpu-uuid GPU-4b164f16-af97-6e48-dac8-1225482ffa03
```

예상 출력:

```text
experiments/full_<timestamp>_<id>/
  config.json
  pretrained_verification.json
  data_audit.json
  train.log
  status.json
  metrics.csv
  metrics.jsonl
  epoch_001.pth ... epoch_010.pth
```

매 Epoch 모든 Train/Val이 종료된 뒤만 Weight를 원자적·배타적으로 발행합니다. Weight파일 자체에는 모델 Tensor dictionary만 있습니다. 실제 Epoch/설정/Validation 연결은 별도 metrics/config/log에 기록합니다. 저장 중 파일은 `.weights-tmp-*`이며 실패 시 자체 임시 파일을 제거하고 다른 파일을 수정하지 않습니다. Epoch11~60은 save 함수가 state_dict를 만들지도 않습니다. 약19.9GB의 Weight 공간이 필요합니다.

`status.json`은 각 Effective Batch 완료 후 Epoch/Batch/global step/LR/loss/FP32 및 GPU peak를 갱신합니다. Full Validation은 기존 gallery/query loader 순서를 그대로 유지하고 GPU input만 memory.evaluation_micro_batch_size(default4)로 나누며 같은 cosine/metric을 계산합니다. 결과6지표는 각 Epoch CSV/JSONL에0~1 단위로 기록합니다.

SIGINT/SIGTERM은 현재 update/저장을 마치고 중단하며 불완전 Epoch Weight를 저장하지 않습니다. 정책상 optimizer/resume snapshot이 없어 중단 이후 정확한 재시작은 지원하지 않습니다. Epoch11 이후의 최종/best weight도 요청에 따라 저장하지 않으며 결과 metric만 남습니다.
