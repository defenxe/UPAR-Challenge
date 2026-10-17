# IRRA-QB-ViT-L14-FrozenMultiScaleLocalAttr

이 5 Epoch 실험은 완료된 QueryBalanced Epoch 2 Checkpoint를 불러옵니다. 기존 모델을 고정하고 기존 multiscale/local Attribute 모듈을 학습합니다.

QueryBalanced 학습이 끝난 뒤 저장소 루트에서 다음 명령을 실행하세요.

```bash
bash scripts/train_pipeline.sh frozen
```

파이프라인은 가장 최근에 완료된 QueryBalanced Run을 찾아 Epoch 2 Checkpoint의 SHA-256을 계산합니다. 해당 Checkpoint의 메타데이터와 GPU UUID를 연결한 Run별 Config를 만든 뒤, 기존 CUDA Batch-32 사전 검사와 Checkpoint 출처 검증을 수행합니다. 검증을 통과한 경우에만 Frozen 학습을 시작합니다.

전체 학습 절차와 결과 경로는 저장소 [학습 안내](../README.md)를 참고하세요. 기존 학습 설정은 `configs/frozen_localattr_e5.json`에 그대로 있습니다.
