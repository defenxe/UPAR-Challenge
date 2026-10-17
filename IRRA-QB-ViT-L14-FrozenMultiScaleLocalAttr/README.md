# QueryBalanced ViT-L/14 with a frozen multiscale local head

This is a separate five-epoch experiment. It restores the original
QueryBalanced ViT-L/14 Epoch 2 state_dict, freezes every original parameter,
and updates only `multiscale.*` and `local_branch.*`.

The added head reads ViT-L/14 visual tokens after blocks 8, 16, and 24. Blocks
8 and 16 are LayerNormed and projected from 1024 to the 768-dimensional joint
space. Block 24 uses the original CLIP LayerNorm/projection. The three patch
sets are fused with trainable softmax weights initialized to 0.2/0.2/0.6. A
40-query, 8-head local attribute branch is trained with `0.5 * BCEWithLogits`.

Global retrieval, text prompts/tokenization, image transforms, validation
metric code, and base model definitions come from the ViT-L/14 project. The
CLIP ViT-L/14 positional grid remains 27x9 for the 378x126 input. The copied
VisionTransformer implementation only adds optional extraction of blocks 8
and 16; the standard global path is unchanged.

`configs/frozen_localattr_e5.json` records the exact source checkpoint, its
SHA256, training configuration, data root, and validation fusion policy. The
shared dataset is referenced through `data_root`; no dataset or pretrained
weights are copied here. Every epoch checkpoint is a plain full inference
model `state_dict()` containing the frozen base and the trained local modules.

Run `scripts/preflight.py` first. Actual training is allowed only after its
CUDA Batch-32 preflight has passed and the assigned GPU is unoccupied. Start
`scripts/train_frozen.py` from a fresh default tmux session with the successful
preflight report path. The trainer saves exactly `epoch_001.pth` through
`epoch_005.pth` and exits normally after Epoch 5 validation and strict reload.
