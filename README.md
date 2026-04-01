# Belief-VLM

This branch, `generalized-benchmarks-sft`, turns the repo into an online-first multimodal supervised fine-tuning and evaluation framework for InternVL-style vision-language models.

Default workflow on this branch:

- generalized supervised VLM training
- benchmark-oriented evaluation
- Hugging Face `datasets` as the preferred data source
- multiple-choice tasks trained generatively by verbalizing the correct answer

Legacy workflows are still present:

- HD-EPIC local supervised training
- PPO answer-head training
- online vector-memory PPO experiments

Those paths remain runnable, but they are no longer the default entrypoint on this branch.

## Main Entry Points

Generalized benchmark path:

- `train_general_vlm.py`: generalized multimodal SFT
- `eval_general_benchmarks.py`: generalized benchmark evaluation
- `launchers/train_general_vlm.sh`: shell launcher for training
- `launchers/eval_general_benchmarks.sh`: shell launcher for evaluation
- `datasets_registry.py`: dataset adapter registry
- `benchmark_registry.py`: benchmark evaluator registry
- `datasets_adapters/`: benchmark and local-manifest adapters
- `benchmark_evaluators/`: benchmark-specific answer scoring
- `general_data_utils.py`: shared multimodal media decoding, prompt packing, and collate utilities

Legacy paths kept intact:

- `train.py`
- `eval_hd_epic_vqa.py`
- `train_ppo_vqa.py`
- `eval_ppo_vqa.py`
- `vector_memory.py`

## Supported Benchmarks

First-wave adapters on this branch:

| Adapter | Default source | Modality | Train | Eval | Notes |
|---|---|---:|---:|---:|---|
| `mmmu` | `MMMU/MMMU` | image | yes | yes | multiple configs / subjects; use `DATASET_CONFIG` |
| `mathvista` | `AI4Math/MathVista` | image | yes | yes | supports free-form and MCQ-style samples |
| `ocrbench` | `LIME-DATA/ocrbench` | image | yes | yes | generative OCR-style scoring |
| `ai2d` | `lmms-lab/ai2d` | image | yes | yes | multiple-choice as generative SFT |
| `videomme` | `lmms-lab/Video-MME` | video | optional | yes | benchmark metadata is public; local or extracted media may still be needed depending on source format |
| `mlvu` | `sy1998/MLVU_Test` | video | optional | yes | benchmark metadata is public; local or extracted media may still be needed depending on source format |
| `local_manifest` | local JSON/JSONL/CSV | image/video | yes | yes | generic local adapter |
| `hd_epic` | local HD-EPIC files | video | yes | yes | legacy-style local adapter retained |

Important:

- The image benchmarks are the cleanest supported path for online-first training right now.
- Some video benchmarks publish metadata via Hugging Face but still package raw videos as zip/tar assets; for those, you may need local extracted media even on this branch.
- The generalized path is benchmark-oriented, not a reproduction of the full InternVL3.5 training recipe.

## Environment

Recommended on Palmetto:

```bash
cd /home/i2r/shahil_ws/Belief-VLM
bash setup_palmetto_env.sh ma-vlcm
conda activate ma-vlcm
```

Minimum important packages:

- Python 3.10
- `torch`, `torchvision`, `torchaudio`
- `transformers`
- `accelerate`
- `datasets`
- `peft`
- `sentencepiece`
- `protobuf`
- `safetensors`

Optional:

- `wandb`
- `faiss-cpu` or `faiss-gpu`

InternVL model downloads:

```bash
huggingface-cli download OpenGVLab/InternVL3_5-2B-HF \
  --local-dir /scratch/shahils/hf_models/InternVL3_5-2B-HF
```

Recommended cache variables:

```bash
export HF_HOME=/scratch/shahils/.cache/huggingface
export HF_HUB_CACHE=/scratch/shahils/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/shahils/.cache/huggingface/transformers
```

## Generalized Training

The new default training mode is supervised causal-LM fine-tuning over multimodal prompts.

Design:

- every adapter normalizes raw samples into a common schema
- multiple-choice tasks are verbalized as text generation targets
- InternVL is trained with masked answer-token cross-entropy
- `LoRA` is the default PEFT mode on this branch

### Default launcher

```bash
cd /home/i2r/shahil_ws/Belief-VLM
bash launchers/train_general_vlm.sh
```

### Example: AI2D

```bash
DATASET_NAME=ai2d \
VL_MODEL_PRESET=custom \
VL_MODEL_NAME=/scratch/shahils/hf_models/InternVL3_5-2B-HF \
PEFT=lora \
MIXED_PRECISION=bf16 \
OUTPUT_DIR=checkpoints_general_ai2d \
bash /home/i2r/shahil_ws/Belief-VLM/launchers/train_general_vlm.sh
```

### Example: MMMU with a specific subject config

```bash
DATASET_NAME=mmmu \
DATASET_CONFIG=Accounting \
TRAIN_SPLIT=dev \
EVAL_SPLIT=validation \
VL_MODEL_PRESET=custom \
VL_MODEL_NAME=/scratch/shahils/hf_models/InternVL3_5-2B-HF \
PEFT=lora \
bash /home/i2r/shahil_ws/Belief-VLM/launchers/train_general_vlm.sh
```

### Example: Local JSON/JSONL/CSV manifest

```bash
DATASET_NAME=local_manifest \
ANNOTATION_PATH=/path/to/manifest.jsonl \
MEDIA_ROOT=/path/to/media \
VL_MODEL_PRESET=custom \
VL_MODEL_NAME=/scratch/shahils/hf_models/InternVL3_5-2B-HF \
bash /home/i2r/shahil_ws/Belief-VLM/launchers/train_general_vlm.sh
```

Supported local manifest schema:

```python
{
  "id": "...",
  "task_name": "...",
  "media_type": "image" or "video",
  "image": "...",      # or image_path / media
  "video": "...",      # or video_path / media
  "question": "...",
  "answer": "...",
  "choices": [...],    # optional
  "correct_idx": 0     # optional
}
```

## Generalized Evaluation

The evaluation path supports:

- free-form generation
- normalized exact-match style scoring
- multiple-choice candidate scoring via sequence NLL

### Default launcher

```bash
cd /home/i2r/shahil_ws/Belief-VLM
bash launchers/eval_general_benchmarks.sh
```

### Example: evaluate AI2D with a trained checkpoint

```bash
DATASET_NAME=ai2d \
CHECKPOINT=checkpoints_general_ai2d/ckpt_epoch_1.pt \
VL_MODEL_PRESET=custom \
VL_MODEL_NAME=/scratch/shahils/hf_models/InternVL3_5-2B-HF \
EVAL_MODE=multiple_choice_nll \
bash /home/i2r/shahil_ws/Belief-VLM/launchers/eval_general_benchmarks.sh
```

### Example: evaluate MathVista

```bash
DATASET_NAME=mathvista \
CHECKPOINT=checkpoints_general_mathvista/ckpt_epoch_1.pt \
VL_MODEL_PRESET=custom \
VL_MODEL_NAME=/scratch/shahils/hf_models/InternVL3_5-2B-HF \
EVAL_MODE=generate \
bash /home/i2r/shahil_ws/Belief-VLM/launchers/eval_general_benchmarks.sh
```

## Important Arguments

Training:

- `DATASET_NAME`
- `DATASET_REPO`
- `DATASET_CONFIG`
- `TRAIN_SPLIT`
- `EVAL_SPLIT`
- `ANNOTATION_PATH`
- `MEDIA_ROOT`
- `VIDEO_ROOT`
- `VL_MODEL_PRESET`
- `VL_MODEL_NAME`
- `PEFT`
- `OUTPUT_DIR`
- `STREAMING`

Evaluation:

- `DATASET_NAME`
- `BENCHMARK_NAME`
- `CHECKPOINT`
- `EVAL_MODE=auto|generate|multiple_choice_nll`
- `SAVE_PREDICTIONS`

## Legacy Paths

The following remain available but are now legacy / task-specific on this branch:

- `train_ddp.sh`
- `eval_hd_epic_vqa.sh`
- `train_ppo_vqa.sh`
- `eval_ppo_vqa.sh`
- `train_ppo_vqa_slurm_multinode.sh`

These are still useful for:

- HD-EPIC local training
- PPO answer-head experiments
- vector-memory PPO research

They are not the default benchmark workflow anymore.

## Current Limitations

- The generalized branch does not reproduce InternVL3.5’s full public benchmark suite yet.
- Some video benchmarks still need local extracted media despite public Hugging Face metadata.
- Adapter defaults are best-effort public dataset handles and may need `DATASET_REPO` / `DATASET_CONFIG` overrides if upstream repos change.
- PPO and vector-memory code are intentionally not integrated into the new default benchmark path.

## Verification

This branch includes compile/syntax checks for:

- dataset adapters
- benchmark evaluators
- generalized train/eval scripts
- launcher shell syntax

Use this branch as the generalized SFT baseline, and treat the older HD-EPIC/PPO code as retained legacy functionality.
