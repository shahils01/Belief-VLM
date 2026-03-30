# Belief-VLM

Belief-VLM is an egocentric video question answering codebase built around:

- an `InternVL` vision-language model as the multimodal encoder
- a supervised VLM training path for multiple-choice HD-EPIC VQA
- a PPO answer-head training path on top of pooled VLM states
- an optional online episodic vector memory for retrieval-conditioned PPO

The repository is currently centered on local HD-EPIC style training and evaluation.

## What Is In This Repo

Core files:

- [model.py](/home/i2r/shahil_ws/Belief-VLM/model.py): InternVL wrapper and pooled hidden-state extraction
- [data_loading.py](/home/i2r/shahil_ws/Belief-VLM/data_loading.py): HD-EPIC loaders, video decoding, prompt construction
- [train.py](/home/i2r/shahil_ws/Belief-VLM/train.py): supervised VLM training
- [eval_hd_epic_vqa.py](/home/i2r/shahil_ws/Belief-VLM/eval_hd_epic_vqa.py): supervised VLM evaluation
- [train_ppo_vqa.py](/home/i2r/shahil_ws/Belief-VLM/train_ppo_vqa.py): PPO answer-head training
- [eval_ppo_vqa.py](/home/i2r/shahil_ws/Belief-VLM/eval_ppo_vqa.py): PPO answer-head evaluation
- [vector_memory.py](/home/i2r/shahil_ws/Belief-VLM/vector_memory.py): online episodic memory used by the PPO path

Launchers:

- [train_ddp.sh](/home/i2r/shahil_ws/Belief-VLM/train_ddp.sh): supervised VLM training
- [eval_hd_epic_vqa.sh](/home/i2r/shahil_ws/Belief-VLM/eval_hd_epic_vqa.sh): supervised VLM evaluation
- [train_ppo_vqa.sh](/home/i2r/shahil_ws/Belief-VLM/train_ppo_vqa.sh): single-node PPO training
- [train_ppo_vqa_slurm_multinode.sh](/home/i2r/shahil_ws/Belief-VLM/train_ppo_vqa_slurm_multinode.sh): multi-node PPO training on Slurm
- [eval_ppo_vqa.sh](/home/i2r/shahil_ws/Belief-VLM/eval_ppo_vqa.sh): PPO evaluation

## Environment

Two environment files/scripts are provided:

- [environment.yml](/home/i2r/shahil_ws/Belief-VLM/environment.yml)
- [palmetto_env.yml](/home/i2r/shahil_ws/Belief-VLM/palmetto_env.yml)

For Palmetto, the intended setup path is:

```bash
cd /home/i2r/shahil_ws/Belief-VLM
bash setup_palmetto_env.sh ma-vlcm
conda activate ma-vlcm
```

For a manual environment, you need at least:

- Python 3.10
- PyTorch 2.3+
- CUDA-enabled `torch`, `torchvision`, `torchaudio`
- `transformers`
- `accelerate`
- `datasets`
- `wandb`
- `peft`
- `sentencepiece`
- `protobuf`
- `safetensors`

Optional:

- `faiss-cpu` or `faiss-gpu` for faster vector-memory retrieval
- `torch-geometric` if you are reusing the older MA-VLCM environment

## Data Layout

The current code expects local HD-EPIC assets.

Typical paths:

- videos:
  `/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos`
- metadata:
  `/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data`
- annotations:
  `/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark`

Video path convention:

```text
<video_root>/<participant>/<video_id>.mp4
```

Metadata path convention:

```text
<metadata_root>/<participant>/<video_id>/framewise_info.jsonl
```

Annotation input can be:

- a single `.json`, `.jsonl`, or `.csv`
- a directory of annotation files
- a comma-separated list of files

The HD-EPIC VQA loader supports benchmark JSON files keyed by sample ID.

## Models

This repo currently targets InternVL 3.5 Hugging Face checkpoints such as:

- `OpenGVLab/InternVL3_5-2B-HF`
- `OpenGVLab/InternVL3_5-4B-HF`

In practice, use a local copy on cluster storage.

Example download:

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

## Training Modes

### 1. Supervised VLM training

This path trains the InternVL model with standard causal LM loss over the answer text.

Example:

```bash
cd /home/i2r/shahil_ws/Belief-VLM
bash train_ddp.sh
```

Useful overrides:

```bash
ANNOTATION_PATH=/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json \
VIDEO_ROOT=/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos \
METADATA_ROOT=/scratch/shahils/hd_epic_dataset/HD-EPIC\\ Intermediate\\ Data \
VL_MODEL_PRESET=custom \
VL_MODEL_NAME=/scratch/shahils/hf_models/InternVL3_5-2B-HF \
bash train_ddp.sh
```

Evaluation:

```bash
bash eval_hd_epic_vqa.sh
```

### 2. PPO answer-head training

This path keeps the VLM as a context encoder and trains a PPO policy/value head over pooled VLM states.

Example:

```bash
cd /home/i2r/shahil_ws/Belief-VLM
bash train_ppo_vqa.sh
```

Key features:

- multiple-choice answer selection
- optional VLM fine-tuning during PPO
- `state_pooling` over InternVL hidden states
- task-uniform training sampling
- optional online vector memory

Evaluation:

```bash
bash eval_ppo_vqa.sh
```

### 3. Multi-node PPO on Slurm

Example inside an allocation:

```bash
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES \
  bash /home/i2r/shahil_ws/Belief-VLM/train_ppo_vqa_slurm_multinode.sh
```

## Online Vector Memory

The PPO path can use an online episodic vector memory.

Current behavior:

- memory entries are created from past training experience
- memory stores:
  - frozen/intermediate context embedding
  - past response embedding
  - past reward
- retrieval uses cosine-style similarity on normalized latent vectors
- retrieved memory is fused into the current PPO state through a gated fusion module

Important:

- memory is written online during training
- memory is saved in PPO checkpoints
- evaluation can load the saved memory from a PPO checkpoint
- this is not an offline pre-built database

Enable it with:

```bash
USE_DB_PRIOR=1 bash train_ppo_vqa.sh
```

Useful memory knobs:

- `DB_TOP_K`
- `DB_INDEX_BACKEND=auto|faiss|numpy`
- `DB_SAME_TASK_FIRST=1`
- `MEMORY_LAYER_IDX`
- `FREEZE_MEMORY_PREFIX=1`

Recommended starting point:

```bash
USE_DB_PRIOR=1 \
DB_TOP_K=1 \
MEMORY_LAYER_IDX=1 \
FREEZE_MEMORY_PREFIX=1 \
bash train_ppo_vqa.sh
```

## Resuming Checkpoints

### Supervised VLM

Use `--resume_checkpoint` via [train_ddp.sh](/home/i2r/shahil_ws/Belief-VLM/train_ddp.sh).

### PPO

Use `--resume_checkpoint` via [train_ppo_vqa.sh](/home/i2r/shahil_ws/Belief-VLM/train_ppo_vqa.sh).

If you are changing the architecture or training regime, prefer loading only weights:

```bash
--load_model_only
```

This is especially important when:

- switching from policy-only PPO to VLM fine-tuning
- enabling the newer vector-memory fusion path

## LoRA / QLoRA

Both supervised and PPO paths support PEFT.

Typical PPO usage:

```bash
PEFT=lora
```

or directly in Python args:

```text
--peft lora
```

For PPO with a trainable VLM:

- `LoRA` is the safer first option
- `QLoRA` is possible but more fragile
- gradient accumulation is recommended if batch size becomes small

## Common Commands

Train PPO on one VQA annotation file:

```bash
ANNOTATION_PATH=/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json \
bash train_ppo_vqa.sh
```

Evaluate PPO on all `3d_perception_*` files:

```bash
ANNOTATION_PATH="$(printf "%s," /scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/3d_perception_*.json | sed 's/,$//')" \
bash eval_ppo_vqa.sh
```

Evaluate the full selected split with no 50-sample cap:

```bash
MAX_VAL_SAMPLES=0 VAL_RATIO=1.0 bash eval_ppo_vqa.sh
```

## Notes on Performance

- Video decoding is done locally from MP4 files.
- The loader uses uniformly sampled frames from the clip window and seeks directly to target frames when possible.
- Long clips are still more expensive than short clips.
- The online vector memory can improve performance, but it also adds compute and retrieval-state complexity.
- If you fine-tune the VLM heavily, retrieval consistency becomes an issue unless the memory/query encoder is kept frozen or partially frozen.

## Current Scope / Limitations

- The plain supervised VLM and PPO paths are the stable baselines.
- The online vector-memory PPO path is implemented and checkpointed, but still an active research direction.
- There is no separate offline memory-building script in the current mainline code.
- The repository is currently optimized for local HD-EPIC experiments, not general-purpose VQA benchmarks.

## Citation

If you use this code, cite the corresponding paper once finalized. The current draft lives in:

- [beliefVLM_paper/root.tex](/home/i2r/shahil_ws/Belief-VLM/beliefVLM_paper/root.tex)
