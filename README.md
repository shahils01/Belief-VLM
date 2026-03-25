# Belief-VLM

`Belief-VLM` is now wired for local HD-EPIC training with an InternVL backbone.
The training objective is causal language modeling over a text target conditioned on video frames.

## HD-EPIC modes

The current pipeline supports two HD-EPIC modes.

1. Preferred: local annotation supervision
- Provide `--annotation_path` with records containing a video id/path plus question/answer fields.
- The model trains on `video + prompt -> answer`.

2. Fallback: metadata-derived supervision
- If `--annotation_path` is omitted, the loader pairs local MP4 files with `framewise_info.jsonl`.
- It then generates a structured target text describing start/middle/end gaze and device motion.
- This lets you train the framework immediately with the data you already downloaded.

## Required local data

- `--video_root`
  Example: `/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos`
- `--metadata_root`
  Example: `/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data`

The loader expects videos at:

```text
<video_root>/<participant>/<video_id>.mp4
```

and metadata at:

```text
<metadata_root>/<participant>/<video_id>/framewise_info.jsonl
```

## Loss

Training uses standard causal LM cross-entropy with prompt masking:

```text
L = CrossEntropy(next_token_logits, target_tokens_only)
```

## Training

Example DDP run with metadata-derived supervision:

```bash
cd /home/i2r/shahil_ws/Belief-VLM
bash train_ddp.sh
```

Example DDP run with an annotation manifest:

```bash
ANNOTATION_PATH=/path/to/hd_epic_manifest.jsonl \
VIDEO_ROOT=/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos \
METADATA_ROOT=/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data \
bash train_ddp.sh
```

## Annotation manifest format

If you have a local supervision file, a minimal JSONL record looks like:

```json
{"video_id":"P08-20240613-122900","participant_id":"P08","question":"What is the wearer doing?","answer":"The wearer walks through a room and looks around."}
```

The loader is configurable with:
- `--question_column`
- `--answer_column`
- `--video_id_column`
- `--video_path_column`
- `--participant_column`

## Notes

- HD-EPIC intermediate metadata alone is not benchmark supervision, but it is enough to create a structured text target and train the VLM pipeline end-to-end.
- InternVL is the default backbone now.
