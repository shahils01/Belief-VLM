VL_MODEL_PRESET="${VL_MODEL_PRESET:-custom}"
VL_MODEL_NAME="${VL_MODEL_NAME:-/scratch/shahils/hf_models/InternVL3_5-2B-HF}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/}"
METADATA_ROOT="${METADATA_ROOT:-/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data}"
VLM_CHECKPOINT="${VLM_CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_99.pt}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-2048}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
MACHINE_RANK="${MACHINE_RANK:-${SLURM_NODEID:-0}}"
NUM_PROCESSES_PER_NODE="${NUM_PROCESSES_PER_NODE:-2}"
NUM_MACHINES="${NUM_MACHINES:-${SLURM_NNODES:-1}}"
TOTAL_PROCESSES="$((NUM_PROCESSES_PER_NODE * NUM_MACHINES))"


if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "SLURM_JOB_NODELIST is not set. Run this inside a Slurm allocation." >&2
  exit 1
fi

MASTER_ADDR_HOSTNAME="${MASTER_ADDR_HOSTNAME:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-$(getent hosts "$MASTER_ADDR_HOSTNAME" | awk '{print $1}' | head -n 1)}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

if [[ -z "$MAIN_PROCESS_IP" ]]; then
  echo "Failed to resolve MAIN_PROCESS_IP from $MASTER_ADDR_HOSTNAME." >&2
  exit 1
fi

CMD=(
  accelerate launch
  --num_processes "$TOTAL_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --machine_rank "$MACHINE_RANK"
  --main_process_ip "$MAIN_PROCESS_IP"
  --main_process_port "$MAIN_PROCESS_PORT"
  train_ppo_vqa.py
  --dataset_type hd_epic_local
  --video_root "$VIDEO_ROOT"
  --metadata_root "$METADATA_ROOT"
  --annotation_path "$ANNOTATION_PATH"
  --video_extension mp4
  --val_ratio 0.1
  --max_val_samples_per_split "$MAX_VAL_SAMPLES"
  --train_sampling_mode task_uniform
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --batch_size 32
  --num_workers 8
  --video_frames 8
  --mixed_precision bf16
  --allow_tf32
  --epochs 500
  --ppo_epochs 2
  --log_every 1
  --policy_lr 1e-4
  --vlm_lr 2e-5
  --vl_model_preset "$VL_MODEL_PRESET"
  --vl_model_name "$VL_MODEL_NAME"
  --gradient_checkpointing
  --save_dir checkpoints_ppo_vqa_fulldataset
  --resume_checkpoint checkpoints_ppo_vqa_01/ckpt_epoch_62.pt
)

if [[ -n "$VLM_CHECKPOINT" ]]; then
  CMD+=(--vlm_checkpoint "$VLM_CHECKPOINT")
fi

echo "MASTER_ADDR_HOSTNAME=$MASTER_ADDR_HOSTNAME"
echo "MAIN_PROCESS_IP=$MAIN_PROCESS_IP"
echo "MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
echo "NUM_MACHINES=$NUM_MACHINES"
echo "MACHINE_RANK=$MACHINE_RANK"
echo "NUM_PROCESSES_PER_NODE=$NUM_PROCESSES_PER_NODE"
echo "TRAIN_SAMPLES_PER_EPOCH=$TRAIN_SAMPLES_PER_EPOCH"
echo "MAX_VAL_SAMPLES=$MAX_VAL_SAMPLES"
echo "VL_MODEL_PRESET=$VL_MODEL_PRESET"
echo "VL_MODEL_NAME=$VL_MODEL_NAME"

"${CMD[@]}"
