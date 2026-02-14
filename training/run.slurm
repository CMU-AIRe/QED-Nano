#!/bin/bash
#SBATCH --job-name=qed-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --output=slrum-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --requeue
#SBATCH --time=14-00:00:00

#########################################
# IMPORTANT INSTRUCTIONS:
# activate env before doing anything else
# make sure to set <FIXME> values below
#########################################

source ~/.bashrc
source qed-train/bin/activate

START_TIME=$(date +%s)
echo "START TIME: $(date)"

#########################################
# default values for the config and job
#########################################
CONFIG_NAME="qed_nano_rc"
JOB_NAME="qed_nano_rc"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: sbatch run.slurm [--config CONFIG_NAME] [--job-name JOB_NAME]"
      exit 1
      ;;
  esac
done

export CONFIG_NAME
export JOB_NAME

# Distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Build hostlist for srun
HOSTLIST_STR=$(IFS=,; echo "${NODELIST[*]}")

# Collect "hostname ip" pairs from the nodes
declare -A IP_OF
while read -r host ip; do
  [[ -n "$host" ]] && IP_OF["$host"]="$ip"
done < <(
  srun -w "$HOSTLIST_STR" --ntasks-per-node=1 bash -c '
    h=$(hostname -s)
    ip=$(hostname -I | tr " " "\n" | grep -m1 -E "^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")
    echo "$h $ip"
  '
)

# Fill the indexed array in the same order as NODELIST
ip_addr=()
for h in "${NODELIST[@]}"; do
  ip_addr+=("${IP_OF[$h]}")
done

# Export master address/port
export ALL_ADDR="$(IFS=,; echo "${ip_addr[*]}")"
if ((${#NODELIST[@]} > 0)); then
  export MASTER_ADDR="${ip_addr[0]}"
  export MASTER_PORT=6379
else
  echo "NODELIST is empty; cannot set MASTER_ADDR" >&2
  exit 1
fi


# Print some info
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NUM_NODES=$NUM_NODES"
echo "Nodes allocated: ${NODELIST[@]}"
echo "Host to IP mapping:"
for i in "${!NODELIST[@]}"; do
  printf "%s %s\n" "${NODELIST[$i]}" "${ip_addr[$i]}"
done
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "ALL_ADDR=$ALL_ADDR"
echo "--------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "CPUs per task/node: $SLURM_CPUS_PER_TASK"

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1


srun -w "$HOSTLIST_STR" --ntasks-per-node=1 \
  bash -lc '
    export OPENAI_API_KEY=<FIXME>
    export HF_TOKEN=<FIXME>
    export OPENAI_BASE_URL=<FIXME>
    export RANK=$SLURM_NODEID
    export WORLD_SIZE=$SLURM_NNODES
    export ALL_ADDR=$ALL_ADDR
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export HYDRA_FULL_ERROR=1 
    echo "[$(hostname -s)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT ALL_ADDR=$ALL_ADDR"
    timestamp=$(date +'%Y%m%d-%H%M%S'); python -m pipelinerl.launch --config-name=$CONFIG_NAME \
      output_dir=runs/${JOB_NAME}_${timestamp}
    '

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"
