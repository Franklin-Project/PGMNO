#!/bin/bash
# =============================================================================
# Main Training Script for PGMNO
# =============================================================================
# Trains full PGMNO model with Response Letter Table A1 hyperparameters.
# Uses 3 seeds for reproducibility.
#
# Memory Optimization:
#   - Gradient checkpointing enabled in LatentMambaOperatorV3
#   - Selective scan memory optimized with torch.no_grad()
#   - Recommended batch_size=32 for A100 40GB
#
# Usage:
#   ./train_main.sh                                # Default (batch_size=32)
#   ./train_main.sh --n_epochs 100 --n_train 500  # Quick test
#   ./train_main.sh --batch_size 16               # Lower memory usage
# =============================================================================

set -e

echo "=============================================="
echo "PGMNO Main Training"
echo "Response Letter Table A1 Configuration"
echo "=============================================="

# Default parameters (Table A1)
# Memory-optimized for A100 GPU (40GB VRAM)
# Note: Gradient checkpointing is enabled by default in enhanced_pgmno_model.py
N_TRAIN=${N_TRAIN:-1000}
N_EPOCHS=${N_EPOCHS:-500}   # 论文Table 4声明: 500 epochs
BATCH_SIZE=${BATCH_SIZE:-32}   # Memory-optimized: 32 for A100 40GB (was 512)
LR=${LR:-0.001}                # Fixed LR as per Response Letter Table A1
K_STEPS=${K_STEPS:-2}
EPSILON=${EPSILON:-0.05}
SEEDS=${SEEDS:-"42"}           # Single seed by default, use "42 123 456" for full experiment
DEVICE=${DEVICE:-cuda}

# Flags
GENERATE_DATA="--generate_data"  # Fixed: now default enabled

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_train) N_TRAIN="$2"; shift 2 ;;
        --n_epochs) N_EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --k_steps) K_STEPS="$2"; shift 2 ;;
        --epsilon) EPSILON="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --no_generate_data) GENERATE_DATA=""; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Training samples: $N_TRAIN"
echo "  Epochs: $N_EPOCHS"
echo "  Batch size: $BATCH_SIZE (memory-optimized)"
echo "  Learning rate: $LR"
echo "  BDF order (k): $K_STEPS"
echo "  Causal weight (ε): $EPSILON"
echo "  Seeds: $SEEDS"
echo "  Device: $DEVICE"
echo "  Horizon: Fixed 20 steps (progressive training removed)"
echo "  Gradient Checkpointing: Enabled (default)"
echo ""

# Environment setup (optional: configure if cuDNN/cuBLAS paths differ)
# Uncomment and modify if using a non-standard environment:
# ENV_ROOT="/path/to/your/conda/env"
# export LD_LIBRARY_PATH=$ENV_ROOT/lib:$LD_LIBRARY_PATH

# Create output directories
mkdir -p checkpoints logs results

# Run training
python scripts/train_main.py \
    --n_train "$N_TRAIN" \
    --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --k_steps "$K_STEPS" \
    --epsilon "$EPSILON" \
    --seeds $SEEDS \
    --device "$DEVICE" \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --results_dir results \
    $GENERATE_DATA

echo ""
echo "=============================================="
echo "Training Complete!"
echo "Results saved to: results/main_training_results.json"
echo "Checkpoints saved to: checkpoints/"
echo "=============================================="
