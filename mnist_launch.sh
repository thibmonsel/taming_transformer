#!/bin/sh
#SBATCH -J vqvae
#SBATCH --gres=gpu:1
#SBATCH --partition=besteffort
#SBATCH --nodes=1
#SBATCH -t 7-0
#SBATCH --output=out%j.out             
#SBATCH --error=out%j.err            
set -x

nvidia-smi

SAVE_DIR="metadata"
DEVICE="cuda"
DATASET="mnist"

# Training hyperparameters
EPOCHS=200
BATCH_SIZE=256
LEARNING_RATE="4e-4" 
COMMITMENT_COST=1.0
SAVE_INTERVAL=20

# Model architecture
BASE_FILTERS=32
N_BLOCKS=2
FILTER_MULTIPLIERS="1 2 2"
NUM_EMBEDDINGS=64
EMBEDDING_DIM=64

TRAINED_VQGAN_PATH=$SAVE_DIR

# Transformer parameters
PRIOR_LR="3e-4"
N_HEADS=8
N_LAYERS=6
BATCH_SIZE_2=256
GPT_EMBEDDING_DIM=256
DROPOUT=0.1
BLOCK_SIZE=256 
DISC_START_EPOCH=20
GAN_WEIGHT=1.0


echo "Starting training for dataset: $DATASET"

python train_vqgan.py \
    --save-plots-dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --dataset "$DATASET" \
    --save-every "$SAVE_INTERVAL" \
    --max-epochs "$EPOCHS" \
    --commitment-cost "$COMMITMENT_COST" \
    --lr "$LEARNING_RATE" \
    --base-channels "$BASE_FILTERS" \
    --n-blocks "$N_BLOCKS" \
    --latent-channels "$EMBEDDING_DIM" \
    --num-embeddings "$NUM_EMBEDDINGS" \
    --batch-size "$BATCH_SIZE" \
    --disc-start-epoch "$DISC_START_EPOCH" \
    --gan-weight "$GAN_WEIGHT" \
    --channel-multipliers $FILTER_MULTIPLIERS 


python train_cond_prior.py \
    --save-plots-dir "$SAVE_DIR" \
    --trained-vqgan-path "$TRAINED_VQGAN_PATH" \
    --device "$DEVICE" \
    --dataset "$DATASET" \
    --save-every "$SAVE_INTERVAL" \
    --max-epochs "$EPOCHS" \
    --commitment-cost "$COMMITMENT_COST" \
    --lr "$PRIOR_LR" \
    --base-channels "$BASE_FILTERS" \
    --n-blocks "$N_BLOCKS" \
    --latent-channels "$EMBEDDING_DIM" \
    --num-embeddings "$NUM_EMBEDDINGS" \
    --batch-size "$BATCH_SIZE_2" \
    --channel-multipliers $FILTER_MULTIPLIERS \
    --n-heads "$N_HEADS" \
    --n-layers "$N_LAYERS" \
    --gpt-embedding-dim "$GPT_EMBEDDING_DIM" \
    --dropout "$DROPOUT" \
    --block-size "$BLOCK_SIZE" \
 

