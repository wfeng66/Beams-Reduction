#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Disable caching for OpenPCDet
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=0

# Create a temporary directory for processing
TEMP_DIR="$SCRIPT_DIR/temp_processed"
mkdir -p $TEMP_DIR

# Create a temporary config file
TEMP_CFG="$SCRIPT_DIR/temp_config.yaml"
cp "$SCRIPT_DIR/cfgs/kitti_models/second.yaml" "$TEMP_CFG"

# Process point clouds with different beam configurations
for beams in "2,28" "4,26" "6,24" "8,22" "10,20" "12,18" "14,16"
do
    echo "Processing with beams: $beams"
    
    # Clear any existing cache
    rm -rf $TEMP_DIR/*
    
    # Copy original data to temp directory
    cp -r data/kitti/training/velodyne $TEMP_DIR/
    
    # Process point clouds with beam reduction
    python damage_beams.py $TEMP_DIR/velodyne 4 32 $beams
    
    # Clear any cached data
    rm -rf data/kitti/training/gt_database
    rm -rf data/kitti/training/gt_info
    
    # Modify the config file to disable caching
    sed -i 's/CACHE_DATASET: True/CACHE_DATASET: False/g' "$TEMP_CFG"
    sed -i 's/CACHE_GT_DATABASE: True/CACHE_GT_DATABASE: False/g' "$TEMP_CFG"
    sed -i 's/CACHE_GT_INFO: True/CACHE_GT_INFO: False/g' "$TEMP_CFG"
    sed -i 's/CACHE_GT_SAMPLES: True/CACHE_GT_SAMPLES: False/g' "$TEMP_CFG"
    sed -i 's/CACHE_GT_SAMPLES_INFO: True/CACHE_GT_SAMPLES_INFO: False/g' "$TEMP_CFG"
    sed -i 's/USE_DB_SAMPLER: True/USE_DB_SAMPLER: False/g' "$TEMP_CFG"
    
    # Clear output directory
    rm -rf output/kitti_models/second/*
    
    # Run training with modified config
    python train.py \
        --cfg_file "$TEMP_CFG" \
        --batch_size 16 \
        --pretrained_model pretrained/second.pth \
        --dataset_version "$beams" \
        --num_epochs_to_eval 1 \
        --worker 16 \
        --train False
done

# Clean up
rm -rf $TEMP_DIR
rm -f "$TEMP_CFG"

