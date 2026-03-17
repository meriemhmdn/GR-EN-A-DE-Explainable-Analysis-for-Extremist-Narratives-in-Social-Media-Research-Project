#!/usr/bin/env bash
# GRENADE-EXPASS Integration Example Script
# 
# This script demonstrates how to run EXPASS with GRENADE outputs.
# Adjust paths to your GRENADE output files.

# Example usage for different GRENADE datasets
datasets=("toxigen" "lgbten" "migrantsen" "french" "german" "cypriot" "slovene")
architectures=("gcn" "graphconv" "leconv")
explainers=("gnn_explainer" "pgmexplainer")

# IMPORTANT: Set these paths to your actual GRENADE output files
GRENADE_EMBEDDINGS_PATH="path/to/embeddings__exp1__ntrials_1.npy"
GRENADE_ADJACENCY_PATH="path/to/adjacency_final__exp1__ntrials_1.pkl"

# Validate paths before running
if [ ! -f "${GRENADE_EMBEDDINGS_PATH}" ] || [ "${GRENADE_EMBEDDINGS_PATH}" = "path/to/embeddings__exp1__ntrials_1.npy" ]; then
    echo "ERROR: GRENADE_EMBEDDINGS_PATH is not set or file does not exist."
    echo "Please edit this script and set GRENADE_EMBEDDINGS_PATH to your actual embeddings file."
    exit 1
fi

if [ ! -f "${GRENADE_ADJACENCY_PATH}" ] || [ "${GRENADE_ADJACENCY_PATH}" = "path/to/adjacency_final__exp1__ntrials_1.pkl" ]; then
    echo "ERROR: GRENADE_ADJACENCY_PATH is not set or file does not exist."
    echo "Please edit this script and set GRENADE_ADJACENCY_PATH to your actual adjacency file."
    exit 1
fi

now=$(date +%F-%R)
logdir="logs/${now}.log"
mkdir -p logs

echo "GRENADE-EXPASS Pipeline Examples" | tee ${logdir}
echo "=================================" | tee -a ${logdir}
echo "" | tee -a ${logdir}

# Example for running with a single dataset (Toxigen by default)
for architecture in ${architectures[@]}; do
    for explainer in ${explainers[@]}; do

        printf "\n\n[dataset: 'toxigen'][architecture: '%12s'][explainer: '%12s'][vanilla: 'OFF']\n" \
            $architecture $explainer | tee -a ${logdir}
        python train.py \
            --grenade-embeddings ${GRENADE_EMBEDDINGS_PATH} \
            --grenade-adjacency ${GRENADE_ADJACENCY_PATH} \
            --dataset-name toxigen \
            --epochs 27 \
            --explainer_epochs 25 \
            --arch $architecture \
            --explainer $explainer 2>&1 | tee -a ${logdir}

        printf "\n\n[dataset: 'toxigen'][architecture: '%12s'][explainer: '%12s'][vanilla: 'ON']\n" \
            $architecture $explainer | tee -a ${logdir}
        python train.py \
            --grenade-embeddings ${GRENADE_EMBEDDINGS_PATH} \
            --grenade-adjacency ${GRENADE_ADJACENCY_PATH} \
            --dataset-name toxigen \
            --epochs 27 \
            --explainer_epochs 25 \
            --vanilla_mode \
            --arch $architecture \
            --explainer $explainer 2>&1 | tee -a ${logdir}
    done
done
