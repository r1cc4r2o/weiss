#!/bin/bash

source ./init_mamba.sh
mamba activate chemformer
nvidia-smi

export HYDRA_FULL_ERROR=1

i_chunk=$SLURM_ARRAY_TASK_ID
n_chunks=1


# model_path=./checkpoints/CF_WEISS_ep=46.pt
# output_file=./results/metrics_scores_seed_00.csv
# output_file_smiles=./results/sampled_molecules_seed_00.json
# output_score_efficiency=./results/out_efficiency_seed_00.csv
# data_path=../dataset/proc_selected_reactions_chemformer/proc_selected_reactions_chemformer.csv
# vocab=./chemformer/bart_vocab_disconnection_aware.json
# seed=0
# batch_size=128
# num_beams=1
# dataset_part=test
# dataset_type=synthesis
# task=backward_prediction
# model_type=bart_weiss
# num_predictions=50
# n_multinomial=null

model_path=./checkpoints/CF_ep=44.pt
output_file=./results/bart/metrics_scores_seed_00.csv
output_file_smiles=./results/bart/sampled_molecules_seed_00.json
output_score_efficiency=./results/bart/out_efficiency_seed_00.csv
data_path=../dataset/proc_selected_reactions_chemformer/proc_selected_reactions_chemformer.csv
vocab=./chemformer/bart_vocab_disconnection_aware.json
seed=0
batch_size=128
num_beams=1
dataset_part=test
dataset_type=synthesis
task=backward_prediction
model_type=bart_weiss
num_predictions=50
n_multinomial=null

echo "model_path: "${model_path} 
echo "data_path: "${data_path} 
echo "vocabulary: "${vocab} 
echo "output_file: "${output_file} 
echo "output_file_smiles: "${output_file_smiles} 
echo "batch_size: "${batch_size} 
echo "num_beams: "${num_beams} 
echo "dataset: "${dataset} 
echo "dataset_type: "${dataset_type} 
echo "dataset_part: "${dataset_part} 
echo "i_chunk: "${i_chunk} 
echo "n_chunks: "${n_chunks} 
echo "task: "${task}
echo "output_score_efficiency: "${output_score_efficiency}
echo "seed: "${seed}


model_path=${model_path//"="/"\="}


if [ -f ${output_file} ]
then
    rm ${output_file}
    rm ${output_file_smiles}

    echo "Temporary output files removed: "${output_file}", "${output_file_smiles}
fi


python -m molbart.inference_score data_path=${data_path} \
        output_score_data=${output_file} \
        output_sampled_smiles=${output_file_smiles} \
        model_path=${model_path} task=${task} \
        i_chunk=${i_chunk} n_chunks=${n_chunks} \
        dataset_part=${dataset_part} dataset_type=${dataset_type} \
        vocabulary_path=${vocab} n_beams=${num_beams} \
        batch_size=${batch_size} model_type=${model_type} \
        +num_predictions=${num_predictions}  \
        +n_multinomial=${n_multinomial} \
        +output_score_efficiency=${output_score_efficiency} \
        seed=${seed}
