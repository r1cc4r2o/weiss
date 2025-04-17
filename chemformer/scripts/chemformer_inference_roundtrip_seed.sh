#!/bin/bash

export HYDRA_FULL_ERROR=1

i_chunk=$SLURM_ARRAY_TASK_ID
n_chunks=50


model_path=
input_data=
backward_predictions=
vocabulary_path=
working_directory=.

seed=1
batch_size=128
n_gpus=1
dataset_part=test
dataset_type=synthesis
output_score_data=
output_file_smiles=

data_path=null # Placeholder argument
task=forward_prediction

echo "model_path: "${model_path} 
echo "input_data: "${input_data}
echo "backward_predictions: "${backward_predictions}
echo "vocabulary_path: "${vocabulary_path}
echo "output_score_data: "${output_score_data}
echo "output_sampled_smiles: "${output_file_smiles}
echo "batch_size: "${batch_size}
echo "dataset_part: "${dataset_part}
echo "task: "${task}
echo "i_chunk: "${i_chunk}
echo "n_chunks: "${n_chunks}
echo "data_path: "${data_path}
echo "working_directory: "${working_directory}
echo "seed: "${seed}


model_path=${model_path//"="/"\="}


if [ -f ${output_file} ]
then
    rm ${output_file}
    rm ${output_file_smiles}

    echo "Temporary output files removed: "${output_file}", "${output_file_smiles}
fi


export HYDRA_FULL_ERROR=1
python -m molbart.modules.retrosynthesis.round_trip_inference data_path=${data_path} \
        output_score_data=${output_score_data}${i_chunk}.csv \
        output_sampled_smiles=${output_file_smiles}${i_chunk}.json \
        model_path=${model_path} task=${task} \
        dataset_type=${dataset_type} dataset_part=${dataset_part} \
        vocabulary_path=${vocabulary_path} \
        seed=${seed} batch_size=${batch_size} \
        n_gpus=${n_gpus} working_directory=${working_directory} \
        i_chunk=${i_chunk} n_chunks=${n_chunks} \
        input_data=${input_data} backward_predictions=${backward_predictions}  