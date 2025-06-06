
# WEISS: Wasserstein Efficient Sampling Strategy for LLMs in Drug Design

🤗 ![License](https://img.shields.io/badge/license-MIT-green) 

## Table of Contents

- [WEISS: Wasserstein Efficient Sampling Strategy for LLMs in Drug Design](#weiss-wasserstein-efficient-sampling-strategy-for-llms-in-drug-design)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
    - [Molecular Property Optimization and Natural Language](#molecular-property-optimization-and-natural-language)
    - [Molecular Property Optimization with Reinforcement Learning](#molecular-property-optimization-with-reinforcement-learning)
    - [Single-step Retro-synthesis](#single-step-retro-synthesis)
  - [Usage](#usage)
    - [Running the Experiments](#running-the-experiments)
      - [Molecular Property Optimization](#molecular-property-optimization)
      - [Molecular Property Optimization with Reinforcement Learning](#molecular-property-optimization-with-reinforcement-learning-1)
      - [Single-step Retro-synthesis](#single-step-retro-synthesis-1)
      - [Natural Language](#natural-language)
    - [Parameters](#parameters)
  - [Code Structure](#code-structure)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)
  - [References](#references)

## Introduction


<p align="center">
 <img src="img/wss-training.png" width="45%">
 
 <img src="img/wss-inference.png" width="45%" >
 <p align="center"> Train&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Inference</p>

 <!-- <figure align="center">
  <figcaption>This repository contains the code used in the WEISS paper.</figcaption>
  <figcaption>[Wasserstein Efficient Sampling Strategy for LLMs in Drug Design](https://iopscience.iop.org/article/10.1088/2632-2153/addc33)</figcaption>
</figure> -->
</p>


This repository contains the code to reproduce the results of the paper [Wasserstein Efficient Sampling Strategy for LLMs in Drug Design](https://iopscience.iop.org/article/10.1088/2632-2153/addc33).

WEISS is a novel framework for efficient diverse sampling applicable to large language models (LLMs) in drug design. We present WEISS (Wasserstein Efficient Sampling Strategy), a novel framework for training and efficient sampling, that enhances the output diversity while preserving its integrity. Our WEISS framework leverages a continuous latent variable to expand the expressiveness of the higher-dimensional representations of each token. The experiments on drug discovery tasks demonstrate that the proposed mechanism not only increases the diversity but also maintains high levels of similarity with the conditioning input. Notably, our model effectively balances the trade-off between generating diverse outputs and ensuring that these outputs remain closely correlated with the input, showing significant improvement overcome the challenges in molecular optimization, with and without RL, and single-step retrosynthesis. Experiments on NLP suggest that WEISS is a versatile framework, applicable to a broader category of autoregressive encoder-decoder models. This README provides instructions on the code usage.


## Installation


We recommend using a virtual environment to manage the dependencies ([mamba](https://github.com/mamba-org/mamba)). The paper presents the results on several experiments, including:

- Molecular Property Optimization
- Molecular Property Optimization with Reinforcement Learning
- Single-step Retro-synthesis
- Natural Language

These can be run separately or together, depending on the desired experiment. To run the code, you need to follow these steps to set up the environment:

### Molecular Property Optimization and Natural Language

1. **Clone the repository:**

    ```bash
    git clone weiss
    cd weiss
    ```

2. **Set up a virtual environment (optional but recommended):**

    ```bash
    mamba create -n weiss python==3.10
    mamba activate weiss
    ```


3. **Install the required dependencies:**

For Linux, you can install the dependencies using the following command:

    ```bash
    pip install -r requirements.lock    
    ```
Then, you can install the same pytorch version used in the paper:

    ```bash
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    Optional: if you want to use **AMD GPUs** on Linux you would need to install the [ROCm PyTorch version](https://pytorch.org/get-started/locally/) manually _after_ installation of the dependencies in point 3, e.g.

   ```bash
  pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
   ```

### Molecular Property Optimization with Reinforcement Learning


1. **Clone the repository:**

    ```bash
    git clone weiss
    cd WEISS/reinvent
    ```

2. **Set up a virtual environment (optional but recommended):**

    ```bash
    mamba create --name reinvent4 python=3.10
    mamba activate reinvent4
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements-linux-64.lock
    ```
    Optional: if you want to use **AMD GPUs** on Linux you would need to install the [ROCm PyTorch version](https://pytorch.org/get-started/locally/) manually _after_ installation of the dependencies in point 3, e.g.

   ```bash
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.7
   ```

   Install the REINVENT4 tool with WEISS integrated. The dependencies were already installed in the previous step, so there is no need to install them again (flag `--no-deps).  If you want to install in editable mode (changes to the code are automatically picked up) add -e before the dot.

    ```bash
    pip install --no-deps . 
    ```


### Single-step Retro-synthesis


1. **Clone the repository:**

    ```bash
    git clone weiss
    cd WEISS/chemformer/chemformer
    ```

2. **Set up a virtual environment (optional but recommended):**
    
    ```bash
    mamba env create -f env.yaml
    conda activate chemformer
    ```

3. **Install the required dependencies:**

    ```bash
    mamba env create -f env.yaml
    mamba activate chemformer
    pip install eco2ai
    pip install --no-deps . 
    cp -r ./molbart/config <your_path>/miniforge3/envs/<name_env>/lib/python3.7/site-packages/molbart/
    ```

    Run the following to enable editable mode:
    ```
    pip install -e .
    ```

## Usage

### Running the Experiments

```bash
cd WEISS/scripts
```

The paper presents the results of several experiments, each experiment needs its own configuration. We provide the scripts to run the experiments in the `scripts/` directory for MPO and NLP. Meanwhile, for Reinforcement Learning and Retro-synthesis, the scripts are in the `reinvent` and `chemformer` directories, respectively.

Download the checkpoints [here](https://drive.google.com/file/d/18RA1OlGQozMeibgiex5V8e9Z7nbcY1Fv/view?usp=drivesdk) and unzip the folder. You can store them under `WEISS/checkpoints`.


#### Molecular Property Optimization

- **Data:** 

The data we employed are openly available and can be downloaded [here](https://zenodo.org/records/6319821). The subset utilized is in the folder `data/similarities`, which contains pairs of SMILES collected according to their Tanimoto similarity (>=0.5). We further checked the validity of the SMILES strings, canonicalized the structures, and removed explicit hydrogens (”[H]”). To convert the smiles in the correct format, we used the `rdkit` library and `molvs` library. Unzip the file `data.tar.gz` and place the `data` folder in the `weiss/dataset` directory. Thus, you can run the following script for the conversion:

```bash
cd WEISS/scripts
python mpo-convert-data.py --path_dataset=../dataset/data/similarity --_split=train 
python mpo-convert-data.py --path_dataset=../dataset/data/similarity --_split=test
python mpo-convert-data.py --path_dataset=../dataset/data/similarity --_split=validation
```

Thus, we stored the molecules in `txt` files inside the `weiss/dataset` folder: 

```bash 

WEISS
├── dataset                    # Contains datasets 
│   ├── train_source.txt                # Raw molecules x for training
│   ├── train_target.txt                # Raw molecules y for training
│   ├── validation_source.txt           # Raw molecules x for validation
│   ├── validation_target.txt           # Raw molecules y for validation
│   ├── test_source.txt                 # Raw molecules x for testing
│   ├── test_target.txt                 # Raw molecules y for testing
    ...

```

Example of the data format:
```bash
linux@weiss:~/WEISS/dataset$ head test_source.txt
Cc1n[nH]c2ccnc(NC(C)C)c12
Cc1ccccc1CN1CCN(c2cn[nH]c(=O)c2Br)CC1=O
O=C1Nc2ccc([N+](=O)[O-])cc2C1=Cc1ccncc1
COc1c(C)cc(-c2cn3c(=O)n(-c4ccccc4)nc3c(N)n2)cc1C
NC(=O)COc1ccc(-c2cn3c(=O)n(-c4ccccc4)nc3c(N)n2)cc1
Nc1ccc(-c2cn3c(=O)n(-c4ccccc4)nc3c(N)n2)cc1
```

To preprocess the data, you can run the following script:

```bash
cd WEISS/scripts
python scripts/mpo-build.py --path_dataset=../dataset --file_name_vocab=vocab.pkl
```

This script will preprocess the data and save the processed `train_test_valid_preprocessed.pt` data by default in the `../dataset/` directory.


- **Training the model:**

To reproduce the results from the paper, you can run the training script in the same configuration of the paper as follows:



```bash
cd WEISS/scripts
python mpo-train.py \
            --model_name # possible values 
                         # [ Mol2MolWEISS, Mol2MolWEISSWithB, 
                         #    Mol2MolVAE, Mol2Mol, Mol2MolLSTM]
            --path_vocabulary # default='../dataset/dict_vocab.pt'
            --path_dataset # default='../dataset/
            --batch_size # default=1
            --device # default='cpu'
            --_lambda # default=10.0
            --n_epochs # default=100
            --lr # default=1e-3
            --hidden_ae_dim # default=4
            --optimizer # default='Noam' but you 
                        # can choose 'Adam'
```

We performed the training with seed 42, but you can mannually change the seed in the script. The training script will save the logs in the `../results/{run_name}_b={_args.hidden_ae_dim}_l={_args._lambda}` directory, and in `../priors/{_args.model_name}/{_args.model_name}_epoch={epoch:02d}_validloss={valid_loss:.3f}.pt` the model weights will be saved. The script will also save the best model over the validation set in `../priors/{_args.model_name}/{_args.model_name}_best.pt`.

Thus, you can use the inference script `scripts/mpo-inference-all-multinomial.py` considering the validation score as a proxy to select the best model. We relese following this proxy the weights of each `prior` of the different model versions.


- **Evaluating the model:**


We provide a script to directly perform the inference on the test set with multinomial varying the temperature. The script will save the results in the `../results/{typemodel}_t={temp:.2f}_gpus={NGPUS:03d}_gpuid={GPU_ID:03d}.csv` directory. 

```bash
cd WEISS/scripts
python mpo-inference-all-multinomial.py \
            --temp # default=0.5
            --ckpt_path # default='../priors/Mol2MolWEISS/Mol2MolWEISS_best.pt'
            --gpu_id # default=0 
            --gpus # default=1
            --batch_size # default=1
            --n_samples # default=2
            --max_seq_len # default=128
            --path_vocabulary # default='../dataset/dict_vocab.pt'
```

We recommend using a GPU for faster training and inference. The parameters `gpu_id` and `gpus` indicate the GPU to be used and the number of GPUs, respectively. The script can chunk the data to be processed in parallel on multiple GPUs. The `n_samples` parameter indicates the number of samples to be generated for each input molecule. The `max_seq_len` parameter indicates the maximum length of the generated SMILES strings.


In addition we provide a script to evaluate `Mol2Mol` with beam search, with just a few modification you could use the same script to evaluate the other models. 

```bash
cd WEISS/scripts
python mpo-inference-mol2mol-beamsearch.py \
            --beam_size # default=10
            --batch_size # default=4
            --device # default='cuda'
            --path_vocabulary # default='../dataset/dict_vocab.pt'
            --base_path_cpk # default='../checkpoints/Mol2Mol_best.pt'
```

Given the pre-trained models you can reproduce the results by running the inference in the same configuration as the paper. Thus, you can use the following script to compute the metrics for the generated molecules.

```bash
cd WEISS/scripts
python mpo-table-02.py 
```

#### Molecular Property Optimization with Reinforcement Learning

- **Data:**

The following molecules have been identified by [He](https://assets-eu.researchsquare.com/files/rs-4106688/v1_covered_479d388e-b8bd-4e9e-ac38-cf6d70903276.pdf?c=1723479140) et al. as active based on the Dopamine receptor D2 (DRD2) activity model using a [tree-based](https://assets-eu.researchsquare.com/files/rs-4106688/v1_covered_479d388e-b8bd-4e9e-ac38-cf6d70903276.pdf?c=1723479140) approach. This experiment aims to generate optimized molecules that exhibit enhanced quantitative estimate of drug-likeness (QED) and a higher likelihood of being active against the DRD2 target.
```python
# molecules to optimize
'CCN1CCC[C@H]1CNC(=O)c1c(Cl)c(C)cc(O)c1OC'
'Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl'
'N1(C(CN2CCN(CC2)CC3=CC=C(C=C3)OC)=O)C(CC4=C1C=CC=C4)C'
'O(C=1C=C2C=3[C@H](N(CC2)C)CC=4C(C3C1)=C(O)C(O)=CC4)CCC'
```


- **Training the model:**

Due to the presence of existing models in reinvent with the same name; we integrated both the models re-naming Mol2MolWEISS and Mol2Mol with One2Many and One2One respectively.

We provide an exxample of config file to run the evaluation with qed and tanimoto as scoring components in `reinvent/one2many/config.toml`. To run the experiment, it is needed to provide the path to the agent and prior files, and the smiles to optimize.

```toml
device = "cpu" # TOFIX
agent_file = "path_to_agent_file" # TOFIX ex. ./WEISS/checkpoints/one2many_prior.pt
prior_file = "path_to_prior_file" # TOFIX ex. ./WEISS/checkpoints/one2many_prior.pt
params.smiles = [
   "MOLECULE_SMILES" # TOFIX -> copy and paste the same molecule
                     #           in the file seed_smiles.smi in 
                     #           the same smiles format
]
```

We provide a jupyter notebook to setup the agent and prior checkpoints `notebooks/SetupRLModel.ipynb`.

To run the experiment, you can do the following:


```bash
cd WEISS/reinvent/one2many # for weiss
reinvent config.toml
```

```bash
cd WEISS/reinvent/one2one # for mol2mol
reinvent config.toml
```

- **Evaluating the model:**

To reproduce the results in the paper, you can run several times the script `reinvent config.toml` following the exact same configurations shown in the paper. Collected the `smiles_1.csv` files resulting from each run, you can reproduce the cumulative plot following the instructions in the paper.


#### Single-step Retro-synthesis

- **Data:**

The USPTO-full is a collection of 1M reactions that can be downloaded from the following repository [USPTO-full](https://zenodo.org/records/7341155/files/selected_reactions.csv?download=1). To reproduce the datasplit we used the Chemformer data preprocessing pipeline available in [AiZynthTrain](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01486). The data is split into train, validation, and test sets using as seed 11. In order to split the data, you'll need to install AiZynthTrain, follow the intruction using `poetry install`. Thus, you can reproduce the splits using the following configuration file:

```bash
chemformer_data_prep:
  selected_reactions_path: selected_reactions.csv
  training_fraction: 0.9
  random_seed: 11
  reaction_components_path: reaction_components.csv
  chemformer_data_path: proc_selected_reactions_chemformer.csv
  nbatches: 100
  reaction_hash_col: PseudoHash
```

And run the following script to split and process the data:

```bash
python -m aizynthtrain.pipelines.chemformer_data_prep_pipeline run --config ${path_config_file} --max-workers 48 --max-num-splits 200
```

- **Fine-tune the model:**

You'll need to download the model as reported in the supplementary materials. To fine-tune the model, you can run the following script:

```bash
cd WEISS/chemformer
python -m molbart.fine_tune 'hydra.searchpath=[./data/current/hydra_template]' experiment=fine_tune.yaml
```

```bash
cd WEISS/chemformer
python -m molbart.fine_tune 'hydra.searchpath=[./data/hydra_template]' experiment=fine_tune_weiss.yaml
```

Make sure the yaml files in `weiss/chemformer/data/hydra_template` contains all the necessary informations to run the experiment. By default the model is in inference model, to fine-tune modify the files looking at the encoding functions in `weiss/chemformer/chemformer/molbart/models/weiss_transformer.py` BARTweiss.

- **Evaluating the model:**


We release the finetuned models in the `WEISS/chemformer/checkpoints` directory. Checkout the scripts in the `WEISS/chemformer/scripts` directory to run the inference and evaluate the model. The inference script will save the results in the `WEISS/chemformer/results/` directory, you can setup by default `WEISS/chemformer/logs` as logging directory. 

To perform the inference with the model look at the script `WEISS/chemformer/scripts/` directory. Here we provide a description of the parameters to run the inference:

```bash
python -m molbart.inference_score data_path=${data_path} \ # path to csv data
        output_score_data=${output_file} \ # path to save the output
        output_sampled_smiles=${output_file_smiles} \ # path to save the sampled smiles json
        model_path=${model_path} task=${task} \ # path to the model checkpoint
        i_chunk=${i_chunk} n_chunks=${n_chunks} \ # chunk of the data to process
                                                  # these are necessary to parallelize the inference
                                                  # on multiple GPUs
        # id_gpu=i_chunk while n_chunks=number of GPUs   
        dataset_part=${dataset_part} dataset_type=${dataset_type} \ 
        vocabulary_path=${vocab} n_beams=${num_beams} \ # number of beams and vocabulary path
        batch_size=${batch_size} model_type=${model_type} \ # the model type
        +num_predictions=${num_predictions}  \ # number of predictions
        +n_multinomial=${n_multinomial} \ # if you want to sample with multinomial
        +output_score_efficiency=${output_score_efficiency} \ # path to save the efficiency score
        seed=${seed} # seed for the random number generator
```


Once you have the inference data, you can run the following script to compute the round trip efficiency:

```bash
python -m molbart.modules.retrosynthesis.round_trip_inference data_path=${data_path} \ 
        output_score_data=${output_score_data}${i_chunk}.csv \ # path store the output score
        output_sampled_smiles=${output_file_smiles}${i_chunk}.json \ # path to store the sampled smiles
        model_path=${model_path} task=${task} \ # path to the model checkpoint, and type of task
        dataset_type=${dataset_type} dataset_part=${dataset_part} \ # type of dataset and part of the dataset
        vocabulary_path=${vocabulary_path} \ # path to the vocabulary
        seed=${seed} batch_size=${batch_size} \ # seed for the random number generator and batch size
        n_gpus=${n_gpus} working_directory=${working_directory} \ # number of GPUs and working directory
        i_chunk=${i_chunk} n_chunks=${n_chunks} \ # chunk of the data to process (necessary to parallelize the inference)
        input_data=${input_data} backward_predictions=${backward_predictions} 
```

The last script it will run a product-prediction Chemformer trained on the reverse task to evaluate the round trip efficiency. Training details related to this Chemformer model can be found in the [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01685).


#### Natural Language

- **Data:**

You can directly download the data from the [Hugging Face](https://huggingface.co/datasets/chembl) repository [YahooA](https://huggingface.co/datasets/sentence-transformers/yahoo-answers) (title-answer), [YahooQA](https://huggingface.co/datasets/nfL6/yahoo_answers_qa) (question-nBestAnswers). 

```bash
cd WEISS/scripts
python nlp-build.py
```

The script will download and preprocess the data and save the processed `yahoo_answers_qa_{_split}.pt` data by default in the `../dataset/` directory. 

- **Training the model:**

To reproduce the results from the paper, you can run the training script in the same configuration of the paper as follows:

```bash
cd WEISS/scripts
python nlp-train.py \
            --gpu # default=None
            --device # default='cuda'
            --world-size # default=-1 if you want to use DDP
            --rank # default=-1 rank for distributed training
            --dist-url # default='env://' url used to set up distributed training
            --dist-backend # default='nccl' distributed backend
            --type_model # default='Mol2MolWEISS' type of model
            --lambda_ # default=40.0 mmd lambda
            --b # default=16 bottleneck size
            --base_path # default='setup_path'
            --local_rank # default=-1 local rank for distributed training
            --batch_size # default=32 batch size
            --epochs # default=10 number of epochs

```

The training script will save the logs, and chepoints in the `../priors/nlp/{run_name}_bottleneck_{args.b}` directory. The script by default it will run at every epoch the validation with temperature 1.0 on the whole validation set. You can disable this or reduce on a smaller subset of the validation set by changing line 171 from `dataset_valid[:]` to `dataset_valid[:subset_size]` for example.

- **Evaluating the model:**

Then you can run the inference evaluating the best models on the test set. The following sript will save: the input questions, the generated answers, the ground truth answers, and the NLL over the generated and ground truth answers. The results will be saved in the `../results/nlp/{run_name}_bottleneck_{args.b}` directory.
    

```bash
cd WEISS/scripts
python nlp-inference-all-multinomial.py \
            --path # default='../priors/nlp/Text2Text_bottleneck_16/Text2Text_best.pt'
            --N # default=None  number of samples to evaluate, if None all
            --K # default=10 number of molecules sampled
            --base_path # default='base_path'
            --device # default='cuda'
            --run_name # default='Text2Text' (type of model)
```


To reproduce the scores in table 4, you can run the following script once it last the inference for the previous script. It takes as input the inference dataframes with the ground truth and generated molecules. Thus, it computes the overlap between the two distributions.

```bash
cd WEISS/scripts
python nlp-table-04.py \
            --path_df_gt # path inference df ground truth molecules
            --path_df_generated # path inference df generated molecules varing the temperature
```


### Parameters

You can modify the parameters in the file `src/setup.py` or pass them directly via the command line. 

## Code Structure

The repository is organized as follows:

```bash
WEISS.
├── dataset                   # Contains datasets, vocab and processed data
├── src                       # Contains the source code
│   ├── model                 # Contains the models
│   ├── module                # Contains the modules
│   ├── utils.py              # Contains the utility functions
│   ├── const.py              # Contains the constants
│   ├── setup.py              # Contains the setup for the experiments
├── checkpoints               # Contains the checkpoints of the models persent in the paper (selected with the validation score)
├── results                   # Contains the results of the experiments
├── scripts                   # Contains the scripts to run the experiments
├── notebooks                 # Contains the notebooks to setup the experiments
├── reinvent                  # Contains the Reinforcement Learning code
├── chemformer                # Contains the Retro-synthesis code
├── logs                      # Contains the logs of the experiments
├── img                       # Contains the images of the architecture
├── requirements.lock # Contains the requirements for the MPO and NLP experiments
├── LICENSE                   # Contains the license
├── README.md        


```


## Results

The results will be saved in the `results/` directory, due to the large size of the files (`>200GB`), we could not upload the full csv files.

## Contributing

We welcome contributions to this project! 

## License

This project is not licensed, but it will be under the MIT License upon acceptance - see the [LICENSE](LICENSE) file for details.

## References

If you use this code in your research, please cite the following paper:

```bibtex
@article{Tedoldi_2025,
    doi = {10.1088/2632-2153/addc33},
    url = {https://dx.doi.org/10.1088/2632-2153/addc33},
    year = {2025},
    month = {jun},
    publisher = {IOP Publishing},
    volume = {6},
    number = {2},
    pages = {025048},
    author = {Tedoldi, Riccardo and Li, Junyong and Engkvist, Ola and Passerini, Andrea and Westerlund, Annie M and Tibo, Alessandro},
    title = {WEISS: Wasserstein efficient sampling strategy for LLMs in drug design},
    journal = {Machine Learning: Science and Technology},
}
```

---

Feel free to reach out if you have any questions or issues. 🤗
