# Chemformer
This repository contains the code used to generate the results in the Chemformer papers [[1]](#1) [[2]](#2).

The Chemformer project aimed to pre-train a BART transformer language model [[3]](#3) on molecular SMILES strings [[4]](#4) by optimising a de-noising objective. We hypothesized that pre-training would lead to improved generalisation, performance, training speed and validity on downstream fine-tuned tasks. 
The pre-trained model was tested on downstream tasks such as reaction prediction, retrosynthetic prediction, molecular optimisation and molecular property prediction in our original manuscript [[1]](#1). Our synthesis-prediction (seq2seq) Chemformer was evaluated for the purpose of single- and multi-step retrosynthesis [[2]](#2).

The public models and datasets available [here](https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq). To run these models with the new version, you first need to update the checkpoint, e.g.:
```
model = torch.load("model.ckpt")
model["hyper_parameters"]["vocabulary_size"] = model["hyper_parameters"].pop("vocab_size")
torch.save(model, "model_v2.ckpt")
```


## Prerequisites
Before you begin, ensure you have met the following requirements:

* Linux, Windows or macOS platforms are supported - as long as the dependencies are supported on these platforms.

* You have installed [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) with python 3.7

## Installation

First clone the repository using Git.

The project dependencies can be installed by executing the following commands in the root of 
the repository:

    conda env create -f env.yaml
    conda activate chemformer

If there is an error "ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found"
it can be mitigated by adding the 'lib' directory from the Conda environment to LD_LIBRARY_PATH

As example:
`export LD_LIBRARY_PATH=/path/to/your/conda/envs/chemformer/lib`

For developers: Run the following to enable editable mode
```
    pip install -e .
```

## User guide
The following is an example of how to fine tune Chemformer using the pre-trained models and datasets available [here](https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq).

1. Create a Chemformer conda environment, as above.
1. Download the dataset of interest and store it locally (e.g. ../data/uspto_50.pickle).
1. Download a pre-trained Chemformer model and store it locally (e.g ../models/pre-trained/combined.ckpt).
1. Update the `fine_tune.sh` shell script in the example_scripts directory (or create your own) with the paths to your model and dataset, as well as the values of hyperparameters you wish to pass to the script.
1. Run the `fine_tune.sh` script.

You can of course run other scripts in the repository following a similar approach. The Scripts section below provides more details on what each script does.

### Scripts
The molbart includes the following scripts:
* `molbart/train.py` runs the pre-training 
* `molbart/fine_tune.py` runs fine-tuning on a specified task
* `molbart/inference_score.py` predicts SMILES and evaluates the performance of a fine-tuned model
* `molbart/predict.py` predict products given input reactants
* `molbart/build_tokenizer.py` creates a tokenizer from a dataset and stores it in a pickle file
* `molbart/modules/retrosynthesis/round_trip_inference.py` runs round-trip inference and scoring using the predicted SMILES from `molbart/inference_score.py`

The scripts use hydra for reading parameters from config files. To run a script from `your/project/folder`, first create an experiment folder: `your/project/folder/experiment/`. In that folder add a config file with the parameters you wish to override the defaults for:
`your/project/folder/experiment/project_config.yaml`.

Example of project config yaml:
```
# @package _global_

seed: 2
dataset_part: test # Which dataset split to run inference on. [full", "train", "val", "test"]
dataset_type: synthesis
n_beams: 5
batch_size: 64
```

The script can then be run with
```
python -m molbart.<scipt_name> 'hydra.searchpath=[file:///your/project/folder] experiment=inference_score.yaml
```

See the default configuration files of each script under molbart/config/ for more details on each argument.

### Notes on running retrosynthesis predictions and round-trip validation 
Example of running inference and calulcating (1) top-N accuracy (stored in `metrics.csv`) and (2) round-trip accuracy (stored in `round_trip_metrics.csv`):
1. Run backward inference
```python -m molbart.inference_score data_path=data.csv output_score_data=metrics.csv output_sampled_smiles=sampled_smiles.json dataset_type=synthesis <additional_args>```
1. Run round-trip inference 
```python -m molbart.modules.retrosynthesis.round_trip_inference input_data=data.csv backward_predictions=sampled_smiles.json output_score_data=round_trip_metrics.csv output_sampled_smiles=round_trip_sampled_smiles.json <additional_args>```

When running analysis using the `dataset_type=synthesis` option (SynthesisDataModule), the input file given by `data_path` is assumed to be a tab-separated .csv file containing the columns `products` (SMILES), `reactants` (SMILES) and `set` (labels of each sample according to which dataset split it belongs to, i.e. "train", "val" or "test").

See the default configuration corresponding to each script in molbart/config/ (molbart/modules/retrosynthesis/config/ for round-trip) more details on each argument.

## Specifying available and custom callbacks
There are default callbacks used when fine-tuning or training, as well as for inference and round-trip evaluations. You can also specify which specific callbacks to use in your config file. Callbacks in molbart.modules.callbacks can now be added to the config file like:
```
callbacks:
  - LearningRateMonitor
  - ModelCheckpoint: # Select which parameter values should override the defaults
    - period: 1
    - monitor: val_loss
  - ValidationScoreCallback
  - OptLRMonitor
  - StepCheckpoint
```
You can also add you own custom callback with relative import (CustomCallback from my_package/callbacks.py):
```
callbacks:
  - my_package.callbacks.CustomCallback
```


## Specifying available and custom scores
There are default scores which are used in all scripts (including in `molbart.inference_score`, `molbart.round_trip`, `molbart.fine_tune`). You can also specify which specific scores to calculate in your config file. Scores in molbart.modules.scores can now be added to the config file like:
```
scorers:
  - FractionInvalidScore
  - FractionUniqueScore
  - TanimotoSimilarityScore:
    - statistics: mean
  - TopKAccuracyScore
```
You can also add you own custom scores with relative import (CustomScore from my_package/scores.py):
```
scorers:
  - my_package.scores.CustomScore
```
The default in `molbart.inference_score` and `molbart.round_trip` is to use the internal callback ScoreCallback which collects the computed scores listed under `scorers:` and writes to the specified output files (`output_score_data` and `output_sampled_smiles`).

## Running with FastAPI service
### Vanilla Chemformer forward or backward synthesis prediction
Chemformer predictions and log-likelihood calculations can be executed with FastAPI.

Install FastAPI libraries
```
    python -m pip install fastapi
    python -m pip install "uvicorn[standard]"
```
Then
```
    cd service
    export CHEMFORMER_MODEL={PATH TO MODEL}
    export CHEMFORMER_VOCAB={PATH TO VOCABULARY FILE}
    export CHEMFORMER_TASK=backward_prediction
    python chemformer_service.py
```
The model URL can for example be used to run multi-step retrosynthesis with [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder)

### Disconnection-aware retrosynthesis prediction
For running the disconnection-Chemformer, run the following (RXN-mapper should be installed in the environment):
```
    cd service
    export CHEMFORMER_MODEL={PATH TO VANILLA CHEMFORMER MODEL} # Optional
    export CHEMFORMER_DISCONNECTION_MODEL={PATH TO DISCONNECTION CHEMFORMER MODEL}
    export CHEMFORMER_VOCAB={PATH TO VOCABULARY FILE}
    export CHEMFORMER_TASK=backward_prediction
    export RXNUTILS_ENV_PATH={PATH TO rxnutils CONDA ENV}
    python chemformer_disconnect_service.py
```
## Code structure

The codebase is broadly split into the following parts:
* Models
* Modules, including utils, data helpers, decoders, etc.
* Scripts for running e.g. fine-tuning, prediction, etc.


### Models

The  `models/transformer_models.py` file contains a Pytorch Lightning implementation of the BART language model, as well as Pytorch Lightning implementations of models for downstream tasks.
`models/chemformer.py` contains the synthesis prediction Chemformer model used for both forward and backward (seq2seq) predictions.

### Modules

The `modules/data` folder contains DataModules for different tasks and datasets.
The classes which inherit from `_AbsDataset` are subclasses of Pytorch's `nn.utils.Dataset` and are simply used to store and split data (molecules, reactions, etc) into its relevant subset (train, val, test).
Our `_AbsDataModule` class inherits from Pytorch Lightning's `LightningDataModule` class, and its subclasses are used to augment, tokenize and tensorize the data before it passed to the model.

Finally, we include a `TokenSampler` class which categorises sequences into buckets based on their length, and is able to sample a different batch size of sequences from each bucket. This helps to ensure that the model sees approximately the same number of tokens on each batch, as well as dramatically improving training speed.

### Tokenization

Our `modules/tokenizer.py` file includes the `MolEncTokeniser` class which is capable of random 'BERT-style' masking of tokens, as well as padding each batch of sequences to be the same length. The `ChemformerTokenizer`, which is used in the synthesis Chemformer makes use of the `SMILESTokenizer` from the `pysmilesutils` library for tokenising SMILES into their constituent atoms.


### Decoding / sampling

We include implementations of greedy and beam search, as well as a GPU-optimized beam search decoding (BeamSearchSampler) in the `modules/decoder.py` file. All implementations make use of batch decoding for improved evaluation speeds. They do not, however, cache results from previous decodes, rather, they simply pass the entire sequence of tokens produced so far through the transformer decoder. The BeamSearchSampler is used by the synthesis Chemformer model in molbart.models.chemformer.


## Contributing

We welcome contributions, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.


To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the remote branch: `git push`
5. Create the pull request.

Please use ``black`` package for formatting.


The contributors have limited time for support questions, but please do not hesitate to submit an issue.

## License
The software is licensed under the MIT license (see LICENSE file), and is free and provided as-is.


## References

<a id="1">[1]</a>
Irwin, R., Dimitriadis, S., He, J., Bjerrum, E.J., 2021. Chemformer: A Pre-Trained Transformer for Computational Chemistry. Mach. Learn. Sci. Technol. [https://doi.org/10.1088/2632-2153/ac3ffb](https://doi.org/10.1088/2632-2153/ac3ffb)

<a id="2">[2]</a>
Westerlund, A.M., Manohar Koki, S., Kancharla, S., Tibo, A., Saigiridharan, L., Mercado, R., Genheden, S., 2023. 
Do Chemformers dream of organic matter? Evaluating a transformer model for multi-step retrosynthesis, ChemRxiv
 [10.26434/chemrxiv-2023-685jv](10.26434/chemrxiv-2023-685jv)


<a id="3">[3]</a>
Lewis, Mike, et al.
"Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension."
arXiv preprint arXiv:1910.13461 (2019).

<a id="4">[4]</a>
Weininger, David.
"SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules."
Journal of chemical information and computer sciences 28.1 (1988): 31-36.
