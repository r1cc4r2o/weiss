import sys
sys.path.append('../')

import os
import torch
import pickle
from tqdm import tqdm

from src.module.vocab import SMILESTokenizer, build_vocabulary
from src.setup import setup_build_vocab

# Parse the arguments
arg_parser = setup_build_vocab()
print('Args: ', arg_parser)


###################################
# Define the paths and parameters #
###################################

PATH_DATASET = arg_parser.path_dataset
FILE_NAME_VOCAB = arg_parser.file_name_vocab


###########################################
# Load the data and create the vocabulary #
###########################################

# Check if the vocabulary is already created
if os.path.exists(f"{PATH_DATASET}/{FILE_NAME_VOCAB}"):
    print("Vocabulary already exists in the following path: ", f"{PATH_DATASET}/{FILE_NAME_VOCAB}")
    # Load the vocabulary
    with open(f"{PATH_DATASET}/{FILE_NAME_VOCAB}", "rb") as fobj:
        vocabulary = pickle.load(fobj)
# If not, create it
else:
    tokenizer = SMILESTokenizer()
    smiles = set()
    for _set in ["train", "validation", "test"]:
        smiles |= set(
            [smi.strip() for smi in open(f"{PATH_DATASET}/{_set}_source.txt").readlines()]
        )
        smiles |= set(
            [smi.strip() for smi in open(f"{PATH_DATASET}/{_set}_target.txt").readlines()]
        )
    vocabulary = build_vocabulary(list(smiles))
    with open(f"{PATH_DATASET}/{FILE_NAME_VOCAB}", "wb") as fobj:
        pickle.dump(vocabulary, fobj)
    torch.save(vocabulary.get_dictionary(), f"{PATH_DATASET}/dict_vocab.pt")
        
        

#############################################
# Encode the data and save the encoded data #
#############################################

full_data = {}
for _set in ["train", "validation", "test"]:
    
    # Load 
    
    sources = set(
        [smi.strip() for smi in open(f"{PATH_DATASET}/{_set}_source.txt").readlines()]
    )
    targets = set(
        [smi.strip() for smi in open(f"{PATH_DATASET}/{_set}_target.txt").readlines()]
    )
    
    # Encode
    
    source_vect = {}
    for s in tqdm(set(sources) | set(targets)):
        s_tokens = tokenizer.tokenize(s)
        x = vocabulary.encode(s_tokens)
        source_vect[s] = torch.from_numpy(x).long()
        
    # Store in a list
    
    data = []
    for s, t in tqdm(zip(sources, targets)):
        data.append((source_vect[s], source_vect[t]))

    full_data[f'data_{_set}'] = data

## Save

torch.save(full_data, f"{PATH_DATASET}/data.pt")