
import sys
sys.path.append('../')

import time 
import glob
import importlib

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.module.sampling import *
from src.setup import setup_inference_o2o_beamsearch

import random
random.seed(42)



#####################################
# Load the model and the test set   #
#####################################

def load_test_set(model):
    test_smiles = open(
        "../dataset/test_source.txt"
    ).readlines()
    test_smiles += open(
        "../dataset/test_source.txt"
    ).readlines()
    test_smiles = set([x.strip() for x in test_smiles])

    X = [
        torch.tensor(model.vocabulary.encode(model.tokenizer.tokenize(x)))
        for x in test_smiles
    ]
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    return X


def main(_args):

    print('Args: ', _args) 

    MODEL_PATH = _args.base_path_cpk # '../priors/Mol2Mol_best.pt'

    ## Load the vocab and tokenizer
    vocabulary_class = getattr(importlib.import_module(f"src.module.vocab"), 'Vocabulary')
    tokenizer = getattr(importlib.import_module(f"src.module.vocab"), 'SMILESTokenizer')()
    vocabulary = vocabulary_class.load_from_dictionary(torch.load(f"{_args.path_vocabulary}"))

    model_class = getattr(importlib.import_module(f"src.model.mol2mol"), 'Mol2Mol')
    model = model_class(vocabulary_size=len(vocabulary.tokens())).eval().to(_args.device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_args.device))
    model.vocabulary = vocabulary
    model.tokenizer = tokenizer

    print('Model loaded')
    
    print('Loading the test set')
    X = load_test_set(model)
    idx = np.arange(len(X))
    idx = random.sample(list(idx), 1000)
    

    data_loader = torch.utils.data.DataLoader(
        X[idx].long(), batch_size=_args.batch_size, shuffle=False, drop_last=False
    )
    print('Test set loaded')
    
    #####################################
    # Run the inference                 #
    #####################################
    
    
    print('Starting inference...')
    inference_time = 0.0
    headings = ["source_id", "source", "predicted_target", "nll", "inference_time"]
    wout_log = open(f"../results/Mol2MolBeamsearch_batchsize={_args.batch_size:04d}_beamsize={_args.beam_size:04d}_device={_args.device}.csv", "a+")
    wout_log.write(",".join(headings) + "\n") # write the headings
    with torch.no_grad():
        
        csv_rows = [] 
        for smi_id, x in enumerate(tqdm(data_loader)):
            B, L = x.shape
            source_tokens = [vocabulary.decode(x[jp].cpu().numpy().tolist()) for jp in range(B)]
            smi = [tokenizer.untokenize(source_tokens[jp]) for jp in range(B)]
            
            start = time.time()
            x = x.long().to(_args.device)
            x_mask = (x == 0).to(_args.device)
            smi_idx = torch.arange(len(smi)).unsqueeze(1).repeat(1, _args.beam_size).reshape(len(smi)*_args.beam_size)
            node = NodeO2O(model, (x, x_mask), vocabulary, _args.device, 
                           type_of_tokenizer = 'reinvent', data_device=_args.device, 
                           batch_size=B)
            stop_criterion = LogicalOr([EOS(), MaxLength(_args.max_seq_lenght)])
            node = beamsearch(node, _args.beam_size, stop_criterion)
            stop = time.time()
            inference_time = stop - start
            
            yp = node.y.cpu()
            nlls = node.loglikelihood.squeeze(-1).cpu().numpy()
                
            # Collect outputs in a df and log on the fly
            for i, i_smi in zip(range(len(yp)), smi_idx):
                idx_yp_i = torch.where(yp[i] == 2)[0]
                if len(idx_yp_i) > 0:
                    if len(yp[i]) > idx_yp_i[0]+1:
                        yp[i][idx_yp_i[0]+1:] = 0
                psmi = tokenizer.untokenize(vocabulary.decode(yp[i].numpy()))
                csv_rows.append((smi_id, smi[i_smi], psmi, nlls[i], inference_time))
                wout_log.write(f"{smi_id},{smi[i_smi]},{psmi},{nlls[i]},{inference_time}\n")
                
                    
        df = pd.DataFrame(csv_rows, columns=headings)
        df.to_csv(f"../results/Mol2MolBeamsearch_batchsize={_args.batch_size:04d}_beamsize={_args.beam_size:04d}_device={_args.device}_backup.csv", index=None)
            
    wout_log.close()
    print('Inference done')


if __name__ == "__main__":
    
    _args = setup_inference_o2o_beamsearch()
    
    main(_args)