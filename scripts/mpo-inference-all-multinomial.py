import sys
sys.path.append('../')

import os
import time
import importlib

import torch
import numpy as np
from tqdm import tqdm

from src.const import *
from src.setup import setup_inference_mpo_multinomial


#####################################
# Load the model and the test set   #
#####################################

def load_model(path: str, typemodel: str=None, zdim: int = None, path_vocabulary: str = '../dataset/dict_vocab.pt'):
    
    print(f"Loading tokenizers and vocabulary from {path_vocabulary}")
    vocabulary_class = getattr(importlib.import_module(f"src.module.vocab"), 'Vocabulary')
    tokenizer = getattr(importlib.import_module(f"src.module.vocab"), 'SMILESTokenizer')()
    vocabulary = vocabulary_class.load_from_dictionary(torch.load(f"{path_vocabulary}"))
    print("Vocabulary loaded!!")

    typemodel = path.split("/")[-1].split("_")[0]
    print(f"Loading model {typemodel} from {path}")
    model_class = getattr(
        importlib.import_module(f"src.model.{modelname2file[typemodel]}"), typemodel
    )

    # if ("WEISSWithB" in path) or ("WEISS" in path) or ("VAE" in path):
    try:
        zdim = int(path.split("/")[-1].split("_")[1].split("=")[1])
        model = model_class(vocabulary_size=len(vocabulary.tokens()), hidden_ae_dim=zdim)
    # elif ("LSTM" in path) or ("Mol2Mol" in path):
    except:
        zdim = None
        model = model_class(vocabulary_size=len(vocabulary.tokens()))
    # else:
    #     raise ValueError(f"Model not found {path}")
    
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
    except:
        raise ValueError(f"Model not found {path} or model is not compatible with {typemodel}")
    print("Model loaded!!")
    model = model.eval()
    model.vocabulary = vocabulary
    model.tokenizer = tokenizer
    return model, zdim, modelname2file[typemodel]


def load_test_set(model, gpu, n_gpus):
    test_smiles = open("../dataset/test_source.txt").readlines()
    test_smiles += open("../dataset/test_target.txt").readlines()
    test_smiles = sorted(list(set([x.strip() for x in test_smiles])))
    idx = np.arange(len(test_smiles))
    gpu_idx = np.array_split(idx, n_gpus)[gpu]
    gpu_test_smiles = []
    for i in gpu_idx:
        gpu_test_smiles.append(test_smiles[i])
    X = [torch.tensor(model.vocabulary.encode(model.tokenizer.tokenize(x)))
        for x in gpu_test_smiles]
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    return X


    
def main(_args):
    
    #####################################
    # Set up configuration for inference#
    #####################################
    
    temp = _args.temp
    ckpt_path = _args.ckpt_path
    GPU_ID= _args.gpu_id
    NGPUS= _args.gpus
    batch_size = _args.batch_size
    n_samples = _args.n_samples
    max_seq_len = _args.max_seq_len
    path_vocabulary = _args.path_vocabulary
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    #####################################
    # Load the model and the test set   #
    #####################################
    
    model, zdim, typemodel = load_model(ckpt_path)
    
    print(typemodel, type(typemodel), zdim, type(zdim), temp, type(temp), GPU_ID, type(GPU_ID), NGPUS, type(NGPUS))

    if zdim is None:
        csvwriter = open(f"../results/{typemodel}_t={temp:.2f}_gpus={NGPUS:03d}_gpuid={GPU_ID:03d}.csv", "a+")
    else:
        csvwriter = open(f"../results/{typemodel}_z={zdim}_t={temp:.2f}_gpus={NGPUS:03d}_gpuid={GPU_ID:03d}.csv", "a+")
        
    X = load_test_set(model, GPU_ID, NGPUS)
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(X.long(), batch_size=batch_size, shuffle=False, drop_last=False)


    #####################################
    # Run inference                    #
    #####################################
    
    with torch.no_grad():
       
        for x in tqdm(dataloader):
            x = x.to(device)
            
            start = time.time()
            y = torch.ones((len(x) * n_samples, 1), dtype=x.dtype, device=x.device)

            if (typemodel == "mol2mol") or (typemodel == "lstm"):
                x = x.unsqueeze(1).repeat(1, n_samples, 1).view(-1, x.shape[-1])
    
            break_condition = torch.zeros(len(y), dtype=torch.bool, device=x.device)
            nlls = torch.zeros(len(y), dtype=torch.float32, device=x.device)
            
            src_m = model.compute_padding_mask(x == 0)
            src_emb = model.pe(model.token_emb(x))
            src_enc = model.transformer.encoder(src_emb, src_m)

            if typemodel == "weiss":
                z = torch.randn((len(x), n_samples, model.hidden_ae_dim), device=x.device)
                src_enc = src_enc[:, None] + model.inv_proj(z).unsqueeze(-2) - model.inv_proj(torch.zeros_like(z)).unsqueeze(-2)
                src_enc = src_enc.view((-1,) + src_enc.shape[2:])
            elif typemodel == "weissb":
                z = torch.randn((len(x), n_samples, model.hidden_ae_dim), device=x.device)
                src_enc = src_enc[:, None] + model.inv_proj(z).unsqueeze(-2)
                src_enc = src_enc.view((-1,) + src_enc.shape[2:])
            elif typemodel == "vae":
                z = torch.randn((len(x) * n_samples, model.hidden_ae_dim), device=x.device)
                src_enc = model.build_past(z) 

            for _ in range(max_seq_len-1):
                log_probas = model.decode(src_enc, y, y == 0)
                probs = torch.softmax(log_probas[:, -1] / temp, -1)
                next = torch.multinomial(probs, 1).ravel() * (~break_condition)
                nlls += -log_probas[:, -1][torch.arange(len(log_probas)), next] * (1 - break_condition.float())
                y = torch.cat((y, next.unsqueeze(-1)), -1)
                break_condition = torch.logical_or(break_condition, next == 2)
                if torch.all(break_condition):
                    break 
                
            inference_time =  time.time() - start
            sources = [model.tokenizer.untokenize(model.vocabulary.decode(xx)) for xx in x.cpu().numpy()]
            targets = [model.tokenizer.untokenize(model.vocabulary.decode(yy)) for yy in y.cpu().numpy()]
            n_target_tokens = [(yy > 0).sum() for yy in y.cpu().numpy()]

            # Write results to csv file
            
            for iii, (target, nll, ntokens) in enumerate(zip(targets, nlls.cpu().numpy(), n_target_tokens)):
                if typemodel == "mol2mol":
                    csvwriter.write(f"{sources[iii]},{target},{ntokens},{nll:.10f},{inference_time:.10f},{len(x)},{n_samples}\n")
                else:
                    csvwriter.write(f"{sources[iii//n_samples]},{target},{ntokens},{nll:.10f},{inference_time:.10f},{len(x)},{n_samples}\n")
                    
    csvwriter.close()


if __name__ == "__main__":

    #####################################
    # Set up configuration for inference#
    #####################################

    _args = setup_inference_mpo_multinomial()
    print('Args: ', _args)
    
    #####################################
    # Run inference                    #
    #####################################
    
    main(_args)
    
    