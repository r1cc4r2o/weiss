
import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from datetime import datetime

import nltk
import evaluate

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from typing import List
import sentencepiece as spm
    
from src.model.mol2mol import Mol2Mol as Text2Text
from src.model.weiss import Mol2MolWEISS as Text2TextWEISS
from src.module.data import PairedDataset, SimpleDataloader
from src.utils import Logger, tok2str, nlls2ppls
from src.setup import setup_inference_multinomial_nlp

#################################################################
# Constants
#################################################################

nltk.download("punkt", quiet=True)
metric_rouge = evaluate.load("rouge")


#################################################################
# Load models                                                   #
#################################################################
    
def get_models(type_model: str, vocabulary, d_model=256,
               dim_feedforward=2048, hidden_ae_dim=16, n_layers = 6, device='cuda') -> torch.nn.Module:
    if type_model == 'Text2TextWEISS':
        model = Text2TextWEISS(vocabulary_size=len(vocabulary), num_layers=n_layers, 
                         d_model=d_model, dim_feedforward=dim_feedforward, 
                         hidden_ae_dim=hidden_ae_dim).to(device)
    elif type_model == 'Text2Text':
        model = Text2Text(vocabulary_size=len(vocabulary), num_layers=n_layers, 
                        d_model=d_model, dim_feedforward=dim_feedforward).to(device)
    else:
        raise ValueError(f'Invalid: type_model {type_model}')
    return model


#################################################################
# Custom metrics                                                #
#################################################################

def get_metrics(y, yp, vocab):
    # pad all the tokens after
    # the <eos> token
    for i in range(len(yp)):
        idx_yp_i = torch.where(yp[i] == 2)[0]
        if len(idx_yp_i) > 0:
            if len(yp[i]) > idx_yp_i[0]+1:
                yp[i][idx_yp_i[0]+1:] = 0
    candidate_corpus = [vocab.decode(p.tolist()) for p in yp]
    references_corpus = [vocab.decode(p.tolist()) for p in y]
    metrics = metric_rouge.compute(predictions=candidate_corpus, references=references_corpus, use_stemmer=True, use_aggregator=False)
    return metrics


def get_nll_src_trg(model, xc, yc, device):
    model.eval()
    with torch.no_grad():
        xc, yc = xc.to(device).long(), yc.to(device).long()
        if type_model in ['Text2TextWEISS']:
            z_t_s, _ = model.get_z(xc, xc == 0, yc, yc == 0)
            nlls_x_y = model.nll(xc, yc, z=[z_t_s, torch.zeros_like(z_t_s)])#, reduction='none')
        else:
            nlls_x_y = model.nll(xc, yc)#, reduction='none')
    return nlls_x_y


def get_x_yp_metrics_multinomial(type_model: str, model: torch.nn.Module, 
                                    x: torch.Tensor, vocab: List[str], device: str, 
                                    greedy = False, max_len = 128, n_sampled = 10,
                                    temperatures = [0.1, 0.5, 1.0, 2.0]):
    dict_results = {}
    
    for temperature in temperatures:

        model.eval()

        with torch.no_grad():
            (B, L)= x.shape
            xc= x.to(device).unsqueeze(1)
            xc = xc.repeat(1, n_sampled, 1).reshape(B * n_sampled, L).long()
            z = torch.randn(len(xc), 16).to(device)
            yp = torch.ones(len(xc), 1).long().to(device)
            break_condition = torch.zeros(len(xc), dtype=torch.bool).to(device)
            nlls = []
            
            for i in range(max_len -1):
                if type_model in ['Text2Text']:
                    log_probas = model(xc, xc == 0, yp, yp < 0)
                elif type_model in ['Text2TextWEISS']:
                    log_probas = model(xc, xc == 0, yp, yp < 0, z = [z, torch.zeros_like(z)])
                    
                if greedy:
                    yp = torch.cat([yp, log_probas.argmax(-1)[:, -1]], dim=1)
                else:
                    probs = torch.softmax(log_probas[:, -1] / temperature, -1)
                    pred = torch.multinomial(probs, 1).ravel() * (~break_condition)
                    nlls.append(-log_probas[:, -1][torch.arange(len(log_probas)), pred] * (
                        1 - break_condition.float()
                    ))
                    break_condition = torch.logical_or(break_condition, pred == 2)
                    yp = torch.cat([yp, pred.unsqueeze(-1)], dim=1)
                    
                if torch.all(break_condition):
                    break
                        
            nlls = torch.stack(nlls, dim=1)
            
            dict_results[temperature] = {
                'xc': xc,
                'yp': yp,
                'nlls': nlls,
                'nlls2ppls': nlls2ppls(nlls)
            }
    
    return dict_results



#################################################################
# Load data                                                     #
#################################################################


def get_testset_loader(vocab: spm.SentencePieceProcessor, N: int = 10):
    test = torch.load(f'../dataset/yahoo_answers_qa_test.pt')
    dict_samples = {}
    for x, y in test:
        x_dec = vocab.decode(x.tolist())
        if x_dec not in dict_samples:
            dict_samples[x_dec] = []
        dict_samples[x_dec].append(y)
    test_samples = sorted(dict_samples.items(), key=lambda x: len(x[1]), reverse=True)
    if N is None:
        dataset_test = [(torch.tensor([vocab.bos_id()] + vocab.encode(k) + [vocab.eos_id()], dtype=torch.int16), _v_) for k, v in test_samples for _v_ in v]
        return dataset_test, [torch.tensor([vocab.bos_id()] + vocab.encode(k) + [vocab.eos_id()], dtype=torch.int16) for k, v in test_samples]
    else:
        dataset_test = [(torch.tensor([vocab.bos_id()] + vocab.encode(k) + [vocab.eos_id()], dtype=torch.int16), _v_) for k, v in test_samples[:N] for _v_ in v]
        return dataset_test, [torch.tensor([vocab.bos_id()] + vocab.encode(k) + [vocab.eos_id()], dtype=torch.int16) for k, v in test_samples[:N]]

def get_loader(dataset_test: List, batch_size: int = 1):
    return torch.utils.data.DataLoader(PairedDataset(dataset_test), batch_size=batch_size, shuffle=False, collate_fn=PairedDataset.collate_fn)

def get_simple_loader(dataset_test: List, batch_size: int = 1):
    return torch.utils.data.DataLoader(SimpleDataloader(dataset_test), batch_size=batch_size, shuffle=False, collate_fn=SimpleDataloader.collate_fn)
    


if __name__ == '__main__':
    
    #################################################################
    # Setup                                                         #
    #################################################################
    
    _args = setup_inference_multinomial_nlp()
    N = _args.N
    K = _args.K
    base_path = _args.base_path
    device = _args.device
    run_name = _args.run_name
    
    #################################################################
    # Load the vocabulary                                           #
    #################################################################
    
    vocab = spm.SentencePieceProcessor(model_file=f"yahooqa_tok.model")


    #################################################################
    # Logger                                                        #
    #################################################################
    
    model_name = run_name
    run_name += '_'
    run_name += datetime.now().strftime("%Y-%m-%d %H:%M").replace(" ", "_")
    save_path = Path(f"../results/nlp/inference/{run_name}_logs")
    save_path.mkdir(parents=True)
    logger = Logger("inference.csv", log_dir=save_path)
    logger_gt = Logger("inference_gt.csv", log_dir=save_path)

    cols = 'methodϞtemperatureϞxϞypϞnlls'
    logger.info(cols)
    cols = 'methodϞxϞyϞnlls'
    logger_gt.info(cols)
                    
               
    #################################################################
    # Load the model and the test set                               #
    #################################################################
         
    dataset_test, dataset_src_test = get_testset_loader(vocab=vocab, N=N)

    type_model = _args.path.split('/')[-1].split('_')[0]
    test_loader = get_simple_loader(dataset_src_test)

    model = get_models(type_model=type_model, vocabulary=vocab, device=device)
    model.load_state_dict(torch.load(_args.path, map_location=device)['model'])
    model = model.to(device)

    for idx, x in tqdm(enumerate(test_loader)):
        dict_o2m = get_x_yp_metrics_multinomial(type_model=type_model, model=model, x=x, vocab=vocab, device=device, n_sampled=K)
        key, value = (type_model, idx), dict_o2m
        for k, v in value.items():
            for i in range(len(v['yp'])):
                logger.info(f"{key[0]}Ϟ{k}Ϟ{tok2str(v['xc'][i].cpu(), vocab)}Ϟ{tok2str(v['yp'][i].cpu(), vocab)}Ϟ{v['nlls'][i].sum().cpu().item()}")
                    
    # nlls_x_y = []
    test_loader = get_loader(dataset_test, batch_size=64)
    for idx, (x, y) in tqdm(enumerate(test_loader)):
        items_nll = get_nll_src_trg(model, x, y, device=device)
        for i, item in enumerate(items_nll):
            # nlls_x_y.append({'nll_x_y': item.cpu().item(), 'type_model': 'o2m', 'x': tok2str(x[i].cpu(), vocab), 'y': tok2str(y[i].cpu(), vocab)})
            logger_gt.info(f"{type_model}Ϟ{tok2str(x[i].cpu(), vocab)}Ϟ{tok2str(y[i].cpu(), vocab)}Ϟ{item.cpu().item()}")
