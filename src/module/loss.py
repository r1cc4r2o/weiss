# This file contains the loss functions for the models
# WEISS, WEISSB, Mol2Mol, LSTM, VAE

import torch
import numpy as np

# import wandb

from src.utils import kl_loss_vae
from src.module.kernel import imq_kernel
from src.module.metrics import compute_metrics


loss_fn = torch.nn.NLLLoss(reduction="none", ignore_index=0)




####################################################################################################




def get_loss_fn_weiss(model: torch.nn.Module, batch: torch.Tensor, NLLsOnly: bool = False) -> torch.Tensor:
    """ It returns the loss function for the model
    Mol2MolWEISS 
    
    Args:
        model: torch.nn.Module The model to train
        batch: torch.Tensor The batch of data
        
    Returns:
        loss: torch.Tensor The loss value
        
    """
    src, trg = batch
    src = src.to(model.device).long()
    trg = trg.to(model.device).long()
    
    z_t_s, z_s_s = model.get_z(src, src == 0, trg, trg == 0)
    yp = model(src, src == 0, trg[:, :-1], trg[:, :-1] == 0, z=[z_t_s, z_s_s])
    loss = loss_fn(yp.transpose(1, 2), trg[:, 1:]).sum(-1).mean()
    if NLLsOnly:
        return loss    
    mmd_loss = imq_kernel(z_t_s, torch.randn_like(z_t_s))
    return loss + mmd_loss * model._lambda + (z_s_s**2).mean() * model._lambda




####################################################################################################




def get_loss_fn_weissb(model: torch.nn.Module, batch: torch.Tensor, NLLsOnly: bool = False) -> torch.Tensor:
    """ It returns the loss function for the model
    Mol2MolWEISS with the bias term
    
    Args:
        model: torch.nn.Module The model to train
        batch: torch.Tensor The batch of data
        
    Returns:
        loss: torch.Tensor The loss value
        
    """
    src, trg = batch
    src = src.to(model.device).long()
    trg = trg.to(model.device).long()
    
    z, _ = model.get_z(src, src == 0, trg, trg == 0)
    yp = model(src, src == 0, trg[:, :-1], trg[:, :-1] == 0, z=z)
    loss = loss_fn(yp.transpose(1, 2), trg[:, 1:]).sum(-1).mean()
    if NLLsOnly:
        return loss
    mmd_loss = imq_kernel(z, torch.randn_like(z))    
    return loss + mmd_loss * model._lambda 
    
    
    

####################################################################################################




def get_loss_fn_mol2mol(model: torch.nn.Module, batch: torch.Tensor, NLLsOnly: bool = False) -> torch.Tensor:
    """ It returns the loss function for the model
    Mol2Mol Transformer
    
    Args:
        model: torch.nn.Module The model to train
        batch: torch.Tensor The batch of data
        
    Returns:
        loss: torch.Tensor The loss value
        
    """
    src, trg = batch
    src = src.to(model.device).long()
    trg = trg.to(model.device).long()
    
    yp = model(src, src == 0, trg[:, :-1], trg[:, :-1] == 0)
    return loss_fn(yp.transpose(1, 2), trg[:, 1:]).sum(-1).mean()



####################################################################################################



def get_loss_fn_lstm(model: torch.nn.Module, batch: torch.Tensor, NLLsOnly: bool = False) -> torch.Tensor:
    """ It returns the loss function for the model
    Mol2Mol Transformer
    
    Args:
        model: torch.nn.Module The model to train
        batch: torch.Tensor The batch of data
        
    Returns:
        loss: torch.Tensor The loss value
        
    """
    src, trg = batch
    src = src.to(model.device).long()
    trg = trg.to(model.device).long()
    
    yp = model(src, trg[:, :-1])
    return loss_fn(yp.transpose(1, 2), trg[:, 1:]).sum(-1).mean()




####################################################################################################




def get_loss_fn_vae(model: torch.nn.Module, batch: torch.Tensor, NLLsOnly: bool = False) -> torch.Tensor:
    """ It returns the loss function for the model
    Mol2MolWEISS with the bias term
    
    Args:
        model: torch.nn.Module The model to train
        batch: torch.Tensor The batch of data
        
    Returns:
        loss: torch.Tensor The loss value
        
    """
    src, trg = batch
    src = src.to(model.device).long()
    trg = trg.to(model.device).long()
    
    yp, mu, logvar = model(src, src == 0, trg[:, :-1], trg[:, :-1] == 0)
    recon_loss = loss_fn(yp.transpose(1, 2), trg[:, 1:]).sum(-1).mean()
    if NLLsOnly:
        return recon_loss
    kl_loss = kl_loss_vae(mu, logvar, training=True).mean()
    return recon_loss + kl_loss
    
    
    
    
    
####################################################################################################



def eval_fn_multinomial_nlp(model: torch.nn.Module, src: torch.Tensor, 
                            trg: torch.Tensor, vocabulary, temperature=0.1):
    """ It evaluates the model using multinomial sampling

    Args:
        model: torch.nn.Module: The model to evaluate
        src: torch.Tensor: The source tensor
        trg: torch.Tensor: The target tensor
        vocabulary: dict: The vocabulary
        temperature: float: The temperature for the softmax
        
    Returns:
        loss_m: torch.Tensor: The loss value
    
    """
    loss_m = 0.0 
    ppl_m = 0.0
    
    model.eval()
    with torch.no_grad():
        src_enc, _ = model.encode(src, src == 0)
        yp_m = torch.ones((len(src), 1)).long().to(src.device)
        nlls_m = []
        break_condition = torch.zeros(len(src), dtype=torch.bool, device=src.device)
        for i in range(trg.shape[-1]-1):
            log_probs = model.decode(src_enc, yp_m, yp_m == 0)[:, -1]
            # multinomial sampling
            next = torch.multinomial(torch.softmax(log_probs / temperature, dim=-1), 1).ravel() * (
                ~break_condition
            )
            nlls = -log_probs[torch.arange(len(log_probs)), next] * (
                1 - break_condition.float()
            )
            nlls_m = nlls_m + [nlls]
            yp_m = torch.cat([yp_m, next.unsqueeze(-1)], dim=-1)
            break_condition = torch.logical_or(break_condition, next == 2)
            
            if torch.all(break_condition):
                break
        
        nlls_m = torch.stack(nlls_m, dim=-1)
        loss_m = nlls_m.sum(-1)
        ppl_m = torch.exp(nlls_m.sum(-1) / (nlls_m != 0.0).sum(-1))
    
    return loss_m, ppl_m, yp_m


####################################################################################################


def get_loss_eval_fn_nlp(model: torch.nn.Module, eval_loader: torch.utils.data.DataLoader, 
                         subset_eval: torch.utils.data.DataLoader, model_type: str, vocabulary, 
                         temperature=[1.0]) -> torch.Tensor:
    """ It logs several metrics for the model evaluation while training.
    
    Args:
        model: torch.nn.Module: The model to evaluate
        eval_loader: torch.utils.data.DataLoader: The evaluation loader
        subset_eval: torch.utils.data.DataLoader: The subset of the evaluation loader
        model_type: str: The model type
        vocabulary: dict: The vocabulary
        temperature: list: The temperature for the softmax
        
    Returns:
        loss: torch.Tensor: The loss value
        nll_multinomial: torch.Tensor: The negative log likelihood for the multinomial sampling
        ppl_multinomial: torch.Tensor: The perplexity value for the multinomial sampling
        _metrics: dict: The metrics for the argmax
        _metrics_m: dict: The metrics for the multinomial sampling
        _metrics_full_corpora: dict: The metrics for the full corpora
        
    """
    model.eval()
    with torch.no_grad():
        loss, losses = 0.0, [] # argmax
        ppl = [] # argmax
        loss_m = 0.0, [] # multinomial
        ppl_m = 0.0, []
        _metrics, _metrics_m = {}, {} # argmax, multinomial
        
        for idx, batch in enumerate(eval_loader):
            
            src, trg = batch
            src = src.to(model.device).long()
            trg = trg.to(model.device).long()
            
            if model_type == 'Text2TextWEISS':
                z_t_s, z_s_s = model.get_z(src, src == 0, trg, trg == 0)
                yp = model(src, src == 0, trg[:, :-1], trg[:, :-1] == 0, z=[z_t_s, z_s_s])
            elif model_type == 'Text2Text':
                yp = model(src, src == 0, trg[:, :-1], trg[:, :-1] == 0)
            else:
                raise ValueError(f"Model type {model_type} not supported")
            
            loss = loss_fn(yp.transpose(1, 2), trg[:, 1:])
            losses.append(loss.sum(-1).mean().item())
            ppl.append(torch.exp(loss.sum(-1) / (loss != 0.0).sum(-1)).mean().item())
            
            if idx == 0: # eval on the subset_eval
                
                for _idx, _batch in enumerate(subset_eval):
                    _src, _trg = _batch
                    _src = _src.to(model.device).long()
                    _trg = _trg.to(model.device).long()
                    
                    for t in temperature:
                        
                        if t not in _metrics_m:
                            _metrics_m[t] = {}
                        loss_m, ppl_m, yp_m = eval_fn_multinomial_nlp(model, _src, _trg, vocabulary, temperature=t)
                        metrics_m, yp_y_text_m = compute_metrics(_trg[:, 1:], yp_m, _src, vocabulary)
                        for k, v in metrics_m.items():
                            if k not in _metrics_m[t]:
                                _metrics_m[t][k] = []
                            _metrics_m[t][k].append(v)
                        if 'ppl' not in _metrics_m[t]:
                            _metrics_m[t]['ppl'] = []
                            _metrics_m[t]['nll'] = []
                            _metrics_m[t]['example'] = []
                        _metrics_m[t]['ppl'].append(ppl_m.tolist())
                        _metrics_m[t]['nll'].append(loss_m.tolist()) 
                        _metrics_m[t]['example'].append(yp_y_text_m)
                   
                if _idx > 0: # flatten the list
                    for t in temperature:
                        _metrics_m[t]['nll'] = [item for sublist in _metrics_m[t]['nll'] for item in sublist]
                        _metrics_m[t]['ppl'] = [item for sublist in _metrics_m[t]['ppl'] for item in sublist]
                        _metrics_m[t]['example'] = [item for sublist in _metrics_m[t]['example'] for item in sublist]
                
                # copy the dict
                _metrics_full_corpora = {t: {k: v for k, v in _metrics_m[t].items()} for t in temperature}
            
            metrics, _ = compute_metrics(trg[:, 1:], yp.argmax(-1), src, vocabulary)
            for k, v in metrics.items():
                if k not in _metrics:
                    _metrics[k] = []
                _metrics[k].append(v)
                
        for k, v in _metrics.items():
            _metrics[k] = np.mean(v)
            # wandb.log({
            #     k: np.mean(v)
            # })
        # wandb.log({
        #     'valid_loss_greedy': np.mean(losses),
        #     'valid_ppl_greedy': np.mean(ppl)
        # })
            
        for t in temperature:
            for k, v in _metrics_m[t].items():
                if k not in ['example']:
                    _metrics_m[t][k] = np.mean(v)
                    k = f"{k}_t{t}_m"
                    # wandb.log({
                    #     k: np.mean(v)
                    # })
                    
            
    return np.mean(losses), {t: np.mean(_metrics_m[t]['nll']) for t in temperature}, {t: np.mean(_metrics_m[t]['ppl']) for t in temperature}, _metrics, _metrics_m, _metrics_full_corpora