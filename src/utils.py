
import os

import torch

import rdkit
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from typing import List


#############################################

class Logger:
    def __init__(self, name, log_dir):
        self.log_dir = log_dir
        self.name = name
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        with open(self.log_dir / f"{self.name}", "w") as f:
            f.write("")

    def info(self, message):
        with open(self.log_dir / f"{self.name}", "a") as f:
            f.write(f"{message}\n")
            
    def close(self):
        pass

#############################################


def tanimoto_similarity(smi1: str, smi2: str, radius: int = 2):
    """ Compute the tanimoto similarity between two smiles
    
    Args:
        smi1 (str): smiles 1
        smi2 (str): smiles 2
        
    Returns:
        float: tanimoto similarity
s    
    """
    try:
        mol1 = rdkit.Chem.MolFromSmiles(smi1)
        mol2 = rdkit.Chem.MolFromSmiles(smi2)
        
        if mol2 is None or mol1 is None: # if one of them is invalid
            return 0.0
        
        fp1 = rdMolDescriptors.GetMorganFingerprint(mol1, radius=radius)
        fp2 = rdMolDescriptors.GetMorganFingerprint(mol2, radius=radius)
        return rdkit.DataStructs.TanimotoSimilarity(fp1, fp2)
    
    except: # if something goes wrong
        return 0.0
    
    
#############################################


def kl_loss_vae(mu, logvar, training=False, min_z=0.5):
    # https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    dimensionwise_loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    if min_z and training:
        dimensionwise_loss[dimensionwise_loss < min_z] = min_z
    loss = dimensionwise_loss.sum(-1)
    return loss


#############################################


def compute_grad_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        return 0.0
    total_norm = sum([torch.sum(p.grad.detach() ** 2).item() for p in parameters]) ** 0.5
    return total_norm


#############################################


class ChekpointMonitor:
    def __init__(self, save_path, model_name):
        self.save_path = save_path
        self.best_path_1 = save_path + f"/{model_name}_best_model_1.pt"
        self.best_path_2 = save_path + f"/{model_name}_best_model_2.pt"
        self.best_path_3 = save_path + f"/{model_name}_best_model_3.pt"
        
        self.best_loss_1 = 1e9
        self.best_loss_2 = 1e9
        self.best_loss_3 = 1e9
        
        self.model_name = model_name
        
    def save(self, model, loss, step, epoch):
        if loss < self.best_loss_1:
            self.best_loss_1 = loss
            torch.save({"model": model.state_dict(), "step": step, "epoch": epoch, "loss": loss}, self.best_path_1)
        if loss < self.best_loss_2:
            self.best_loss_2 = loss
            torch.save({"model": model.state_dict(), "step": step, "epoch": epoch, "loss": loss}, self.best_path_2)
        if loss < self.best_loss_3:
            self.best_loss_3 = loss
            torch.save({"model": model.state_dict(), "step": step, "epoch": epoch, "loss": loss}, self.best_path_3)
        
        torch.save({"model": model.state_dict(), "step": step, "epoch": epoch, "loss": loss}, self.save_path + f"/{self.model_name}_last_model.pt")
        
    def get_best(self):
        return torch.load(self.best_path_1)
    
    
    
#############################################



def del_list_inplace(l, id_to_del):
    for i in sorted(id_to_del, reverse=True):
        del(l[i])
        
        
#############################################

        
def nlls2ppls(nlls):
    return torch.exp(nlls.sum(-1) / (nlls != 0.0).sum(-1))


#############################################


def tok2str(x, vocab):
    idx_x_i = torch.where(x == 2)[0]
    if len(idx_x_i) > 0:
        if len(x) > idx_x_i[0]+1:
            x[idx_x_i[0]+1:] = 0
    x = x.numpy()[x.numpy()!=0.0]
    out = vocab.decode(x.tolist())
    return out


#############################################


def write_to_txt_file_smiles_data(filename: str, molecules: List[str]) -> None:
    with open(filename, 'w') as f:
        for mol in molecules:
            f.write(f'{mol}\n')
            
            
#############################################


