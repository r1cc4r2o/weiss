import sys
sys.path.append('../')

import importlib

import torch
import os
import numpy as np
from tqdm import tqdm   

from pathlib import Path
from datetime import datetime

from src.const import *
from src.utils import Logger
from src.setup import setup_train_mpo
from src.module.optim import NoamOptimizer


####################################
# Set the seed for reproducibility #
####################################

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def main(_args):
    
    #############################################
    # Load the model, loss and dataset classes  #
    #############################################
    
    model_class = getattr(
        importlib.import_module(f"src.model.{modelname2file[_args.model_name]}"), _args.model_name
    )
    loss_fn = getattr(importlib.import_module(f"src.module.loss"), f"get_loss_fn_{modelname2file[_args.model_name]}")
    dataset_class = getattr(importlib.import_module(f"src.module.data"), 'PairedDataset')

    run_name = _args.model_name + '_'
    run_name += datetime.now().strftime("%Y-%m-%d %H:%M").replace(" ", "_")
    save_path = Path(f"results/{run_name}_b={_args.hidden_ae_dim}_l={_args._lambda}")
    save_path.mkdir(parents=True)
    logger = Logger("train", log_dir= save_path / 'logs')
    logger.info(f"train_loss, valid_loss, epoch")


    ###########################################
    # Load the data and create the vocabulary #
    ###########################################

    vocabulary_class = getattr(importlib.import_module(f"src.module.vocab"), 'Vocabulary')
    dict_vocabulary = torch.load(f"{_args.path_vocabulary}")
    vocabulary = vocabulary_class.load_from_dictionary(dict_vocabulary)

    data = torch.load(f"{_args.path_dataset}")
    train_dataset = dataset_class(data['data_train'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=_args.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=dataset_class.collate_fn,
    )

    valid_dataset = dataset_class(data['data_valid'])
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=_args.batch_size,
        drop_last=True,
        shuffle=False,
        collate_fn=dataset_class.collate_fn,
    )

    #####################################
    # Initialize the model and optimizer #
    #####################################

    if (_args.model_name == 'Mol2MolVAE') or (_args.model_name == 'Mol2MolWEISSWithB') or (_args.model_name == 'Mol2MolWEISS'):
        model = model_class(vocabulary_size=len(vocabulary), hidden_ae_dim=_args.hidden_ae_dim)
    else:
        model = model_class(vocabulary_size=len(vocabulary))
        
    model.device = _args.device
    model._lambda = _args._lambda
    model.to(_args.device)

    opt = torch.optim.Adam(model.parameters(), lr=_args.lr) if _args.optimizer == 'Adam' else NoamOptimizer(model.parameters(), 256)


    #####################################
    # Run the training loop             #
    #####################################
    
    best_epoch, best_loss = 0, 1e9
    pbar = tqdm(range(_args.n_epochs))
    for epoch in pbar:
        
        model.train() # Ensure the model is in training mode
        
        losses = []
        for bid, batch in enumerate(train_loader):
            opt.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_description(
                f"Loss={np.mean(losses):.2f} Batch={bid:d}/{len(train_loader):d} Best Epoch={best_epoch:d}"
            )

        model.eval() # Ensure the model is in evaluation mode
        with torch.no_grad(): # Save the best model based on the nlls on the validation set
            # This is not the best way to do this, but it is the simplest
            # Check out the article to know how to properly save the best model
            # considering the validation score we've defined
            valid_loss = np.mean([loss_fn(model, batch, NLLsOnly=True).item() for batch in tqdm(valid_loader)])
            
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            if not os.path.exists("../priors"):
                os.makedirs("../priors")
            
            if (modelname2file[_args.model_name] == 'vae') or (modelname2file[_args.model_name] == 'weissb') or (modelname2file[_args.model_name] == 'weiss'):
                torch.save(model.state_dict(), f"../priors/{_args.model_name}_b={_args.hidden_ae_dim:02d}_best.pt")
            else:
                torch.save(model.state_dict(), f"../priors/{_args.model_name}_best.pt")
                
        if not os.path.exists(f"../priors/{_args.model_name}"):
            os.makedirs(f"../priors/{_args.model_name}")
        if (modelname2file[_args.model_name] == 'vae') or (modelname2file[_args.model_name] == 'weissb') or (modelname2file[_args.model_name] == 'weiss'):
            torch.save(model.state_dict(), f"../priors/{_args.model_name}/{_args.model_name}_b={_args.hidden_ae_dim:02d}_epoch={epoch:02d}_validloss={valid_loss:.3f}.pt")
        else:
            torch.save(model.state_dict(), f"../priors/{_args.model_name}/{_args.model_name}_epoch={epoch:02d}_validloss={valid_loss:.3f}.pt")
            
        logger.info(f"{np.mean(losses):.3f}, {valid_loss:.3f}, {epoch:d}")
        
    logger.close()
    
    
if __name__ == "__main__":
    
    ############################
    # Run training and logging #
    ############################

    _args = setup_train_mpo()

    print('Args: ', _args)
    main(_args)


