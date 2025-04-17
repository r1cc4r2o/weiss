import sys
sys.path.append('../')

import os
import builtins
import importlib
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import sentencepiece as spm
from hostlist import expand_hostlist

import torch
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from src.const import *
from src.setup import setup_training_nlp
from src.module.loss import get_loss_eval_fn_nlp
from src.utils import Logger, ChekpointMonitor, del_list_inplace

# import wandb
# wandb.login(key='key-id')

########################################
### Set the seed for reproducibility ###
########################################

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

        

def main(args):

    #####################################
    # Load the classes and functions    #
    #####################################
    
    model_name = mpomodel2nlpmodel[args.type_model] 
    model_class = getattr(importlib.import_module(f"src.model.{modelname2file[args.type_model]}"), args.type_model)
    loss_fn = getattr(importlib.import_module(f"src.module.loss"), f"get_loss_fn_{modelname2file[args.type_model]}")
    dataset_class = getattr(importlib.import_module(f"src.module.data"), "PairedDataset")
    metrics = getattr(importlib.import_module(f"src.module.metrics"), "get_metrics")
    
    #####################################
    # Load the vocabulary               #
    #####################################
    
    vocabulary = spm.SentencePieceProcessor()
    vocabulary.load(f"yahooqa_tok.model")
    
    
    ##################################### 
    # Set up configuration for training #
    #####################################
    
    args.world_size = int(os.environ.get("SLURM_NTASKS", args.world_size))
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        print("Multi GPU")
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif "SLURM_PROCID" in os.environ:  # for slurm scheduler
            args.rank = int(os.environ["SLURM_PROCID"])
            args.gpu = args.rank % torch.cuda.device_count()
            args.gpu = int(os.environ.get("SLURM_LOCALID", args.gpu))
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    else:
        print("Single GPU")
        args.gpu = 0 if torch.cuda.device_count() > 0 else None
        args.rank = 0

    #####################################
    # Load the model and the dataset    #
    #####################################
    
    checkpoint = None
    if checkpoint is None:
        model = model_class(vocabulary_size=len(vocabulary)) if model_name == "Text2Text" else model_class(vocabulary_size=len(vocabulary), hidden_ae_dim=args.b)
    else:
        model = model_class.load_from_file(checkpoint, device="cpu")

    master_node = args.rank == 0

    ### suppress printing if not on master gpu ###
    ### maybe a bettter way to do it
    if not master_node:

        def dummy_print(*args):
            pass

        class DummyClass:
            def __init__(self):
                pass

            def dummy_method(self, *args, **kwargs):
                pass

        builtins.print = dummy_print
        train_logger = DummyClass()
        val_logger = DummyClass()
        writer = DummyClass()
        writer.add_scalar = writer.dummy_method
        train_logger.info = train_logger.dummy_method
        val_logger.info = val_logger.dummy_method
        run_name = ""
        save_path = ""
        metrics = DummyClass()
        metrics.compute = metrics.dummy_method
        pbar = range(args.epochs)
    else:
        run_name = model_name
        run_name += '_'
        run_name += datetime.now().strftime("%Y-%m-%d %H:%M").replace(" ", "_")
        save_path = Path(f"../priors/nlp/{run_name}_bottleneck_{args.b}")
        save_path.mkdir(parents=True)
        checkpoint_monitor = ChekpointMonitor(f"../priors/nlp/{run_name}_bottleneck_{args.b}", model_name)
        checkpoint_monitor_t10 = ChekpointMonitor(f"../priors/nlp/{run_name}_bottleneck_{args.b}", model_name+"_t1.0")
        writer = SummaryWriter(save_path / "logs")
        train_logger = Logger("train", log_dir=save_path / "logs")
        val_logger = Logger("val", log_dir=save_path / "logs")
        pbar = tqdm(range(args.epochs))
        
        # project = 'NLP'
        # name = args.type_model+'_l=40.0_b='+str(args.b)
        # run = wandb.init(
        #     project=project,
        #     name=name,
        #     config=args,
        #     job_type='train'
        # )
        # run.config.update(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.to(args.gpu)
        model.device = args.device
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_no_ddp = model.module
        else:
            model_no_ddp = model
    else:
        model.to(args.device)
        model_no_ddp = model

    model_no_ddp.device = args.device
    model_no_ddp.vocabulary = vocabulary
    model_no_ddp._lambda = args.lambda_
    
    #####################################
    # Load the dataset                  #
    #####################################
    
    dataset_train = torch.load(f"../dataset/yahoo_answers_qa_train.pt")
    idx_valid = torch.load(f"../dataset/yahoo_answers_qa_valid_idx_train.pt")
    dataset_valid = [dataset_train[_idx_] for _idx_ in idx_valid]
    dataset_subset_valid = dataset_valid[:128]
    del_list_inplace(dataset_train, idx_valid)
    dataset_train = dataset_train
    args.workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    train_dataset = dataset_class(dataset_train)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        shuffle_training = False
    else:
        train_sampler = None
        shuffle_training = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_training,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=dataset_class.collate_fn,
        persistent_workers=args.workers > 0,
    )
    train_logger.info(f"train data loaded with size : {len(train_dataset)}")

    val_dataset = dataset_class(dataset_valid)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset_class.collate_fn,
    )
    val_logger.info(f"val data loaded with size : {len(val_dataset)}")
    
    val_dataset_subset = dataset_class(dataset_subset_valid)
    val_loader_subset = torch.utils.data.DataLoader(
        val_dataset_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset_class.collate_fn,
    )
    val_logger.info(f"val subset data loaded with size : {len(val_dataset_subset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_batches = len(train_dataset) // args.batch_size
    scheduler_epoch = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,  
        total_steps=args.epochs * n_batches,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
    )
    torch.backends.cudnn.benchmark = True
    
    #####################################
    # Training loop                     #
    #####################################
    
    
    step = 0 
    for epoch in pbar:
        model_no_ddp.train()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        for batch_id, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = loss_fn(model_no_ddp, batch)
            loss.backward()
            optimizer.step()
            
            if master_node:
                pbar.set_description(
                    f"Epoch:[{epoch}], batch:[{batch_id}/{n_batches}], train loss :{loss.item():.3f}"
                )
                train_logger.info(
                    f"Epoch:[{epoch}], batch:[{batch_id}/{n_batches}], train loss :{loss.item():.3f}"
                )
                # if (step + 1) % 100 == 0: 
                #     wandb.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
                    
            writer.add_scalar("total_loss_train", loss, step)
            writer.add_scalar("lr", scheduler_epoch.get_last_lr()[0], step)
            step = step + 1
            
            scheduler_epoch.step()
                
        if master_node:
            eval_loss, eval_loss_m, eval_ppl_m, metrics, metrics_m, _metrics_full_corpora = get_loss_eval_fn_nlp(model_no_ddp, val_loader, 
                                                                                                                  val_loader_subset, model_name, 
                                                                                                                  vocabulary)
            # add metrics to log k: v
            str_metrics = " "
            for k, v in metrics.items():
                str_metrics += f"{k}: {v:.3f} "
            for t in metrics_m.keys():
                for k, v in metrics_m[t].items():
                    if k not in ['example']:
                        str_metrics += f"{k} m t{t}: {v:.3f} "
            val_logger.info(
                f"Epoch: {epoch:04d}, val_loss: {eval_loss:.3f} {str_metrics}"
            )
            for t in _metrics_full_corpora.keys():
                sample_multinomial_logger = Logger(f"epoch_{epoch:03d}_multinomial_t{t}_test.csv", log_dir=save_path / "logs")
                strings_metrics, cols = [], ""
                for k, v in _metrics_full_corpora[t].items():
                    if k not in ['example']:
                        if k in ['ppl', 'nll']:
                            cols += f"{k}Ϟ"    
                            if strings_metrics == []:
                                for i, v_i in enumerate(v):
                                    strings_metrics.append(f"{v_i:.3f}Ϟ")
                            else:
                                for i, v_i in enumerate(v):
                                    strings_metrics[i] += f"{v_i:.3f}Ϟ"
                    else:
                        example = v
                        
                
                cols = "xϞyϞypϞ" + cols
                sample_multinomial_logger.info(cols) # write the columns
                for _s, (x_str_sample, y_str_sample, yp_str_sample) in zip(strings_metrics, example): # log the sampled molecules with the metrics
                    sample_multinomial_logger.info(f"{x_str_sample}Ϟ{y_str_sample}Ϟ{yp_str_sample}Ϟ{_s}")
            
            del metrics, metrics_m, _metrics_full_corpora
            # check if is the best model
            checkpoint_monitor.save(model_no_ddp, eval_loss, step, epoch)
            checkpoint_monitor_t10.save(model_no_ddp, eval_loss_m[1.0], step, epoch)
            
            if master_node:
                del eval_loss, eval_loss_m, eval_ppl_m
                
                
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    HOSTNAMES = expand_hostlist(os.environ.get("SLURM_JOB_NODELIST", "localhost"))
    os.environ["MASTER_ADDR"] = HOSTNAMES[0]
    os.environ["MASTER_PORT"] = "12345"
    args = setup_training_nlp()
    main(args)

