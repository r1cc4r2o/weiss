
import os
import time
import torch
import numpy as np
import pandas as pd

from .util import imq_kernel
from molbart.models import BARTModel



############################################################################################################
# WEISS BART withoout bias
############################################################################################################

class BARTweiss(BARTModel):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocabulary_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        num_predictions: int = 1,
        schedule: str = "cycle",
        warm_up_steps: any = None,
        dropout: float = 0.1,
        lambda_: float = 20,
        hidden_ae_dim: int = 4,
        **kwargs,
    ):
        super().__init__(
            decode_sampler,
            pad_token_idx,
            vocabulary_size,
            d_model,
            num_layers,
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs,
        )

        # This deep neural net aims to learn a mapping between the out of the
        # Transformer encoder and lower-dimensional representation z that it 
        # should be sampled from a ~ N(0, I) distribution. 
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model//2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model//2, d_model//4),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model//4, hidden_ae_dim),
        )
        
        # Interpolation function between two vectors 
        self.interpolate_f = torch.nn.Linear(d_model*2, d_model, bias=False)
        
        # Re-map z to the original decoder prior distribution
        self.inv_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_ae_dim, d_model//4),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model//4, d_model//2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model//2, d_model),
            # torch.nn.LayerNorm(d_model, **layer_norm_kwargs),
        )

        # Smothing of the convergence, it is a hyperparameter
        # to flatten the energy function of the MMD and avoid
        # a too quick convergence towards a wired local minimum
        self.lambda_ = lambda_
        
        # SP hidden dimension
        self.hidden_ae_dim = hidden_ae_dim
        
        # Numeber of molecules sampled for each input
        # molecule in one single step
        self.num_predictions = num_predictions
        
        self._z = None
        
        
    def init_z(self, batch_size, device_z):
        self._z = torch.randn(self.num_predictions, self.hidden_ae_dim).to(device_z) \
                .unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, self.hidden_ae_dim)
                

    def training_step(self, batch, batch_idx):
        """
        Lightning training step
        
        Args:
            batch (dict): Input given to model
            batch_idx (int): Index of batch
            
        Returns:
            loss (singleton tensor)
            
            
        """
        
        self.train()

        model_output, z_t_s, z_s_s = self.forward(batch)
        log_lhs, loss, mmd = self._calc_loss(batch, model_output, z_t_s, z_s_s)

        self.log("training_loss", loss, on_step=True, logger=True, sync_dist=True)
        self.log("training_log_lhs", log_lhs, on_step=True, logger=True, sync_dist=True)
        self.log("training_mmd", mmd, on_step=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        """
        Lightning validation step
        
        Args:
            batch (dict): Input given to model
            batch_idx (int): Index of batch
            
        Returns:    
            metrics (dict): Dictionary containing validation metrics
                                - validation_loss: cross-entropy loss
                                - val_token_accuracy: token accuracy GT - generated
        
        
        """
        
        
        self.eval()

        with torch.no_grad():
            
            model_output, z_t_s, z_s_s = self.forward(batch)
            target_smiles = batch["target_smiles"]

            log_lhs, loss, mmd = self._calc_loss(batch, model_output, z_t_s, z_s_s)
            token_acc = self._calc_token_acc(batch, model_output)

            # sampled_smiles, log_likelihood_sampled = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
            # sampled_metrics = self.sampler.compute_sampling_metrics(sampled_smiles, target_smiles)

            metrics = {
                "validation_log_lhs": log_lhs,
                "validation_loss": loss,
                "validation_mmd": mmd,
                "val_token_accuracy": token_acc,
                # "log_likelihood_sampled": np.mean(np.array(log_likelihood_sampled)),
                "(z_s_s**2).mean()": (z_s_s**2).mean(),
            }

            # metrics.update(sampled_metrics)
            
        return metrics
    

    def test_step(self, batch, batch_idx, output_scores_efficiency=None):
        """
        Lightning test step
        
        Args:
            batch (dict): Input given to model
            batch_idx (int): Index of batch
            output_scores_efficiency (str): File name to save the output scores
            
        Returns:
            metrics (dict): Dictionary containing test metrics
                                - test_loss: cross-entropy loss
                                - test_token_accuracy: token accuracy GT - generated
                                - log_lhs: log likelihoods
                                - sampled_molecules: sampled molecules
                                - target_smiles: target smiles
                                
        
        """
        
        self.eval()
        with torch.no_grad():
            target_smiles = batch["target_smiles"]
            
            # only at the first ten batches

            if output_scores_efficiency is None:
                sampled_smiles, log_likelihoods = self.sample_molecules(batch, sampling_alg=self.test_sampling_alg, task='test')
            else:
                
                
                if os.path.exists(output_scores_efficiency):
                    df = pd.read_csv(output_scores_efficiency)
                else:
                    df = pd.DataFrame()
                    
                start = time.time()
                sampled_smiles, log_likelihoods = self.sample_molecules(batch, sampling_alg=self.test_sampling_alg, task='test')
                stop = time.time()
                
                use_cuda = torch.cuda.is_available()
                
                if use_cuda:
                    __CUDNN = torch.backends.cudnn.version()
                    N_devices = torch.cuda.device_count()
                    DeviceName = torch.cuda.get_device_name(0) # it assume all the device have the same name
                    DeviceMemoryGB = torch.cuda.get_device_properties(0).total_memory/1e9 # it assume all the device have the same amount of memory
                    
                temp = {"batch_idx": batch_idx, "time": stop-start, "use_cuda": use_cuda, "cudnn_version": __CUDNN, "n_devices": N_devices, "device_name": DeviceName, "device_memory_GB": DeviceMemoryGB}
                df = df.append(temp, ignore_index=True)
                df.to_csv(output_scores_efficiency, index=False)
                
            log_likelihoods = np.array(log_likelihoods)
            log_likelihoods = log_likelihoods.reshape(-1, self.num_predictions * self.num_beams)
            
            sampled_smiles = np.array(sampled_smiles)
            sampled_smiles = sampled_smiles.reshape(-1, self.num_predictions * self.num_beams)
            sampled_smiles = sampled_smiles.tolist()
            
        for i in range(len(sampled_smiles)):
            
            idx = np.argsort(log_likelihoods[i])[::-1]
            sampled_smiles[i] = np.array(sampled_smiles[i])[idx]
            log_likelihoods[i] = log_likelihoods[i][idx]
            
        sampled_metrics = self.sampler.compute_sampling_metrics(sampled_smiles, target_smiles)

        metrics = {
            "batch_idx": batch_idx,
            "log_lhs": log_likelihoods,
            "sampled_molecules": sampled_smiles,
            "target_smiles": target_smiles,
        }

        metrics.update(sampled_metrics)
        return metrics
    

    def _calc_loss(self, batch_input, model_output, z_t_s, z_s_s):
        """Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model
            z: sphere projection matrix

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)
        mmd_loss = imq_kernel(z_t_s, torch.randn_like(z_t_s)) # check the overlap between the two distributions
        mmd = self.lambda_ * mmd_loss + self.lambda_ * (z_s_s ** 2).mean()
        final_loss = token_mask_loss + mmd

        return token_mask_loss, final_loss, mmd
    
    

    def _calc_mask_loss(self, token_output, target, target_mask):
        """Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocabulary_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_function(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    
    def encode_validation(self, batch):
        """Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """
        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        
        S, B = encoder_input.size()
        
        encoder_embs = self._construct_input(encoder_input)
        
        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        
        if self._z is None:
            device_z = memory.device
            self.init_z(B, device_z)   
            
        # Fine-tuning 
        # memory = memory + self.inv_proj(self._z.reshape(B, self.num_predictions, self.hidden_ae_dim)[:, 0, :]).unsqueeze(0) * (1 - encoder_pad_mask.unsqueeze(2).transpose(0, 1).float())
        # return memory
        # Inference
        z_t_s = self._z.reshape(B, self.num_predictions, self.hidden_ae_dim)[:, 0, :] 
        z_s_s = torch.zeros_like(z_t_s).to(memory)
        memory = memory + (self.inv_proj(z_t_s).unsqueeze(0) - self.inv_proj(z_s_s).unsqueeze(0))
        return memory

    
    def encode(self, batch):
        """Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size * num_predictions, d_model))
            pad mask (Tensor of shape (batch_size * num_predictions, seq_len))
            
        WARNING!!!
            encoder_pad_mask where True is a padded element and False is not padded

        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1) # True is a padded element and False is not padded (nn.TransformerEncoder requires this format)
        
        # Get encoder embeddings
        encoder_embs = self._construct_input(encoder_input)
        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        
        # Get shape
        S, B, D = memory.shape
        
        # Reshape mask and encoder out
        memory = memory.unsqueeze(2).repeat(1, 1, self.num_predictions, 1).reshape(S, B * self.num_predictions, D)
        encoder_pad_mask = encoder_pad_mask.unsqueeze(1).repeat(1, self.num_predictions, 1).reshape(B * self.num_predictions, S)
        
        # Apply SP
        if self._z is None:
            device_z = memory.device
            self.init_z(B, device_z)
        
        # Fine-tuning 
        # memory = memory + self.inv_proj(self._z).unsqueeze(0) * (1 - encoder_pad_mask.transpose(0, 1).float().unsqueeze(-1))
        # return memory, encoder_pad_mask
        # Inference
        z_s_s = torch.zeros_like(self._z).to(memory)
        memory = memory + (self.inv_proj(self._z).unsqueeze(0) - self.inv_proj(z_s_s).unsqueeze(0))
        return memory, encoder_pad_mask
    
    def get_z(self, src_embs, src_pad_mask, tgt_embs, tgt_pad_mask):
        """Encode with sphere projection.

        Args: 
            src_embs: (src_len, batch_size, d_model)
            src_pad_mask: (batch_size, src_len)
            tgt_embs: (tgt_len, batch_size, d_model)
            tgt_pad_mask: (batch_size, tgt_len)

        Returns:
            memory: (src_len, batch_size, d_model)
            z: sphere projection matrix with shape (batch_size, 64)
        """
        
        # Get encoded version of the sequences
        memory = self.encoder(src_embs, src_key_padding_mask=src_pad_mask).transpose(0, 1)
        memory = memory * ( 1 - src_pad_mask.float()[...,None] )
        
        tgt_enc = self.encoder(tgt_embs, src_key_padding_mask=tgt_pad_mask).transpose(0, 1)
        tgt_enc = tgt_enc * ( 1 - tgt_pad_mask.float()[...,None] )
        
        # Get the mean of the encoded sequences        
        z_src_den = torch.sum(1 - src_pad_mask.float(), dim=1, keepdims=True)
        z_src_den = torch.maximum(z_src_den, torch.ones_like(z_src_den))
        z_src = torch.sum(memory, dim=1) / z_src_den

        z_trg_den = torch.sum(1 - tgt_pad_mask.float(), dim=1, keepdims=True)
        z_trg_den = torch.maximum(z_trg_den, torch.ones_like(z_trg_den))
        z_tgt = torch.sum(tgt_enc, dim=1) / z_trg_den
        
        z_src_trg = torch.cat([z_src, z_tgt], dim=1)
        z_src_src = torch.cat([z_src, z_src], dim=1)
        
        z_t_s = self.proj(self.interpolate_f(z_src_trg))
        z_s_s = self.proj(self.interpolate_f(z_src_src))
        
        memory = (memory + (self.inv_proj(z_t_s).unsqueeze(1) - self.inv_proj(z_s_s).unsqueeze(1))).transpose(0, 1)
        
        return memory, z_t_s, z_s_s


    def forward(self, x):
        """Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size * num_predictions)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size * num_predictions)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output"), sphere projection matrix z
        """
        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)
        
        encoder_embs = self._construct_input(encoder_input)
        decoder_embs = self._construct_input(decoder_input)

        Sm, B = decoder_input.shape
        tgt_mask = self._generate_square_subsequent_mask(Sm, device=encoder_embs.device)

        memory, z_t_s, z_s_s = self.get_z(encoder_embs, encoder_pad_mask, decoder_embs, decoder_pad_mask)
        
        model_output = self.decoder(
            decoder_embs,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=encoder_pad_mask.clone(),
        )
        token_output = self.token_fc(model_output)
        output = {"model_output": model_output, "token_output": token_output}

        return output, z_t_s, z_s_s
