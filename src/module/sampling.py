
import torch
import torch.utils.data as tud


class NodeO2O:
    def __init__(
        self,
        model: torch.nn.Module,
        x: tuple,
        vocabulary: any, 
        device: str,
        type_of_tokenizer: str,
        data_device: str = "cpu", 
        batch_size: int = 64) -> None:
        """
        Initialize a Node used for autoregression
        predictions, such as greedy search, multinomial
        sampling, or beam search

        params: model: torch.nn.Module: The model to use for autoregression.
        params: x: tuple(torch.tensor,): The input data.
        params: vocabulary: Vocabulary: The vocabulary object.
        params: device: torch.device, str: The device where to place the model.
        params: type_of_tokenizer: str: The type of tokenizer to use.
        params: data_device: torch.device, str: The device where to place the data.
        params: batch_size: int: The internal batch size used for the beam search.

        return: None

        """
        assert isinstance(device, torch.device) or isinstance(device, str)
        assert isinstance(data_device, torch.device) or isinstance(data_device, str)

        if isinstance(device, str):
            device = torch.device(device)

        if isinstance(data_device, str):
            data_device = torch.device(data_device)

        self.model = model
        self.device = device
        self.data_device = data_device
        
        src, src_mask = x

        self.batch_size = batch_size  # min(batch_size, len(src))
        self.num_heads = model.transformer.nhead

        with torch.no_grad():
            self.model = self.model.eval()

            if next(self.model.parameters()).device != self.device:
                self.model = self.model.to(self.device)
            if src.device != self.device:
                src = src.to(self.device)
            if src_mask.device != self.device:
                src_mask = src_mask.to(self.device)


            # src_mask = self.compute_padding_mask(src_mask) # shape: (B*H, N, N)
            src = self.model.pe(self.model.token_emb(src)) # shape: (B, N, D)
            src_m = self.compute_padding_mask(src_mask)
            self.x = self.model.transformer.encoder(src, src_m).detach()
            self.x_mask = src_mask.detach()

            if self.x.device != self.data_device:
                self.x = self.x.to(self.data_device)
            if self.x_mask.device != self.data_device:
                self.x_mask = self.x_mask.to(self.data_device)

        self.vocabulary = vocabulary

        # add element inside the object
        if type_of_tokenizer == 'llama2':
            self.bos_token = self.vocabulary.bos_id
            self.eos_token = self.vocabulary.eos_id
            self.len_vocab = vocabulary.n_words

        elif type_of_tokenizer == 'reinvent':
            # pad=0, start=1, end=2, unk=3 (unknow token)
            self.bos_token = 1
            self.eos_token = 2
            self.len_vocab = len(vocabulary.tokens())

        self.y = (
            torch.ones((len(self.x), 1), dtype=torch.long) * self.bos_token
        )
        self.y = self.y.detach()

        if self.y.device != self.data_device:
            self.y = self.y.to(self.data_device)

        self.ll_mask = torch.tensor([False])
        self.pos = 0

    def compute_padding_mask(self, mask):
        """This function computes the padding mask.
        it takes in a mask and returns a padding mask.

        The padding mask serves to mask the padding tokens 
        in the input sequence. 1 corresponds to the padding
        token, and 0 corresponds to the non-padding token.

        params: mask: torch.Tensor: The mask tensor.

        return: torch.Tensor: The padding mask tensor. shape (B*H, N, N)
                    where H corresponds to the number of heads, N corresponds
                    to the sequence length.

        """
        pmask = mask.unsqueeze(1)
        pmask = pmask.expand(len(pmask), pmask.shape[-1], pmask.shape[-1])
        pmask = pmask.unsqueeze(1)
        pmask = pmask.expand(len(pmask), self.num_heads, pmask.shape[-1], pmask.shape[-1])
        pmask = pmask.reshape((-1, pmask.shape[-1], pmask.shape[-1]))
        return pmask.to(mask.device)


    def compute_causal_mask(self, mask):
        """This function computes the causal mask.
        it takes in a mask and returns a causal mask.

        The causal mask serves to mask the future tokens
        in the input sequence. 1 corresponds to the future
        token, and 0 corresponds to the non-future token.

        params: mask: torch.Tensor: The mask tensor.

        return: torch.Tensor: The causal mask tensor. shape (1, N, N)

        """

        attn_shape = (1, mask.shape[-1], mask.shape[-1])
        cmask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
        return cmask.to(mask.device)

    def set_beam_width(self, beam_width):
        self.beam_width = beam_width

    def _get_topk(self, loglikelihood):
        v = loglikelihood.shape[-1]
        loglikelihood, next_chars = loglikelihood.topk(
            k=min(v, self.beam_width), axis=-1
        )
        if v < self.beam_width:
            d = self.beam_width - self.len_vocab
            pl = -1e20 * torch.ones(
                (len(loglikelihood), d),
                dtype=loglikelihood.dtype,
                device=loglikelihood.device,
            )
            pc = torch.zeros(
                (len(next_chars), d),
                dtype=next_chars.dtype,
                device=loglikelihood.device,
            )
            loglikelihood = torch.cat((loglikelihood, pl), dim=-1)
            next_chars = torch.cat((next_chars, pc), dim=-1)
        return loglikelihood, next_chars

    def _init_action(self, loglikelihood):
        # Perform the first step
        loglikelihood, next_chars = self._get_topk(loglikelihood)

        self.loglikelihood = loglikelihood.view(-1, 1)
        next_chars = next_chars.view(-1, 1)

        self.y = (
            self.y.view(len(self.y), 1, -1).repeat(1, self.beam_width, 1).view(-1, 1)
        )
        self.x = (
            self.x[:, None]
            .repeat(1, self.beam_width, 1, 1)
            .view((-1,) + tuple(self.x.shape[1:]))
        )
        self.x_mask = (
            self.x_mask[:, None]
            .repeat(1, self.beam_width, 1, 1)
            .view((-1,) + tuple(self.x_mask.shape[1:]))
        )
        self.y = torch.cat((self.y, next_chars), dim=-1)

        # VERY IMPORTANT! we need a mask for
        # the log likelihood when reaching the eos
        # self.ll_mask = torch.zeros(len(self.loglikelihood), dtype=torch.bool)
        self.ll_mask = torch.any(self.y == self.eos_token, dim=-1)

    def get_actions(self):
        batch_size = self.batch_size
        # y_mask = self.subsequent_mask(self.y.shape[-1]).to(self.device)
        next_loglikelihood = []

        local_dataset = tud.TensorDataset(self.x, self.x_mask, self.y)
        local_loader = tud.DataLoader(local_dataset, batch_size=batch_size)

        # make sure that the local_loader
        # will be iterated over only once
        iterator = iter(local_loader)

        with torch.no_grad():
            for x, x_mask, y in local_loader:
                if x.device != self.device:
                    x = x.to(self.device)
                if x_mask.device != self.device:
                    x_mask = x_mask.to(self.device)
                if y.device != self.device:
                    y = y.to(self.device)

                y_mask = y < 0
                y = self.model.pe(self.model.token_emb(y)) # shape: (B, M, D)
                trg_m = self.compute_padding_mask(y_mask)
                tgt_mask = torch.logical_or(trg_m, self.compute_causal_mask(y_mask))
                out = self.model.transformer.decoder(y, memory=x, tgt_mask=tgt_mask)
                ll = torch.log_softmax(self.model.generator(out), dim=-1)[:, -1]
                    
                # ll = self.model.generator(out)[:, -1]
                next_loglikelihood.append(ll)
                
        next_loglikelihood = torch.cat(next_loglikelihood, axis=0)
        next_loglikelihood = next_loglikelihood.detach()
        if next_loglikelihood != self.data_device:
            next_loglikelihood = next_loglikelihood.to(self.data_device)
        return next_loglikelihood

    def action(self, next_loglikelhihood):
        if self.pos == 0:
            self._init_action(next_loglikelhihood)
        else:
            vocabulary_size = self.len_vocab
            # set loglikehihood to the maxium (0)
            # when observed an eos_token
            next_loglikelhihood[self.ll_mask, :] = (
                torch.minimum(self.loglikelihood.min(), next_loglikelhihood.min()) - 1.0
            )
            next_loglikelhihood[self.ll_mask, self.eos_token] = 0.0
            # done

            ll = (self.loglikelihood + next_loglikelhihood).view(
                -1, self.beam_width, vocabulary_size
            )
            ll, idx = self._get_topk(ll.flatten(start_dim=1))

            # tricky indexing
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            if best_candidates.device != self.device:
                best_candidates = best_candidates.to(self.device)
            # done

            y = self.y.view(-1, self.beam_width, self.y.shape[-1])
            i = torch.arange(len(y)).unsqueeze(-1).repeat(1, self.beam_width).flatten()
            j = best_candidates.flatten()
            self.y = y[i, j].view(-1, self.y.shape[-1])

            self.y = torch.cat((self.y, next_chars), dim=-1)
            self.loglikelihood = ll.view(-1, 1)

            # update ll mask
            self.ll_mask = torch.any(self.y == self.eos_token, dim=-1)
        self.pos = self.pos + 1

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        A = subsequent_mask == 0
        return A.type(torch.long)
    
    
 
class Criterion:
    def __call__(self, node):
        raise NotImplementedError("Not implemented")


class MaxLength(Criterion):
    """ It implements the maximum length criterion."""
    def __init__(self, max_length):
        super(MaxLength, self).__init__()
        self.max_length = max_length

    def __call__(self, node):
        return node.pos >= self.max_length


class EOS(Criterion):
    """ It implements the end of sequence criterion."""
    def __init__(self):
        super(EOS, self).__init__()

    def __call__(self, node):
        return torch.all(node.ll_mask).item()


class LogicalAnd(Criterion):
    """ It implements the logical AND criterion."""
    def __init__(self, criteria):
        super(LogicalAnd, self).__init__()
        self.criteria = criteria

    def __call__(self, node):
        return all([c(node) for c in self.criteria])


class LogicalOr(Criterion):
    """ It implements the logical OR criterion."""
    def __init__(self, criteria):
        super(LogicalOr, self).__init__()
        self.criteria = criteria

    def __call__(self, node):
        return any([c(node) for c in self.criteria])


def beamsearch(node, beamsize, stop_criterion):
    """ It performs beam search to generate molecules.
    It returns the node with the generated molecules.
    and negative log likelihood.
    
    params: node: Node: The node object.
    params: beamsize: int: The beam size.
    params: stop_criterion: Criterion: The stopping criterion.
    
    return: Node: The node object with the generated molecules and negative log likelihood.
        
    
    """
    
    node.set_beam_width(beamsize)

    while not stop_criterion(node):
        a = node.get_actions()
        node.action(a)

    a = node.get_actions()

    end_tokens = node.eos_token * torch.logical_not(node.ll_mask).type(
        node.y.dtype
    )

    node.y = torch.cat((node.y, end_tokens.view(-1, 1)), dim=-1)

    ll_tail = a[torch.arange(len(a)), end_tokens] * torch.logical_not(
        node.ll_mask
    ).type(a.dtype)
    node.loglikelihood = node.loglikelihood + ll_tail.view(-1, 1)
    return node