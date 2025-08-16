import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from src.utils.number_token_selector import NumberTokenSelector
from src.model.regress_lm.vocabs import DecoderVocab

class NumberTokenLoss:
    def __init__(self, vocab: DecoderVocab[float], device, loss_function=F.mse_loss, weight=0.5):
        self.loss_function = loss_function
        self.weight = weight
        
        self.selector = NumberTokenSelector(vocab, device) # self.nvocab)
        self.nvocab = self.selector.nvocab # torch.full((vocab_size,), float("nan"), device=device) 


    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")
        # print(logits.shape, labels.shape)
        logits, number_token_mask = self.selector.select_number_tokens(logits)

        # Compute the weighted average of number tokens (yhat)
        softmaxed = F.softmax(logits, dim=-1)
        yhat = torch.sum(softmaxed * self.nvocab[number_token_mask], dim=-1)
        y = self.nvocab[labels]

        loss = self.loss_function(yhat[~torch.isnan(y)], y[~torch.isnan(y)])
        return loss