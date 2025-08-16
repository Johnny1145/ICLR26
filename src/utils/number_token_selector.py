import torch
import torch.nn.functional as F

from torch._tensor import Tensor

from model.regress_lm.vocabs import DecoderVocab
from model.regress_lm.tokenizers import DecoderTokenizer, P10Tokenizer

class NumberTokenSelector:
    '''
    Select number tokens 
    '''
    def __init__(self, vocab: DecoderVocab[float], device): # nvocab):
        self.tokenizer = vocab.tokenizer
        self.vocab = vocab
        self.nvocab = torch.full((len(vocab),), float("nan"), device=device)

        hashed_num_tokens = set(self.tokenizer.get_num_tokens())

        for token, id in self.vocab.stoi.items():
            if token in hashed_num_tokens:
                self.nvocab[id] = self.tokenizer.token_to_number(token)

        # Extract indices and values of number tokens
        self.number_token_mask = ~torch.isnan(self.nvocab)
        self.number_token_indices = torch.nonzero(self.number_token_mask, as_tuple=False).squeeze()

        self.number_token_values = self.nvocab[self.number_token_indices]

    def select_number_tokens(self, logits: Tensor):
        # Create a mask to filter out non-digit tokens and labels
        logits = logits[:, :, self.number_token_mask]
        return logits, self.number_token_mask

def main():
    tokenizer = P10Tokenizer(num_digits=4, exponent_range=10)
    vocab = DecoderVocab(tokenizer)
    device = torch.device("cuda")
    number_token_selector = NumberTokenSelector(vocab, device)
    print(number_token_selector.number_token_indices)
    print(number_token_selector.number_token_values)

if __name__ == "__main__":
    main()
