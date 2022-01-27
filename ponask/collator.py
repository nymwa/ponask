import torch
from torch.nn.utils.rnn import pad_sequence as pad
from ponask.batch import Batch

class Collator:

    def __init__(
            self,
            vocab,
            mask_th = 0.15,
            replace_th = 0.03):

        self.vocab = vocab
        self.mask_th = mask_th
        self.replace_th = replace_th

    def __call__(self, batch):
        ei = [torch.tensor(sent) for sent in batch]
        eo = [torch.tensor(sent) for sent in batch]
        el = [len(sent) for sent in batch]

        ei = pad(ei, padding_value = self.vocab.pad)
        eo = pad(eo, padding_value = self.vocab.pad)

        rand_tensor = torch.rand(ei.shape)
        rand_token = torch.randint(self.vocab.msk, len(self.vocab), ei.shape)
        normal_token = ei > self.vocab.msk
        position_to_mask = (rand_tensor < self.mask_th) & normal_token
        position_to_replace = (rand_tensor < self.replace_th) & normal_token

        ei.masked_fill_(position_to_mask, self.vocab.msk)
        ei.masked_scatter_(position_to_replace, rand_token)
        eo.masked_fill_(~position_to_mask, self.vocab.pad)
        epm = (ei == self.vocab.pad).T

        return Batch(ei, eo, el, epm)

