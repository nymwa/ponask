from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from seriejo import Seriejo
from ponask.vocab import Vocab
from ponask.dataset import Dataset
from ponask.sampler import Sampler
from ponask.collator import Collator
from ponask.bert import BERT

from ponask.accumulator import Accumulator
from ponask.scheduler import WarmupScheduler

from ponask.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

def load_vocab(path):
    with open(path) as f:
        tokens = [x.strip() for x in f]
    vocab = Vocab(tokens)
    return vocab


def load_dataset():
    data = Seriejo('data/train')
    dataset = Dataset(data)
    return dataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--vocab', default = 'vocab.txt')
    parser.add_argument('--max-tokens', type = int, default = 10000)
    parser.add_argument('--mask-th', type = float, default = 0.15)
    parser.add_argument('--replace-th', type = float, default = 0.03)
    parser.add_argument('--shift-prob', type = float, default = 0.80)
    parser.add_argument('--max-shift', type = int, default = 128)
    parser.add_argument('--hidden-dim', type = int, default = 128)
    parser.add_argument('--nhead', type = int, default = 4)
    parser.add_argument('--feedforward-dim', type = int, default = 256)
    parser.add_argument('--dropout', type = float, default = 0.3)
    parser.add_argument('--attention-dropout', type = float, default = 0.2)
    parser.add_argument('--activation-dropout', type = float, default = 0.2)
    parser.add_argument('--num-layers', type = int, default = 12)
    parser.add_argument('--max-len', type = int, default = 256)
    parser.add_argument('--lr', type = float, default = 0.002)
    parser.add_argument('--weight-decay', type = float, default = 0.01)
    parser.add_argument('--clip-norm', type = float, default = 1.0)
    parser.add_argument('--warmup-steps', type = int, default = 8000)
    parser.add_argument('--epochs', type = int, default = 300)
    return parser.parse_args()


def main():
    args = parse_args()
    vocab = load_vocab(args.vocab)
    dataset = load_dataset()
    sampler = Sampler(dataset, args.max_tokens)
    collator = Collator(
            vocab,
            args.mask_th,
            args.replace_th,
            args.shift_prob,
            args.max_shift)
    loader = DataLoader(
            dataset,
            batch_sampler = sampler,
            collate_fn = collator)

    model = BERT(
            len(vocab),
            args.hidden_dim,
            args.nhead,
            args.feedforward_dim,
            args.dropout,
            args.attention_dropout,
            args.activation_dropout,
            args.num_layers,
            padding_idx = vocab.pad,
            max_len = args.max_len)
    model = model.cuda()
    logger.info('#params : {} ({})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    criterion = nn.CrossEntropyLoss(ignore_index = vocab.pad)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    scheduler = WarmupScheduler(optimizer, args.warmup_steps)
    clip_norm = args.clip_norm

    num_steps = 0
    for epoch in range(args.epochs):
        accum = Accumulator(epoch, len(loader))
        for step, batch in enumerate(loader):
            batch.cuda()
            pred = model(batch)
            pred = pred.view(-1, pred.size(-1))
            loss = criterion(pred, batch.outputs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if clip_norm > 0:
                grad = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            else:
                grad = None
            optimizer.step()
            scheduler.step()
            num_steps += 1
            lr = scheduler.get_last_lr()[0]
            accum.update(batch, loss, lr, grad)
            logger.info(accum.step_log())
        logger.info(accum.epoch_log(num_steps))
    torch.save(model.state_dict(), 'bert.pt')


if __name__ == '__main__':
    main()

