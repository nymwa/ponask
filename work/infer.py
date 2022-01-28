from argparse import ArgumentParser
from ilonimi import Normalizer, Tokenizer, Splitter
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from ponask.vocab import Vocab
from ponask.bert import BERT
from ponask.batch import Batch
from tabulate import tabulate

def load_vocab(path):
    with open(path) as f:
        tokens = [x.strip() for x in f]
    vocab = Vocab(tokens)
    return vocab

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', default = 'bert.pt')
    parser.add_argument('--vocab', default = 'vocab.txt')
    parser.add_argument('--hidden-dim', type = int, default = 128)
    parser.add_argument('--nhead', type = int, default = 4)
    parser.add_argument('--feedforward-dim', type = int, default = 256)
    parser.add_argument('--dropout', type = float, default = 0.3)
    parser.add_argument('--attention-dropout', type = float, default = 0.2)
    parser.add_argument('--activation-dropout', type = float, default = 0.2)
    parser.add_argument('--num-layers', type = int, default = 12)
    parser.add_argument('--max-len', type = int, default = 64)
    return parser.parse_args()

def main():
    args = parse_args()
    vocab = load_vocab(args.vocab)
    normalizer = Normalizer()
    tokenizer = Tokenizer(
            convert_unk = True,
            convert_number = False,
            convert_proper = False,
            ignore_set = {'<msk>'})
    splitter = Splitter(sharp = False)

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
    model.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu'))
    model = model.cuda()
    model.eval()

    while x := input():
        x = x.strip()
        x = normalizer(x)
        x = tokenizer(x)
        x = splitter(x)
        x = x.split()
        x = [vocab.bos] + [vocab(w) for w in x] + [vocab.eos]
        mask_indices = [i for i, w in enumerate(x) if w == vocab.msk]
        x = torch.tensor(x)
        x = pad([x], padding_value = vocab.pad)
        x = Batch(x)
        x = x.cuda()
        with torch.no_grad():
            x = model(x)
        x = x.transpose(1, 0)[0]
        for i in mask_indices:
            prob = torch.softmax(x[i], dim = -1)
            values, indices = torch.topk(prob, 10)
            tab = [
                ['word'] + [vocab[index] for index in indices],
                ['prediction (%)'] + ['{:.2f}'.format(value * 100) for value in values]]
            tab = tabulate(tab, tablefmt = 'psql')
            print('index: {}'.format(i))
            print(tab)




if __name__ == '__main__':
    main()

