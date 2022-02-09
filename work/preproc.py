from collections import Counter
from contextlib import ExitStack
from ilonimi import Normalizer, Tokenizer, Splitter
from seriejo import SeriejoWriter
from pathlib import Path
import random as rd
from ponask.vocab import Vocab
from ponask.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

class Preproc:

    def __init__(self):
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer(
                convert_unk = True,
                convert_number = False,
                convert_proper = False)
        self.splitter = Splitter(sharp = False)

    def __call__(self, sent):
        sent = sent.strip()
        sent = self.normalizer(sent)
        sent = self.tokenizer(sent)
        sent = self.splitter(sent)
        return sent


def sents_to_data(vocab, sents):
    def sent_to_data(sent):
        sent = sent.split()
        sent = [vocab(token) for token in sent]
        return sent
    return [sent_to_data(sent) for sent in sents]


def make_seriejo(base, name, data):
    Path(base).mkdir(parents = True, exist_ok = True)
    with SeriejoWriter('{}/{}'.format(base, name)) as f:
        for x in data:
            f.write(x)
    logger.info('Write Seriejo ({}/{}): {}'.format(base, name, len(data)))


def get_sents():
    preproc = Preproc()

    path_list = [
            '../../tokipona-corpus-collection/100tokipona/100tokipona.txt',
            '../../tokipona-corpus-collection/tokipona1000/tokipona1000.txt',
            '../../tokipona-corpus-collection/tatoeba/tatoeba.txt',
            '../../tokipona-corpus-collection/pu/pu.txt',
            '../../tokipona-corpus-collection/matthew/dave.txt',
            '../../tokipona-corpus-collection/matthew/mika.txt',
            '../../tokipona-corpus-collection/matthew/ote.txt',
            '../../tokipona-corpus-collection/matthew/prince.txt']

    with ExitStack() as stack:
        sents = [
            preproc(sent)
            for path
            in path_list
            for sent
            in stack.enter_context(open(path))]

    sents = [
        sent
        for sent
        in sents
        if len(sent.split()) <= 100]
    return sents


def make_tokens(train_sents):
    freq = Counter([
        word
        for sent
        in train_sents
        for word
        in sent.split()
        ]).most_common()
    tokens = [w for w, f in freq if w != '<unk>']
    tokens = ['<pad>', '<bos>', '<eos>', '<msk>', '<unk>'] + tokens
    logger.info('Make Tokens -> vocab size: {}'.format(len(tokens)))
    return tokens


def split_sents(sents, valid_size, test_size):
    train_sents = sents[: -(valid_size + test_size)]
    valid_sents = sents[-(valid_size + test_size) : -test_size]
    test_sents = sents[-test_size :]
    logger.info('split -> train: {}'.format(len(train_sents)))
    logger.info('split -> valid: {}'.format(len(valid_sents)))
    logger.info('split -> test : {}'.format(len(test_sents)))
    return train_sents, valid_sents, test_sents


def main():
    sents = get_sents()

    rd.seed(200)
    rd.shuffle(sents)

    train_sents, valid_sents, test_sents = split_sents(
            sents,
            valid_size = 1000,
            test_size = 1000)

    tokens = make_tokens(train_sents)
    with open('vocab.txt', 'w') as f:
        for x in tokens:
            print(x, file = f)

    vocab = Vocab(tokens)

    train_data = sents_to_data(vocab, train_sents)
    valid_data = sents_to_data(vocab, valid_sents)
    test_data = sents_to_data(vocab, test_sents)
    make_seriejo('data', 'train', train_data)
    make_seriejo('data', 'valid', valid_data)
    make_seriejo('data', 'test', test_data)


if __name__ == '__main__':
    main()

