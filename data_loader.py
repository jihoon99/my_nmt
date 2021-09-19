import os
import torchtext
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data, datasets
else:
    from torchtext.legacy import data, datasets

class DataLoader():
    '''
    train_fn : train-path without extentions
    valid_fn : valid-path without extentions
    exts : extentions [en, ko]
    batch_size : 64
    device : cpu
    max_vocab = 9999999
    max_length = 255 'b/c max length of train dataset is 160'
    fix_length = size of length    e.g) [batch,length]
    use_bos : <bos> token
    use_eos : <eos> token
    shuffle : true
    '''
    def __init__(self,
                train_fn = None, 
                valid_fn = None,
                exts = None,
                batch_size = 64,
                device = 'cpu',
                max_vocab = 9999999999,
                max_length = 255,
                fix_length = None,
                use_bos = True,
                use_eos = True,
                shuffle = True,
                ):

        super(DataLoader, self).__init__() # ??? 상속받을게 없는데?

        # process : field -> translationDataset -> bucketIterator, and build vocab
        # 1. field
        self.src_field = data.Field(sequential=True,
                                use_vocab=True,
                                batch_first = True,
                                fix_length = fix_length,
                                init_token = '<BOS>' if use_bos else None,
                                eos_token = '<EOS>' if use_eos else None)
        
        self.tgt_field = data.Field(sequential=True,
                                use_vocab = True,
                                batch_first = True,
                                fix_length = fix_length,
                                init_token = '<BOS>' if use_bos else None,
                                eos_token = '<EOS>' if use_eos else None)

        # 2. TrainslationDataset
        if train_fn is not None and valid_fn is not None and exts is not None:

            self.train = TranslationDataset(path = train_fn,
                                            fields = [('src',self.src_field),('tgt',self.tgt_field)],
                                            exts = exts,
                                            max_length = max_length,
                                            )
            self.valid = TranslationDataset(path = valid_fn,
                                            fields = [('src',self.src_field),('tgt',self.tgt_field)],
                                            exts = exts,
                                            max_length = max_length)
            # next(iter(train.src))
            # ['▁▁My', '▁▁love', '▁▁for', '▁▁him', '▁▁does', '▁▁not', '▁▁change', '▁.']

            # 3. BucketIterator
            self.train_iter = data.BucketIterator(self.train,
                                                batch_size = batch_size,
                                                sort_key = lambda x: len(x.src)*max_length + len(x.tgt),
                                                shuffle = shuffle,
                                                device = 'cuda:%d' % device if device >= 0 else 'cpu',
                                                sort_within_batch=True)

            self.valid_iter = data.BucketIterator(self.valid,
                                                batch_size = batch_size,
                                                sort_key = lambda x: len(x.src)*max_length + len(x.tgt),
                                                shuffle = shuffle,
                                                device = 'cuda:%d' % device if device >= 0 else 'cpu',
                                                sort_within_batch = True)
            
            # 4. build vocab
            self.src_field.build_vocab(self.train, max_size = max_vocab)
            self.tgt_field.build_vocab(self.train, max_size = max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src_field.vocab = src_vocab
        self.tgt_field.vocab = tgt_vocab    



class TranslationDataset(data.Dataset):
    '''
    this class is modified from : data.TabularDataset.splits
        # path, format, field

    which params are : [path, exts, fields, max_length]
        exts : [en, ko]
        fields : [src_field, tgt_field]
    '''

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        if not path.endswith('.'):
            path += '.'

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip() # 오른쪽끝 스페이스 제거.
                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())): 
                    continue
                if src_line != '' and trg_line != '':
                    # 한쌍으로 만들기.
                    examples += [data.Example.fromlist([src_line, trg_line], fields)]
        super().__init__(examples, fields, **kwargs)


if __name__ == '__main__':
    # see if it works
    import sys
    # python data_loader.py ./data/corpus.shuf.test.tok.bpe ./data/corpus.shuf.test.tok.bpe en ko
    loader = DataLoader(
        sys.argv[1],
        sys.argv[2],
        (sys.argv[3], sys.argv[4]),
        device = -1
    )

    print(len(loader.src_field.vocab))
    print(len(loader.tgt_field.vocab))

    for batch_index, batch in enumerate(loader.train_iter):
        print(batch.src)
        print(batch.tgt)

        if batch_index > 1:
            break