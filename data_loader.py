import os
import torchtext
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data, datasets
else:
    from torchtext.legacy import data, datasets

# class DataLoader():
#     '''
#     train_fn : train-path without extentions
#     valid_fn : valid-path without extentions
#     exts : extentions [en, ko]
#     batch_size : 64
#     device : cpu
#     max_vocab = 9999999
#     max_length = 255 'b/c max length of train dataset is 160'
#     fix_length = size of length    e.g) [batch,length]
#     use_bos : <bos> token
#     use_eos : <eos> token
#     shuffle : true
#     '''
#     def __init__(self,
#                 train_fn = None, 
#                 valid_fn = None,
#                 exts = None,
#                 batch_size = 64,
#                 device = 'cpu',
#                 max_vocab = 9999999999,
#                 max_length = 255,
#                 fix_length = None,
#                 use_bos = True,
#                 use_eos = True,
#                 shuffle = True,
#                 ):

#         super(DataLoader, self).__init__() # ??? 상속받을게 없는데?

#         # process : 

class TranslationDataset(data.Dataset):
    '''
    this class is modified from : data.TabularDataset.splits
    which contains : [path, train, validation, format, fields]

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


# if __name__ == "__main__":
#     src = data.Field( sequential=True,
#                                 use_vocab = True,
#                                 batch_first = True,
#                                 include_lengths = True,
#                                 fix_length = 100,)
#     tgt = data.Field( sequential=True,
#                                     use_vocab = True,
#                                     batch_first = True,
#                                     include_lengths = True,
#                                     fix_length = 100,)
#     print(os.path.curdir)
# # my_nmt/data/corpus.shuf.test.tok.bpe.en
#     a = TranslationDataset("./data/corpus.shuf.test.tok.bpe.",
#                             ['en','ko'],
#                             [('src',src),('tgt',tgt)])
