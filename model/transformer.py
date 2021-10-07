import torch.nn as nn
import torch

import sys

# import data_loader


# let's build blocks : multi-head attention(attention) / encoder / decoder / 
class Attention(nn.Module):
    '''
    init : None
    forward : QW, KW, VW, dk
    '''

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, QW, KW, VW, dk, mask=None):
        '''
        QW=KW=VW and it's one of the head : [bs, n, hs]
        return : [bs, m, hs/n_splits]
        '''
        numerator = torch.bmm(QW, KW.transpose(1,2))/dk**(.5) # |QW|,|KW| : [bs, m, hs], [bs, n, hs]

        if mask is not None: # mask : [bs, m, n]
            assert numerator.size() == mask.size()
            numerator.masked_fill_(mask, int('-inf'))

        numerator = self.softmax(numerator)                   # |numerator| : [bs, m, n]
        self_attn = torch.bmm(numerator, VW)                  # |self_attn| : [bs, m, hs/n_splits]
        return self_attn



class MultiHead(nn.Module):
    '''
        hidden_size : hidden size of one head
        n_splits : number of heads

        make (n_splits of Q,K,V) and multiply

        forward(Q,K,V,mask=None)
    '''

    def __init__(self, hidden_size, n_splits):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_splits = n_splits

        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias = False)

        self.Attn = Attention()

        self.linear = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, Q, K, V, mask=None):
        '''
        Q : [bs, m, hs] enc
        K : [bs, n, hs] dec
        V : [bs, n, hs] dec
        mask : [bs, m, n]
        '''

        # assert (self.hidden_size%self.n_splits) != 0
        QWs = self.Q_linear(Q).split(self.hidden_size//self.n_splits, dim = -1) # Q : [bs, m, hs] -> [bs, m, hs/n_splits]*n_splits
        KWs = self.K_linear(K).split(self.hidden_size//self.n_splits, dim = -1)
        VWs = self.V_linear(V).split(self.hidden_size//self.n_splits, dim = -1)

        QWs = torch.cat(QWs, dim=0) # QWs : [bs*n_splits, m, hs/n_splits]
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0) # |mask| : [bs, m, n] -> [bs*n_splits, m, n]
        
        attn = self.Attn(QWs, KWs, VWs, self.hidden_size//self.n_splits, mask=mask) # [bs*n_splits, m, hs/n_splits]
        assemble = attn.split(Q.size(0), dim = 0) # [bs, m, hs/n_splits] * n_splits
        assemble = torch.cat(assemble, dim = -1) # [bs,m,hs]
        result = self.linear(assemble)
        return result


class EncoderBlock(nn.Module):


    def __init__(self, hidden_size, n_splits, dropout_p=0.1, use_leaky_relu=False):
        super().__init__()

        self.attn_layernorm = nn.LayerNorm(hidden_size)
        self.multihead = MultiHead(hidden_size, n_splits)
        self.attn_dropout = nn.Dropout(dropout_p)
        # residual connection
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU,
            nn.Linear(hidden_size*4, hidden_size),
        )
        # residual
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        x = self.attn_layernorm(x)
        multihead = self.attn_dropout(self.multihead(
                                                    Q=x,
                                                    K=x,
                                                    V=x,
                                                    mask=mask))
        z = x + multihead

        result = z + self.fc_dropout(self.fc(self.fc_norm(z)))

        return result, mask



class DecoderBlock(nn.Module):
    '''
        x -> attn_layernorm -> masked multi head attention -> dropout -> residual
        -> attn_layernorm2 -> encoder decoder multi head attn -> dropout -> residual
        -> fc_norm -> fc -> dropout -> residual
        -> result
    '''
    def __init__(self, hidden_size, n_splits, dropout_p = 0.1, use_leaky_relu = False):
        super().__init__()

        self.masked_layernorm = nn.LayerNorm(hidden_size)
        self.masked_multihead = MultiHead(hidden_size, n_splits)
        self.masked_dropout = nn.Dropout(dropout_p)

        self.attn_layernorm = nn.LayerNorm(hidden_size)
        self.multihead = MultiHead(hidden_size, n_splits)
        self.attn_dropout = nn.Dropout(dropout_p)
        # residual
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size*4, hidden_size),
        )
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, key_value, prev_tensor, mask, future_mask):
        
        # prev_tensor exists only for inferencing
        if prev_tensor is None:
            
            z = self.masked_layernorm(x)
            z = z + self.masked_dropout(
                            self.masked_multihead(z, 
                                           z, 
                                           z, 
                                           mask = future_mask))
        # inferencing
        else:
            # here x work differently
            # |x|           =  |b, 1, hs|
            # |prev|        =  |b, t-1, hs|
            # |future_mask| = None

            normed_prev_tensor = self.masked_layernorm(prev_tensor)

            z = self.masked_layernorm(x) # |b, 1, 
            z = z+self.masked_dropout(
                            self.masked_multihead()
            )
            







if __name__ == '__main__':
    import sys
    sys.path.append('/home/rainism/Desktop/projects/my_nmt/')
    from data_loader import DataLoader
    
    # python data_loader.py ./data/corpus.shuf.test.tok.bpe ./data/corpus.shuf.test.tok.bpe en ko
    loader = DataLoader(
        '/home/rainism/Desktop/projects/my_nmt/data/corpus.shuf.test.tok.bpe',
        '/home/rainism/Desktop/projects/my_nmt/data/corpus.shuf.test.tok.bpe',
        ('en', 'ko'),
        device = -1
    )
    print(len(loader.src_field.vocab))
    print(len(loader.tgt_field.vocab))

    for batch_index, batch in enumerate(loader.train_iter):

        print(f'batch src')
        print(batch.src)
        print(f'printing shape of batch.src : {batch.src.shape}')
        print('-----------------------------------------------')
        print(f'batch tgt')
        print(f'printing shape of batch.tgt : {batch.tgt.shape}')
        print(batch.tgt)

        if batch_index > 0:
            break  
    
    input_size = len(loader.src_field.vocab)
    output_size = len(loader.tgt_field.vocab)
    input_ = batch.src
    output_ = batch.src
    
    hidden_size = 128
    embed = nn.Embedding(input_size, hidden_size)
    input_ = embed(input_)
    # print(f'shape of inpu_ {input_.shape}')

    # multi = MultiHead(hidden_size, 4)
    # multi_output = multi(input_, input_, input_, mask=None)
    # print(multi_output)
    encoderblock = EncoderBlock(hidden_size,4,dropout_p=0.1,use_leaky_relu=True)
    result = encoderblock(input_, mask = None)
    print(result[0].shape)
    