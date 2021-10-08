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
            numerator.masked_fill_(mask, float('-inf'))

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
            z = x + self.masked_dropout(
                            self.masked_multihead(
                                z, 
                                z, 
                                z, 
                                mask = future_mask))
        # inferencing
        else:
            # here x works differently
            # |x|           =  |b, 1, hs|
            # |prev|        =  |b, t-1, hs|
            # |future_mask| = None

            normed_prev_tensor = self.masked_layernorm(prev_tensor)
            z = self.masked_layernorm(x) # |b, 1, hs|
            z = x + self.masked_dropout(
                            self.masked_multihead(
                                z, 
                                normed_prev_tensor,
                                normed_prev_tensor,
                                mask = None)) # mask is not need since it's AR inferencing

        normed_key_value = self.attn_layernorm(key_value)
        z_ = self.attn_layernorm(z)
        z = z + self.attn_dropout(self.multihead(
                                        z_,               # |b, m, hs
                                        normed_key_value, # |b, n, hs|
                                        normed_key_value,
                                        mask = mask))
        # z : |b, m, hs|, m = 1 when inferencing

        z = z + self.attn_dropout(self.fc(self.fc_norm(z)))

        return z, key_value, prev_tensor, mask, future_mask


class CustomSequential(nn.Sequential):
    # since nn.Sequential class does not provide multiple input arguments and returns.
    def forward(self, *x):
        for block in self._modules.values():
            x = block(*x)

        return x



class Transformer(nn.Module):

    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            n_splits,
            max_length,
            dropout_p,
            num_enc_layer,
            num_dec_layer,
            use_leaky_relu,

            ):

        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits
        self.dropout_p = dropout_p
        self.num_enc_layer = num_enc_layer

        self.embed_enc = nn.Embedding(input_size, hidden_size)
        self.embed_dec = nn.Embedding(output_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout_p)

        self.pos_enc = self._positional_encoding_(hidden_size,max_length)


        self.Encoder = CustomSequential(
                            *[EncoderBlock(hidden_size, n_splits, dropout_p, use_leaky_relu)
                            for _ in range(num_enc_layer)])

        self.Decoder = CustomSequential(
                            *[DecoderBlock(hidden_size, n_splits, dropout_p, use_leaky_relu)
                            for _ in range(num_dec_layer)]                    
                            )
        
        self.generator = nn.Sequential(
                                nn.LayerNorm(hidden_size),
                                nn.Linear(hidden_size, output_size),
                                nn.LogSoftMax(dim = -1)        
                                )

    @torch.no_grad()
    def _positional_encoding_(self, hidden_size, max_length):
        '''
        this is for training

        예) pos = 3(word) // dim_idx = 2 = 2*i
                    pos
            sin( ---------  )
                    10^4(2*i/d)

        returning : [max_length, hs]
        '''
        empty = torch.FloatTensor(max_length, hidden_size).zero_()
        
        pos = torch.arange(0, max_length).unsqueeze(-1).float()    # |max_length, 1|
        dim = torch.arange(0, hidden_size//2).unsqueeze(0).float() # |1, hidden_size//2|
        
        empty[:, 0::2] = torch.sin(pos/1e+4**dim.div(float(hidden_size)))
        empty[:, 1::2] = torch.cos(pos/1e+4**dim.div(float(hidden_size)))

        return empty
    

    # me
    @torch.no_grad()
    def _generate_mask_(self, length):
        mask = []

        max_length = max(length)

        for l in length:
            if max_length - l >0:
                # l is smaaller than max_length
                tmp_tensor = torch.zeros([max_length])
                tmp_tensor[l:] = 1
                mask += [tmp_tensor.unsqueeze(0).bool()]

            else:
                mask += [torch.zeros([max_length]).unsqueeze(0).bool()]

        mask = torch.cat(mask, dim = 0)
        return mask

    
    @torch.no_grad()
    def _positional_encoding_inf_(self, hidden_size, max_length):
        print(1)

    def forward(self, x, y):
        # embed for encoder
        x[0] = self.embed_enc(x[0])

        with torch.no_grad():
            mask = self._generate_mask_(x[1]) # |batch, n|
            mask_enc = mask.unsqueeze(1).expand(*x[0].size(), mask.size(-1)) # |batch, 1, n| expand to [|batch, n| * n(mask.size(-1))]
            mask_dec = mask.unsqueeze(1).expand(*y.size(), mask.size(-1)) # |bs, m, n|
        



if __name__ == '__main__':
    import sys
    sys.path.append('/home/rainism/Desktop/projects/my_nmt/')
    from data_loader import DataLoader
    
    # print(torch.arange(0, 100).shape)

    # python data_loader.py ./data/corpus.shuf.test.tok.bpe ./data/corpus.shuf.test.tok.bpe en ko
    loader = DataLoader(
        '/home/rainism/Desktop/projects/my_nmt/data/corpus.shuf.test.tok.bpe',
        '/home/rainism/Desktop/projects/my_nmt/data/corpus.shuf.test.tok.bpe',
        ('en', 'ko'),
        device = -1
    )


    # # print(len(loader.src_field.vocab))
    # # print(len(loader.tgt_field.vocab))

    # for batch_index, batch in enumerate(loader.train_iter):

    #     # print(f'batch src')
    #     # print(batch.src)
    #     # print(f'printing shape of batch.src : {batch.src.shape}')
    #     # print('-----------------------------------------------')
    #     # print(f'batch tgt')
    #     # print(f'printing shape of batch.tgt : {batch.tgt.shape}')
    #     # print(batch.tgt)

    #     if batch_index > 0:
    #         break  
    
    # input_size = len(loader.src_field.vocab)
    # output_size = len(loader.tgt_field.vocab)
    # x = batch.src
    # y = batch.tgt

    # # print(batch.src)
    # hidden_size = 128
    # embed = nn.Embedding(input_size, hidden_size)
    # input_ = embed(x[0])
    # # print(f'shape of inpu_ {input_.shape}')

    # embed2 = nn.Embedding(output_size, hidden_size)
    # output_ = embed2(y[0])

    # # # multi = MultiHead(hidden_size, 4)
    # # # multi_output = multi(input_, input_, input_, mask=None)
    # # # print(multi_output)
    # encoderblock = EncoderBlock(hidden_size,4,dropout_p=0.1,use_leaky_relu=True)
    # result = encoderblock(input_, mask = None)

    # decoderblock = DecoderBlock(hidden_size, 4, dropout_p=0.1, use_leaky_relu=False)


    # future_mask = torch.triu(x[0].new_ones((y[0].size(1), y[0].size(1))), diagonal=1).bool() # triangle upper
    # future_mask = future_mask.unsqueeze(0).expand(y[0].size(0), *future_mask.size())
    # result1 = decoderblock(output_, result[0], None, mask = None, future_mask = future_mask)

