import math
import torch
from torch import nn
import torch.nn.functional as F




class RATransformer(nn.Module):
    def __init__(self, num_layers, interval, d_feature, d_model, h, context_len, rat_b=False, relation_aware=True, context_attn=True, start_idx=0, dropout=0.1):
        # num_layers: number of layer for encoder and decoder
        # interval: time interval for the input sequence, such as 30 days
        # d_fearture: number of feature, which shoud be 4 ( open, highest, lowest, close)
        # d_model: model dimension 
        # h: number of heads
        # context_len: sequnce length of the context window
        # start_idx: can be ignored first
        super(RATransformer, self).__init__()
        self.num_layers = num_layers
        d_model = d_model
        h, dropout = h, dropout
        
#         self.max_length = config.max_length

        self.emb_enc = FeatureEmbeddingLayer(d_feature, d_model, start_idx, dropout)
        self.emb_dec = FeatureEmbeddingLayer(d_feature, d_model, start_idx, dropout)
        self.encoder = Encoder(num_layers, d_model, h, context_len, dropout, context_attn, relation_aware)
        self.decoder = Decoder(num_layers, d_model, h, context_len, dropout, context_attn, relation_aware)
        self.decision = DecisionLayer(d_model, d_feature, rat_b)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    def forward(self, enc_input, dec_input, previous):
        # enc_input: batch_size, num_asset, time, n_feature   
        # dec_input: batch_size, num_asset, time, n_feature
        # previous: b, m, 1  t
        
        # normalised price by the close price of the last day 
        #enc_input = enc_input/enc_input[:,:,-1:,-1:]  # b, m, t, d_feature
        enc_input = self.emb_enc(enc_input)  # b, m, t, d_feature
        enc_output = self.encoder(enc_input)
        
        #dec_input = dec_input/dec_input[:,:,-1:,-1:]  # b, m, t, d_feature
        dec_input = self.emb_dec(dec_input)  # b, m, t, d_feature
#         dec_mask = get_mask(dec_input)   # b, m, t, t
        dec_output = self.decoder(dec_input, enc_output)
        out = self.decision(dec_output, previous) #  b, m, 1
        
        return out
        
        
    

    
class DecisionLayer(nn.Module):
    def __init__(self, d_model, n_feature, rat_b=False):
        super(DecisionLayer, self).__init__()
        self.initial_portfolio = nn.Linear(d_model+1, 1)
        self.short_sale = nn.Linear(d_model+1, 1)
        self.reinvestment = nn.Linear(d_model+1, 1)
        self.money = torch.nn.Parameter(torch.zeros([1,1,1]))
        self.money2 = torch.nn.Parameter(torch.zeros([1,1,1]))
        self.money3 = torch.nn.Parameter(torch.zeros([1,1,1]))
        
        self.rat_b = rat_b

    def forward(self, x, previous):
        # x: b, m, 1, d_model
        # previous: b, m+1, 1
        b = x.size()[0]
        x = x[:,:,-1,:]
        previous = previous[:,1:,:]   # previous: b, m, 1  remove money
         # get the last day.  #x.squeeze(-2)
        x = torch.cat([x, previous], 2) # x: b, m, d_model+1
        
        
        money = self.money.repeat(b,1,1)    #b,1,1
        a = self.initial_portfolio(x) #  b, m, 1
        a = torch.cat([money,a],1)  #  b, m+1, 1
        a = F.softmax(a, 1) #  b, m+1, 1
        
        if self.rat_b:
            return a
        
        money2 = self.money2.repeat(b,1,1) #b,1,1
        a_s = self.short_sale(x) #  b, m, 1
        a_s = torch.cat([money2,a_s],1)  #  b, m+1, 1
        a_s = F.softmax(a_s, 1) #  b, m+1, 1
        
        money3 = self.money3.repeat(b,1,1) #b,1,1
        a_r = self.reinvestment(x) #  b, m, 1
        a_r = torch.cat([money3,a_r],1)  #  b, m+1, 1
        a_r = F.softmax(a_r, 1) #  b, m+1, 1
        
        #  b, m+1, 1
        return a-a_s+a_r
        
        
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, h, context_len, dropout, context_attn, relation_aware):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, context_len, dropout, context_attn, relation_aware) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
        
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, h, context_len,  dropout, context_attn, relation_aware):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, context_len, dropout, context_attn, relation_aware) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.norm(x)


    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, context_len, dropout, context_attn, relation_aware):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, context_len, dropout, context_attn)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer =  nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.relation_aware= relation_aware
        
    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, relat_attn=self.relation_aware))
        return self.sublayer[1](x, self.pw_ffn)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, context_len, dropout, context_attn, relation_aware):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, context_len, dropout, context_attn)
        self.src_attn = MultiHeadAttention(d_model, h, context_len, dropout, context_attn)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])
        self.relation_aware= relation_aware

    def forward(self, x, m, tgt_mask):
#         print('dec0', x.shape, m.shape)
        x = self.sublayer[0](x, lambda i: self.self_attn(i, i, i, tgt_mask, relat_attn=self.relation_aware, padding=True)) 
#         print('dec1', x.shape)
        #x = x[:,:,-1:,:]
        x = self.sublayer[1](x, lambda i: self.src_attn(i, m, m, relat_attn=False))
#         print('dec2', x.shape)
        return self.sublayer[2](x, self.pw_ffn)

   


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, context_len, dropout, context_attn):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.context_len = context_len
        self.W_v = nn.Linear(d_model, d_model)
        self.W_q = nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True)#nn.Linear(d_model, d_model)
        self.W_k = nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True) #nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.context_attn = context_attn
        
    def forward(self, query, key, value, mask=None, relat_attn=True, padding=True):
        # b, m, t, d_model
        b, m, t_q, d_model = query.size()
        b, m, t_v, d_model = value.size()
#         print('query', query.shape)
        
        query = self.W_q(query.permute((0,3,1,2))).permute((0,2,3,1)) #b, m, t, d_model
        key = self.W_k(key.permute((0,3,1,2))).permute((0,2,3,1) )    #b, m, t, d_model
        value = self.W_v(value) #b, m, t, d_model
#         print('query', query.shape, 'key', key.shape)
        if self.context_attn:
            context_query = ContextAttention(query, self.context_len, padding).contiguous().view(b, m, -1, self.h, self.d_k).transpose(2, 3) #b, m, t, d_model
            context_key = ContextAttention(key, self.context_len).contiguous().view(b, m, t_v, self.h, self.d_k).transpose(2, 3) #b, m, t, d_model
        else:
            context_query = query.contiguous().view(b, m, -1, self.h, self.d_k).transpose(2, 3) #b, m, t, d_model
            context_key = key.contiguous().view(b, m, t_v, self.h, self.d_k).transpose(2, 3) #b, m, t, d_model
            
        value = value.contiguous().view(b, m, t_v, self.h, self.d_k).transpose(2, 3) #b, m, h, t, d_k
        
#         print('context_query', context_query.shape, 'context_key', context_key.shape, 'value', value.shape)
        # scale-dot attention
        x, attn_weight = scaled_attention(context_query, context_key, value, mask)  #b, m, h, t_v,  d_k
        #x = x.transpose(1, 2).contiguous()().contiguous().view(b, m, -1, self.h * self.d_k)   
        x = x.transpose(2, 3)   #b, m, t_v, h,  d_k
        if relat_attn:
            x, asset_weight = RelationAttentionLayer(x)
        x = x.transpose(1, 0).contiguous().view(b, m, -1, self.h * self.d_k)
        
        x = self.fc(x)
        return x

    
    
def ContextAttention(x, context_length, padding=True):
    # x: b, m, t, d_model
    
    b, m, t, d_model = x.size()
    if padding:
        padding_x = torch.zeros((b, m, context_length-1, d_model)).to(x.device)
        x = torch.cat([padding_x, x], 2)
    
    attn_weight = torch.matmul(x[:,:,context_length-1:,:], x.transpose(-2, -1))/ math.sqrt(d_model)   #  x: b, m, t, t
    attn_weight_list = [F.softmax(attn_weight[:,:,i:i+1,i:i+context_length], dim = -1).permute((0, 1, 3, 2)) for i in range(attn_weight.size(-2))]  
    # (b, m, l, 1) * (t)    
    
    x_list = [x[:,:, i:i+context_length, :] for i in range(attn_weight.size(-2))] # (b, m, l, d_model) * (t)
    weighted_x_list = [attn_weight_list[i]*x_list[i] for i in range(attn_weight.size(-2))]  # (b, m, l, d_model) * (t)
    weighted_x_list = [torch.sum(i, 2, keepdim=True) for i in weighted_x_list ]  # (b, m, 1, d_model) * (t)
    weight_x = torch.cat(weighted_x_list, 2)   # b, m, t, d_model
    
    return weight_x

    
    
def RelationAttentionLayer(x,  mask=None):
        # x: b, m, t, h, d_f
        #b, m, t, h, d_f = x.size()
        #x = x.contiguous().view(b, m, t, self.h, self.d_f)
        x = x.permute((0, 2, 3, 1, 4 ))  # x: b, t, h, m, d_f
        x, asset_weights = scaled_attention(x, x, x, mask) # x: b, t, h, m, d_f
        x = x.permute((3, 0, 1, 2, 4 ))  # x: b, m, t, h, d_f
        return x, asset_weights


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)




class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        
    def forward(self, x):
        return self.mlp(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, start_idx, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.start_idx=start_idx
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: b*m, t, d_model
        x = x + self.pe[:, self.start_idx:self.start_idx+x.size(1)]
        return self.dropout(x)


    
class FeatureEmbeddingLayer(nn.Module):
    def __init__(self, d_feature, d_model, start_idx, dropout):
        super(FeatureEmbeddingLayer, self).__init__()
        self.feature_encode = nn.Linear(d_feature, d_model)
        self.pos_encode = PositionalEncoding(d_model, start_idx, dropout)
        self.d_model = d_model
        
    def forward(self, x):
        # x: b, m, t, d_feature
        # print(x.size())
        b, m, t, d_feature = x.size()
         
        x = x.contiguous().view(b*m, t, d_feature)
        x = self.feature_encode(x)
        x = self.pos_encode(x)
        x = x.contiguous().view(b, m, t, self.d_model)
        return x
    

def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m


# def clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def scaled_attention(query, key, value, mask=None):
    # mask: 0 for masked elements 
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask==0, float('-inf'))
    attn_weight = F.softmax(scores, -1)
    attn_feature = attn_weight.matmul(value)

    return attn_feature, attn_weight


# def get_mask(x):
#     b, m, t, d = x.size()
#     x_mask = (torch.ones(b,1,1)==1)            
#     x_mask = x_mask & ((torch.triu(torch.ones((1, t, t)), diagonal=1)==0).type_as(x_mask.data))   
#     return x_mask.to(x.device)  