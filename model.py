import torch
import torch.nn as nn

        
class GPT(nn.Module):
    def __init__(self, dim, vocab_size, max_seq_len, num_blocks, num_heads, device):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.token_embedding = torch.nn.Embedding(vocab_size, dim)
        self.positional_encoding = torch.nn.Embedding(max_seq_len, dim)

        self.device = device
        self.register_buffer('positions', torch.arange(max_seq_len))

        self.decoder_blocks = nn.ModuleList([DecoderBlock(dim, num_heads) for _ in range(num_blocks)])
        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        # x: B x T x C 
        position_emb = self.positional_encoding(self.positions[:x.shape[1]]).expand(x.shape[0], -1, -1)
        x = x + position_emb
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x) # B x T x C 

        x = self.linear(x) # => B x T x vocab_size  
        x = nn.functional.log_softmax(x, dim=2) # => B x T x vocab_size  
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, dim):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x 


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(DecoderBlock, self).__init__()

        self.multi_head_attention = MultiHeadAttention(dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(dim)

        self.feed_forward = FeedForwardBlock(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.multi_head_attention(x) + x
        x = self.layer_norm1(x) 
        x = self.feed_forward(x) + x
        x = self.layer_norm2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(dim, dim)
        assert dim % num_heads == 0

        self.heads = nn.ModuleList([SelfAttention(dim, int(dim/num_heads)) for _ in range(num_heads)])

    def forward(self, x):
        # x: B x T x C 
        x = [self.heads[i](x) for i in range(self.num_heads)] # => num_heads x B x T x (C/num_heads) 
        x = torch.concat(x, dim=2) # => B x T x C 
        x = self.linear(x)
        return x



class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super(SelfAttention, self).__init__()
        self.d_out = torch.tensor(d_out)

        self.q_proj = nn.Linear(d_in, d_out) 
        self.k_proj = nn.Linear(d_in, d_out) 
        self.v_proj = nn.Linear(d_in, d_out) 
        
    def forward(self, x):
        # x: B x T x C 
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x) # B x T x d_out 

        qk = torch.bmm(q, k.transpose(2,1)) / torch.sqrt(self.d_out) # B x T x T 

        tril = torch.ones(qk.shape).tril()
        qk[tril == 0] = float("-inf")

        logits = nn.functional.softmax(qk, dim=2) # B x T x T 

        res = torch.bmm(logits, v) # B x T x d_out 
        return res


