import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import tiktoken


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        # GELU(approximate='tanh')  # gaussian Rectifier approx by tanh (original was slow in tf)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):  # transformer block
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(
            self.ln_1(x))  # residual connections according to diagram plus ln after attn i.e. ln in residuals
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:  # gpt 124M
    block_size = 1024  # max sequence length
    vocab_size = 50257  # 50k BPE + 256 utf-8 + 1 <|endoftext|>
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.2
    bias = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(  # submodule indexing
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # weight token embd                  nn.Embedding -> wrapper for array of nums
                wpe=nn.Embedding(config.block_size, config.n_embd),  # weight pos embd
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # hidden         # index using numeric as per statedict
                ln_f=nn.LayerNorm(config.n_embd),  # additional layernorm
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # classifier/head   n_embd -> vocab size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Block size {self.config.block_size} exceeds block size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape T
        pos_emb = self.transformer.wpe(pos)  # shape T,n_embd
        tok_emb = self.transformer.wte(idx)  # shape B,T,n_embd
        x = tok_emb + pos_emb  # broadcast

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # forward final layernorm
        logits = self.lm_head(x)  # B,T,vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(
                -1))  # cross entropy doesnt prefer multidimensional inputs   logits = B*T,vocab
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):  # load pretrained
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig()
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


if __name__ == '__main__':
    model = GPT.from_pretrained('gpt2')

    num_return_sequences = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_length = 30
    model.eval()
    model.to(device)

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello World")
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # idx
    x = tokens.to(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]  # take logits at end
            probs = F.softmax(logits, dim=-1)
            topx_probs, topx_tokens = torch.topk(probs, 50, dim=-1)  # huggingface default /keep top 50
            ix = torch.multinomial(topx_probs, 1)  # select from top
            xcol = torch.gather(topx_tokens, -1, ix)  # gather corresponding index
            x = torch.cat((x, xcol), dim=1)  # append

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()  # rows
        decoded = enc.decode(tokens)
        print(decoded)