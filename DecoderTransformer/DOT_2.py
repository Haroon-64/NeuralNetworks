#  decoder only transformer
#  produce output unconditionaly

import torch
import torch.nn as nn
import torch.nn.functional as F

path = 'input.txt'
#------------------------------
#   hyperparameters
block_size = 256 # time dimension of the sequence /context size
batch_size = 64 # number of sequences in parallel
max_iters = 5000
learning_rate =3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters =500
n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.2 # 20% dropout
#--------------------------------

data  = open(path,'r').read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

# mappping of characters to integers
s_i = {c:i for i,c in enumerate(chars)}
i_s = {i:c for i,c in enumerate(chars)}
def encode(x): return [s_i[c] for c in x]  # encode input string to integers
def decode(x): return ''.join([i_s[i] for i in x])# decode integers to input string


# train-val split
data = torch.tensor(encode(data), dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

# data loading
def create_batch(slpit):
    data = train_data if slpit=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimateLoss():
    out ={}
    model.eval()
    for split in ['train', 'test']:
        losses =torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = create_batch(split)
            logits,loss =model(X,Y)
            losses[k] =loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    

class Head(nn.Module):
    """single head of self attention"""
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # convention for non parameters
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        K = self.key(x)
        q = self.query(x)
        # attention score / affinity 
        wei =  q @ K.transpose(-2,-1) * (C**-0.5) # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # mask out future tokens
        wei = F.softmax(wei,dim=-1) # (B,T,T)
        wei = self.dropout(wei) # (B,T,T) # randomly prevent tokens from attending to each other 
        # weighted average of values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    
    
class multiHead(nn.Module):
    """multi head self attention in parallel"""
    def __init__(self,num_heads,head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)        
        
    def forward(self,x):
        out =torch.cat([head(x) for head in self.heads],dim=-1)
        out =self.proj(out)
        return out
    
    
class FeedForward(nn.Module):
    """linear projection + activation"""
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # grow resiudal connection over time
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout) # dropout for regularization around residual connection   # change random neurons to zero (train on smaller networks)
        )
        
    def forward(self,x):
        return self.net(x)
        
        
class Block(nn.Module):
    """transformer block"""
    def __init__(self, n_emd,n_head) -> None:
        super().__init__()
        head_size = n_embd//n_head
        self.sa_heads = multiHead(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self,x):
        x = x+self.sa_heads(self.ln1(x)) # residual connection with layer norm
        x = x+self.ffwd(self.ln2(x)) 
        return x
        
    
class BigramModel(nn.Module):
    def __init__(self): 
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_embd)   # add intermediate embedding for tokens
        self.position_emb_table = nn.Embedding(block_size, n_embd)  # encode position of token in sequence
        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_heads) for _ in range(n_layers)]) # stack 3 transformer blocks
        self.lnf = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # add linear layer to go to logits from token embedding
    
    def forward(self, idx,targets=None):
        
        B,T = idx.shape
        
        tok_emb = self.token_emb_table(idx)          # (batch, time ,channel)
        pos_emb = self.position_emb_table(torch.arange(T,device=device)) # (time, channel)
        x = tok_emb + pos_emb         # (batch, time ,channel)
        x = self.blocks(x)            # (batch, time ,channel)
        logits = self.lm_head(x)               # (batch, time ,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)             # cross entropy expects batch x channel x time
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits,loss
    
    def generate(self,idx,max_new):                 # idx -> (B,T) for current sequence
        for _ in range(max_new):
            idx_cond =idx[:,-block_size:]           # (B,T) # prevent overflow
            logits,loss =self(idx_cond) 
            logits = logits[:,-1,:]                 # (B,C)
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,1)   # (B,1)
            idx = torch.cat((idx,idx_next),dim=1)   # (B,T+1)
        return idx 
    
model = BigramModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate) 

for iter in range(max_iters):
    if iter % eval_iters==0:
        losses = estimateLoss()
        print(losses.items())
    xb,yb = create_batch('train')
    
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#%%
context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(context, max_new=5000)[0].tolist())) 
