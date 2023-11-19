import torch
import torch.nn.functional as F
import random
path = 'names.txt'
words = open(path,'r').read().split('\n')

chars = sorted(list(set(''.join(words))))
S_I = {c:i+1 for i,c in enumerate(chars)}
S_I['.'] = 0
I_S = {i:c for c,i in S_I.items()}

block_size = 3  # context size 
vocab_size = len(S_I)
def build_dataset(words):
    # build the dataset


    X,Y = [],[] # input, labels
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = S_I[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.Tensor(X).int()
    Y = torch.Tensor(Y).int()
    print(X.shape,Y.shape)
    return X,Y

# split into train val and test
random.seed(42)
random.shuffle(words)

n1 = int(len(words)*0.8)
n2 = int(len(words)*0.9)

X_train,Y_train = build_dataset(words[:n1])  #train network
X_val,Y_val = build_dataset(words[n1:n2])    # tune hyperparameters
X_test,Y_test = build_dataset(words[n2:])    # final evaluation


# MLP initialization
n_embed = 10 # embedding size / size of lookup table
n_hidden = 200 # hidden layer size
GAIN =5/3 # gain for tanh
INIT_SCALE = GAIN/(n_embed * block_size)**.5 

g = torch.Generator().manual_seed(2147483647)

C = torch.randn((vocab_size,n_embed),generator=g) # lookup table
W1 = torch.randn(( n_embed * block_size ,n_hidden ),generator=g)* INIT_SCALE # weights for the hidden layer
B1 = torch.randn((n_hidden,),generator=g) *.001
W2 = torch.randn((n_hidden,vocab_size),generator=g)*.1  # weights for the output layer
B2 = torch.randn((vocab_size),generator=g) *0

params = [C,W1,B1,W2,B2]
for p in params:
    p.requires_grad_(True)
    
    
    
BATCH_SIZE = 64
MAX_STEPS = 90000
lossi=[]

for i in range(MAX_STEPS):
    # create minibatch
    ix = torch.randint(0,len(X_train),(BATCH_SIZE,)) # random indices
    Xb,Yb = X_train[ix],Y_train[ix]
    
    # forward pass
    emb = C[Xb] # embedding layer / input to tanh layer 
    embcat = emb.reshape((emb.shape[0],-1)) # flatten
    h_pre = embcat @ W1 + B1 # pre-activation
    h_pre = (h_pre - h_pre.mean(0,keepdim=True)) / h_pre.std(0,keepdim=True) 
    h = torch.tanh(h_pre ) # hidden layer
    logits = h @ W2 + B2
    loss = F.cross_entropy(logits,Yb.long())
    
    #backward pass
    for p in params:
        p.grad = None
    loss.backward(retain_graph=True)
   
    # update the parameters
    lr = 0.1 if i<30000 else 0.01
    for p in params:
        p.data -= p.grad * lr
        
    #track
    if i%5000==0:
        print(i,loss.item())
    lossi.append(loss.log10().item())
    
    
print(loss.item())  # loss for a single batch

@torch.no_grad() # don't track gradients
def split_loss(split):
    x,y={
        'train':(X_train,Y_train),
        'val':(X_val,Y_val),
        'test':(X_test,Y_test)
    }[split]
    emb = C[x]
    embcat = emb.view((emb.shape[0],-1))
    h = torch.tanh(embcat @ W1 + B1)
    logits = h @ W2 + B2
    loss = F.cross_entropy(logits,y.long())
    print(split,loss.item())
    
split_loss('train')
split_loss('val')


# sampling
G = torch.Generator().manual_seed(2147483647+1)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1,-1) @ W1 + B1)
        logits = h @ W2 + B2
        probs = F.softmax(logits,dim=1)
        ix = torch.multinomial(probs,num_samples=1,generator=G).item()
        context = context[1:] + [ix]
        out.append(I_S[ix])
        if I_S[ix] == '.':
            break 
    print(''.join(out))