import torch
import torch.nn.functional as F


words = open("names.txt",'r').read().splitlines()

chars = sorted(list(set((''.join(words)))))
SI = {c:i+1 for i,c in enumerate(chars)} # char to index
SI['<.>'] = 0 

IS = {i:c for c,i in SI.items()} # index to char


# create a dataset
xs, ys = [], [] 
for w in words:
    ch = ['<.>'] + list(w) + ['<.>']  
    for c1,c2 in zip(ch,ch[1:]):  
        i1 = SI[c1]
        i2 = SI[c2]
        xs.append(i1)
        ys.append(i2)
        
xs = torch.tensor(xs)
ys = torch.tensor(ys)  
num = xs.nelement() # number of elements
print(f"number of elements: {num}")

# interpret the output as log of counts : exp(log_counts)
g = torch.Generator().manual_seed(2147483647)  # set a generator to sample from the distribution
W  = torch.randn((27,27),requires_grad=True,generator=g)  



# gradient descent
for i in range(100):
    eX = F.one_hot(xs, num_classes = 27).float()  #input to the network
    
    # forward pass
    logits = eX @ W # matrix multiplication (log counts)
    counts = torch.exp(logits) # equivalent to N

    # softmax to get probabilities
    probs = counts / counts.sum(dim=1, keepdims=True) # equivalent to P
    loss = -(probs[torch.arange(num), ys]).log().mean()  
    print(loss)
    
    #backward pass
    W.grad = None
    loss.backward(retain_graph=True)
    
    #update 
    W.data -= 50 * W.grad





# sampling
g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    output = []
    index = 0 # init
    while True:
                 
        index = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item() 
        output.append(IS[index])
        if index == 0: # if <.> then break
            break
    print(''.join(output))