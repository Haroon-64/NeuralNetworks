from Network import *
from visualize import *
import random


net = MLP(10, [10, 10, 1])                  # number of inputs, [number of neurons in each layer, number of outputs]
xs = [[x] for x in [random.randint(1, 100) for _ in range(10)]]        # each input needs to be a list of one element to be iterable
ys = [1 if x[0] % 2 == 0 else 0 for x in xs]




for k in range(30): 
    yp = [net(i) for i in xs]
    loss = sum([(out - ygt)**2 for ygt, out in zip(ys,yp)])
    
    for p in net.parameters():
        p.grad = 0.

    loss.backward()

    for p in net.parameters():
        p.data += -.025 * p.grad
    print(k, loss.data)


preds = [yp.data for yp in yp]

print(f"input: {xs} prediction: {preds},\n ground truth: {ys}")

draw_dot(loss, save_as_png=False)