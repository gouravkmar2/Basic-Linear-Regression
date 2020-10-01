import numpy as np
import torch

x=torch.tensor(3.)
w=torch.tensor(4. , requires_grad=True)
b=torch.tensor(5., requires_grad=True)

y=w*x+b
y.backward()
print("dy/dw: ", w.grad)
print("dy=db: ",b.grad)

inputs = np.array([[73, 67, 43],[91,88,64],[87,134,58],[102,43,37],[69,96,70]],dtype='float32')
target=np.array([[56,70],[81,101],[119,133],[22,37],[103,119]],dtype='float32')
inputs = torch.from_numpy(inputs)
target = torch.from_numpy(target)
w=torch.randn(2,3,requires_grad=True)
b=torch.randn(2 , requires_grad=True)
def model(x):
    return x @ w.t() + b
preds = model(inputs)
print(preds)
def mse(t1,t2):
    diff = t1-t2
    return torch.sum(diff*diff)/diff.numel()
loss=mse(preds,target)
print(loss)
loss.backward()
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)
loss.backward()
print(b.grad)
with torch.no_grad():
    w -= w.grad* 1e-5
    b -= b.grad* 1e-5
    w.grad.zero_()
    b.grad.zero_()
preds=model(inputs)
loss= mse(preds,target)
print(loss)
for i in range(100):
    preds = model(inputs)
    loss = mse(preds,target)
    loss.backward()
    with torch.no_grad():
        w -= w.grad* 1e-5
        b -= b.grad* 1e-5
        w.grad.zero_()
        b.grad.zero_()
preds=model(inputs)
loss= mse(preds,target)
print(loss)
