import torch
from engine import Value

def test_activations():

    x = Value(0.5)
    w = Value(-1.5)
    h = (x * w + 2).tanh()
    z = h * x.exp() + (x - w).relu() + (x + w).relu()
    out = z / (1 + h**2) + x * h
    out.backward()
    xmg, wmg, omg = x, w, out

    x = torch.tensor([0.5], dtype=torch.double, requires_grad=True)
    w = torch.tensor([-1.5], dtype=torch.double, requires_grad=True)
    h = (x * w + 2).tanh()
    z = h * x.exp() + (x - w).relu() + (x + w).relu()
    out = z / (1 + h**2) + x * h
    out.backward()

    tol = 1e-6
    # forward pass went well
    assert abs(omg.data - out.item()) < tol
    # backward pass went well
    assert abs(xmg.grad - x.grad.item()) < tol
    assert abs(wmg.grad - w.grad.item()) < tol
