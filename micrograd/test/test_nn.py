import random
import pytest
import torch
from nn import MLP

TOL = 1e-6


def test_structure():
    model = MLP(2, [16, 16, 1], act='tanh')
    # (2+1)*16 + (16+1)*16 + (16+1)*1
    assert len(model.parameters()) == 337
    # hidden layers use the activation, the last one stays linear
    assert all(n.act == 'tanh' for n in model.layers[0].neurons)
    assert model.layers[-1].neurons[0].act is None
    # a single output is returned as a Value, not a list
    assert not isinstance(model([1.0, 2.0]), list)


def test_zero_grad():
    model = MLP(3, [4, 1])
    model([1.0, -2.0, 0.5]).backward()
    assert any(p.grad != 0 for p in model.parameters())
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())


@pytest.mark.parametrize("act, torch_act", [("tanh", torch.tanh), ("relu", torch.relu)])
def test_mlp_matches_torch(act, torch_act):
    random.seed(0)
    model = MLP(3, [4, 1], act=act)
    x = [0.5, -1.0, 2.0]
    out = model(x)
    out.backward()

    # same weights, copied into PyTorch tensors
    def weights(layer):
        W = torch.tensor([[w.data for w in n.w] for n in layer.neurons], dtype=torch.double, requires_grad=True)
        b = torch.tensor([n.b.data for n in layer.neurons], dtype=torch.double, requires_grad=True)
        return W, b

    l1, l2 = model.layers
    W1, b1 = weights(l1)
    W2, b2 = weights(l2)
    h = torch_act(W1 @ torch.tensor(x, dtype=torch.double) + b1)
    out_t = (W2 @ h + b2).sum()
    out_t.backward()

    assert abs(out.data - out_t.item()) < TOL
    for layer, W, b in [(l1, W1, b1), (l2, W2, b2)]:
        for i, n in enumerate(layer.neurons):
            for j, w in enumerate(n.w):
                assert abs(w.grad - W.grad[i, j].item()) < TOL
            assert abs(n.b.grad - b.grad[i].item()) < TOL


def test_training_reduces_loss():
    random.seed(0)
    model = MLP(3, [4, 4, 1], act='tanh')
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    def loss_fn():
        return sum((model(x) - y) ** 2 for x, y in zip(xs, ys))

    first = loss_fn().data
    for _ in range(20):
        loss = loss_fn()
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad

    assert loss_fn().data < first
