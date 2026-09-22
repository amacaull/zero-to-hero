# micrograd

A tiny scalar autograd engine and neural network library in pure Python, rebuilt from scratch while following Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) lecture.

About 150 lines of code, no dependencies except PyTorch for testing.

## Files

```
engine.py              # Value: scalar autograd (forward + backward)
nn.py                  # Neuron, Layer, MLP
test/test_engine.py    # every operation and gradient checked against PyTorch
test/test_nn.py        # MLP structure, zero_grad, gradients vs PyTorch, training
```

## How it works

- `Value` wraps a single number and remembers which operation produced it, building a computation graph as you compute.
- `backward()` sorts the graph topologically and applies the chain rule node by node, from the output back to the inputs.
- Gradients are accumulated with `+=`, so a value used several times receives the sum of its contributions.
- Supported: `+ - * / **`, `exp`, `tanh`, `relu`.

## Example

```python
from engine import Value

a = Value(2.0)
b = Value(-3.0)
c = (a * b + a).tanh()
c.backward()

print(a.grad, b.grad)   # dc/da, dc/db
```

```python
from nn import MLP

model = MLP(2, [16, 16, 1], act='tanh')   # 2 inputs, two hidden layers, 1 linear output
out = model([0.5, -1.0])
```

## Tests

Every forward result and gradient is compared with PyTorch: first on an expression combining all operations with reused variables, then on a full MLP (tanh and ReLU).

```bash
pip install torch pytest
pytest
```
