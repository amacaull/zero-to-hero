# makemore

Character-level language models built from scratch while following
Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series.
Trained on `names.txt`, 32 033 first names.

One directory per video, each containing the code written along with the
video, a rewrite done from memory and comprehension the next morning, and the exercises from
the video description.

| part | topic | status |
|---|---|---|
| [`part_1/`](part_1) | bigram and trigram models, counting vs gradient descent | done |
| [`part_2/`](part_2) | MLP with character embeddings (Bengio et al., 2003) | done |

## Part 1 — bigram and trigram models

| notebook | content |
|---|---|
| `with_video.ipynb` | bigram model by counting, then as a one-layer neural network |
| `from_scratch.ipynb` | the same model rewritten from memory, without the video |
| `E01.ipynb` | trigram model, by counting and by gradient descent |
| `E02.ipynb` | train/dev/test split, generalization of the bigram vs the trigram |
| `E03.ipynb` | tuning the smoothing constant on the dev split |
| `E04.ipynb` | replacing the one-hot product with direct row indexing |
| `E05.ipynb` | `F.cross_entropy` and the numerical stability it provides |

Average negative log-likelihood, in nats per character.

| model | smoothing | dev |
|---|---|---|
| uniform | — | 3.2958 |
| bigram | 1 | 2.4533 |
| trigram | 1 | 2.2365 |
| trigram | 0.127 (tuned) | 2.2222 |

Test loss of the final model: 2.2236, evaluated once.

The counting model and the neural network converge to the same distribution:
after training, `W.exp()` normalized row-wise reproduces the count table with
a mean absolute error of 5e-4, and both generate identical names from the
same seed.

### What the exercises show

The trigram gains 0.22 nats over the bigram on unseen words, so the extra
context is worth far more than any hyperparameter tuning, which adds 0.014.

Only the trigram overfits. With 19 683 parameters for 182 000 training
examples it reacts to smoothing: lowering it from 1 to 0.001 improves the
training loss by 0.033 and degrades the dev loss by 0.008. The bigram, with
729 parameters, shows no train/dev gap at all.

Multiplying a one-hot vector by `W` is an indexing operation in disguise.
Replacing it with `W[xs]` makes the forward pass 65 times faster and drops
the allocation from 665 MB to 25 MB per iteration, with identical gradients.

Computing the softmax and the log by hand overflows in float32 once logits
reach ~100, silently producing `inf` or `nan`. `F.cross_entropy` subtracts
the maximum logit first, which leaves the result unchanged mathematically
and defined numerically. It is also 1.45x faster over a full
forward-backward step, most of the gain coming from its fused backward.

## Part 2 — MLP with character embeddings

| notebook | content |
|---|---|
| `with_video.ipynb` | MLP following Bengio et al. (2003): embeddings, tanh hidden layer, minibatches, learning rate search |
| `from_scratch.ipynb` | the same model rewritten from memory, then tuned to beat the video's loss (exercises E01 and E02) |

Average negative log-likelihood, in nats per character.

| model | params | train | dev |
|---|---|---|---|
| trigram, counting (part 1) | 19 683 | — | 2.2222 |
| MLP, video result | ~11 900 | ~2.13 | ~2.17 |
| MLP, fixed init, context 3, embd 10, hidden 100 | 6 097 | 2.1106 | 2.1376 |
| MLP, final: context 6, embd 50, hidden 500 | 165 377 | 1.798 | 1.978 ± 0.002 |

The final dev loss is the mean and standard deviation over 4 seeds.
Test loss of the final model: 1.9695, evaluated once.

### What the exercises show

The default initialization starts at a loss of 22.5 instead of
ln 27 = 3.30: logits are too large and most tanh units saturate, so the
first thousands of steps only undo the init. Scaling W2 by 0.01 and W1 by
(5/3)/√fan_in fixes both. Model size cannot be judged before this is fixed.

Tracking the dev loss during training shows a plateau at lr = 0.1, then a
drop right after the decay to 0.01. The gain comes from the schedule, not
from training longer, and the stages below 0.001 change nothing.

Context helps up to 6 characters, embeddings up to 50 dimensions, the
hidden layer up to 500 units. Past these points the training loss keeps
falling while the dev loss stays flat or rises.

L2 regularization shows the expected U shape: 1e-3 underfits (dev 2.06),
1e-5 overfits (train 1.74, dev 1.99), 1e-4 sits in between.

The best batch size depends on the model. For a small MLP, batches of 64 and
128 gave the same result. For the final one, batch 64 overfits more than
32 at the same number of steps (dev 1.985 vs 1.975), since it sees twice as
many examples.

The spread between seeds is about 0.002, so differences below ~0.005 in the
tuning runs, such as hidden 500 vs 1000 or context 6 vs 7, are noise.

A shuffle applied to a copy of the word list but not used for the split sent
the dev loss to 2.41: the dev set then held names unlike those in the
training set. A train/dev gap can come from the data before the model.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch matplotlib
jupyter lab
```
