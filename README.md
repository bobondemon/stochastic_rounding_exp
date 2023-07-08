
Try to reproduce experiments in Table 1 in paper "[Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568)"
According to the descriptions in paper, we:
- use SQNR observer (the class `utilities.observers.MySQNRObserver`), we modified C codes in [AIMET](https://github.com/quic/aimet) into python
- fuse batch normalization into convolution (as described in Section 5 Experimental setup)
- use symmetric 4-bit weight quantization (as described in Section 5 Experimental setup)
- use per-tensor as quantization scheme (NOT per-channel)
- quantized first layer as described in paper
- apply stochastic rounding described in the paper "[Deep Learning with Limited Numerical Precision](https://arxiv.org/abs/1502.02551)"

Results in paper:
|  Rounding scheme   | Acc(%)  |
|  ----  | ----  |
| Nearest  | 52.29 |
| Ceil  | 0.10 |
| Floor  | 0.10 |
| Stochastic  | 52.06±5.52 |
| Stochastic (best)  | 63.06 |

Our results:
|  Rounding scheme   | Acc(%)  |
|  ----  | ----  |
| Nearest  | 58.60 |
| Ceil  | 0.13 |
| Floor  | 0.13 |
| Stochastic test1 | 52.18 |
| Stochastic test2 | 54.98 |
| Stochastic test3 | 55.12 |
| Stochastic test4 | 54.73 |
| Stochastic test5 | 47.48 |

> We tried several times and found that stochastic rounding is NOT better than nearest rounding. It seems weired since stochastic rounding has about 50% chance to be better than nearest rounding as shown in paper.

## Main Package Version
```
hydra-core             1.2.0
pytorch-lightning      1.8.4.post0
torch                  1.10.1+cu102
torchaudio             0.10.1+cu102
torchmetrics           0.11.0
torchvision            0.11.2+cu102
```

## How to Run
Run experiments
`python main.py`

## More Results
Several settings can be tried:
- symmetric/asymmetric scheme
- with/without fusing BN into Conv
- quantized first layer or all layers
- per-channel or per-tensor
- rounding scheme, e.g. `round` for nearest rounding; `ceil`; `floor`; `stochastic` for stochastic rounding
- int8 or int4

See `doc/analyze_rounding.xlsx`
