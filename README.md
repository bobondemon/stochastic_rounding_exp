
Try to implement AdaRound

# Usage

Main Package Version
```
hydra-core             1.2.0
pytorch-lightning      1.8.4.post0
torch                  1.10.1+cu102
torchaudio             0.10.1+cu102
torchmetrics           0.11.0
torchvision            0.11.2+cu102
```

## How to Train


- Training from scratch
    ```
    python train.py ckpt_path=null resume_training=false
    ```
- Resume training from ckpt
    ```
    python train.py ckpt_path=path_of_ckpt/last.ckpt resume_training=true
    ```


## How to Test
```
python test.py ckpt_path=path_of_ckpt/last.ckpt
```

## Monitoring with tensorboard
```
tensorboard --logdir ./outputs/train/tblog/lightning_logs/
```

# Results
