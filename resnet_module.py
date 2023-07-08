# https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py

import pytorch_lightning as pl
import torch.nn as nn
import torchvision.models as models
from torch.optim import SGD, Adam
from torchmetrics import Accuracy

# from adaround.dataloader.imagenet_dataset_and_loader import ImageNetKaggle, ImageNetValTestDataLoader


# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    optimizers = {"adam": Adam, "sgd": SGD}

    def __init__(
        self,
        num_classes,
        resnet_version,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=True,
        try_keep_original_fc=True,
        tune_fc_only=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        self.loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        # create accuracy metric
        self.acc = Accuracy(task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes)
        # Using a pretrained ResNet backbone
        self.resnet_model = self.resnets[resnet_version](pretrained=transfer)
        if not (try_keep_original_fc and self.num_classes == 1000):
            # Replace old FC layer with Identity so we can train our own
            linear_size = list(self.resnet_model.children())[-1].in_features
            # replace final layer for fine tuning
            self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.acc(preds.argmax(axis=1), y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)


if __name__ == "__main__":
    # Unit test
    num_classes = 1000
    resnet_version = 18
    pl_module = ResNetClassifier(num_classes, resnet_version)
    print(pl_module.resnet_model)
