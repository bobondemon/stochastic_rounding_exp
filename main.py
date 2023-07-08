import copy

import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from resnet_module import ResNetClassifier
from utilities.helper import forward_pre_hook_fn, fuse_all_conv_and_bn_inplace
from imagenet_dataset_and_loader import ImageNetKaggle, ImageNetValTestDataLoader

DATA_ROOT_DIR = "D:\WORKINGSPACE\Corpus\ImageNet"


if __name__ == "__main__":
    calculate_FP32 = False
    # ==================== [Config START]
    quant_first_layer = True
    fuse_bn_into_conv = True
    qconf = {"is_symmetric": True, "is_per_channel": False, "is_int4": True, "rounding": "round"}
    # ==================== [Config END]

    def get_pl_module(resnet_version=18):
        num_classes = 1000
        pl_module = ResNetClassifier(num_classes, resnet_version, tune_fc_only=False)
        return pl_module

    resnet_version = 18
    pl_module = get_pl_module(resnet_version)

    if fuse_bn_into_conv:
        fuse_all_conv_and_bn_inplace(pl_module.resnet_model)

    def get_test_pl_dataloader(batch_size, num_workers=4):
        # calculate accuracy before quantize
        dataset = ImageNetKaggle(DATA_ROOT_DIR, "val")
        return ImageNetValTestDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    batch_size = 128
    num_workers = 4
    pl_dataloader = get_test_pl_dataloader(batch_size, num_workers)

    trainer = Trainer(gpus=1)

    # calculate accuracy BEFORE quantize
    # resnet18: 0.6976000070571899
    # resnet50: 0.76146000623703
    if calculate_FP32:
        test_result = trainer.test(pl_module, pl_dataloader, verbose=False)
        test_acc = test_result[0]["test_acc_epoch"]
        print(f"test_acc BEFORE quantize = {test_acc}")

    # register pre_hook
    handles = []
    if quant_first_layer:
        pl_module.resnet_model.conv1.qconf = copy.deepcopy(qconf)
        handles += [pl_module.resnet_model.conv1.register_forward_pre_hook(forward_pre_hook_fn)]
    else:
        for m in pl_module.resnet_model.modules():
            if isinstance(m, nn.Conv2d):
                m.qconf = copy.deepcopy(qconf)
                handles += [m.register_forward_pre_hook(forward_pre_hook_fn)]

    print("=" * 20 + f" Having {len(handles)} number of quantized weights")

    dummy_img = torch.randn(1, 3, 224, 224)  # [1, 3, 224, 224]
    pl_module.eval()  # set to eavl(), or it'll affect the running mean/var in BN
    pl_module(dummy_img)  # will calculate scale and zero_point then do fake quant for weights

    # calculate accuracy AFTER quantize
    test_result = trainer.test(pl_module, pl_dataloader, verbose=False)
    test_acc = test_result[0]["test_acc_epoch"]
    print(f"test_acc AFTER quantize = {test_acc}")

    # remember to remove the handles
    for h in handles:
        h.remove()
