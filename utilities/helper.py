import torch
from utilities.fake_quant import get_qparam4weight
from torch import nn


# ref from: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
def fuse_conv_and_bn_inplace(conv, bn):
    #
    # init
    # fusedconv = torch.nn.Conv2d(
    #     conv.in_channels,
    #     conv.out_channels,
    #     kernel_size=conv.kernel_size,
    #     stride=conv.stride,
    #     padding=conv.padding,
    #     bias=True,
    # )
    assert conv.out_channels == bn.weight.size(0), "[ERROR]: Conv's output channel is not equal to BN's shape"
    #
    # prepare filters
    # conv.weight: [out_ch, in_ch, k1, k2]
    w_conv = conv.weight.clone().view(conv.out_channels, -1)  # [out_ch, in_ch*k1*k2]
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))  # [out_ch, out_ch]
    conv.weight.data.copy_(torch.mm(w_bn, w_conv).view(conv.weight.size()))  # [out_ch, in_ch*k1*k2]
    #
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias  # [out_ch, ]
    else:
        b_conv = torch.zeros(conv.weight.size(0))  # [out_ch, ]
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))  # [out_ch, ]
    if conv.bias is None:
        conv.bias = nn.Parameter(torch.matmul(w_bn, b_conv) + b_bn)
    else:
        conv.bias.data.copy_(torch.matmul(w_bn, b_conv) + b_bn)
    #
    # we're done
    # reset bn, so that weight is 1 and bias is 0
    bn.reset_parameters()


def fuse_all_conv_and_bn_inplace(resnet18):
    # only fuse resnet18
    if len(list(resnet18.modules())) == 68:
        # Merge all Conv+BatchNorm
        fuse_conv_and_bn_inplace(resnet18.conv1, resnet18.bn1)
        fuse_conv_and_bn_inplace(resnet18.layer1[0].conv1, resnet18.layer1[0].bn1)
        fuse_conv_and_bn_inplace(resnet18.layer1[0].conv2, resnet18.layer1[0].bn2)
        fuse_conv_and_bn_inplace(resnet18.layer1[1].conv1, resnet18.layer1[1].bn1)
        fuse_conv_and_bn_inplace(resnet18.layer1[1].conv2, resnet18.layer1[1].bn2)
        fuse_conv_and_bn_inplace(resnet18.layer2[0].conv1, resnet18.layer2[0].bn1)
        fuse_conv_and_bn_inplace(resnet18.layer2[0].conv2, resnet18.layer2[0].bn2)
        fuse_conv_and_bn_inplace(resnet18.layer2[0].downsample[0], resnet18.layer2[0].downsample[1])
        fuse_conv_and_bn_inplace(resnet18.layer2[1].conv1, resnet18.layer2[1].bn1)
        fuse_conv_and_bn_inplace(resnet18.layer2[1].conv2, resnet18.layer2[1].bn2)
        fuse_conv_and_bn_inplace(resnet18.layer3[0].conv1, resnet18.layer3[0].bn1)
        fuse_conv_and_bn_inplace(resnet18.layer3[0].conv2, resnet18.layer3[0].bn2)
        fuse_conv_and_bn_inplace(resnet18.layer3[0].downsample[0], resnet18.layer3[0].downsample[1])
        fuse_conv_and_bn_inplace(resnet18.layer3[1].conv1, resnet18.layer3[1].bn1)
        fuse_conv_and_bn_inplace(resnet18.layer3[1].conv2, resnet18.layer3[1].bn2)
        fuse_conv_and_bn_inplace(resnet18.layer4[0].conv1, resnet18.layer4[0].bn1)
        fuse_conv_and_bn_inplace(resnet18.layer4[0].conv2, resnet18.layer4[0].bn2)
        fuse_conv_and_bn_inplace(resnet18.layer4[0].downsample[0], resnet18.layer4[0].downsample[1])
        fuse_conv_and_bn_inplace(resnet18.layer4[1].conv1, resnet18.layer4[1].bn1)
        fuse_conv_and_bn_inplace(resnet18.layer4[1].conv2, resnet18.layer4[1].bn2)


def forward_pre_hook_fn(module, inps):
    if hasattr(module, "qconf") and not hasattr(module, "scale"):
        # nn.Linear and nn.Conv2d have parameters: `weight` and `bias`
        # we calculate qparam for `weight` (per channel)
        qparam4weight = None
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if isinstance(module, nn.Linear):
                module.qconf["is_per_channel"] = False
            qparam4weight = get_qparam4weight(module.qconf)
        else:
            raise ValueError(f"Only support nn.Conv2d and nn.Linear, but got {type(module)}")
        w_hat, scale, zero_point = qparam4weight(module.weight)
        module.register_buffer("scale", scale)
        module.register_buffer("zero_point", zero_point)
        # modify the `weight` parameters of the module
        # note that for the next time fowward, model will use quantized weight (w_hat) to inference
        # module.register_buffer("w_hat", w_hat)
        module.weight.data = w_hat.data
        # let `inps` be intact
    return inps
