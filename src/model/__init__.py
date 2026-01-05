from .resnet import *
from .pyramidnet import *
from .wideresnet import *



MODEL_MAP =  {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "wideresnet28": WideResNet28,
    "wideresnet34": WideResNet34,
    "pyramidnet": PyramidNet,
}