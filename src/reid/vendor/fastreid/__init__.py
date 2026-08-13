"""Vendored subset of FastReID's model definition. See PROVENANCE.md."""

from .batch_norm import IBN, get_norm
from .non_local import Non_local
from .pooling import GeneralizedMeanPooling, GeneralizedMeanPoolingP
from .resnet import BasicBlock, Bottleneck, ResNet
from .se_layer import SELayer

__all__ = ["IBN", "get_norm", "Non_local", "GeneralizedMeanPooling",
           "GeneralizedMeanPoolingP", "BasicBlock", "Bottleneck", "ResNet",
           "SELayer"]
