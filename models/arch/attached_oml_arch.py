from collections import OrderedDict

from torch import nn
from oml.models import ResnetExtractor
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained

PRETRAINED_PATH = "pretrained_weights/"


class AttachedNetArch(nn.Module):
    def __init__(
        self,
        model_type="resnet",
        weights=None,
        custom_pretrained=None,
        extra_layers=None,
    ):
        super().__init__()

        feature_extractor = None
        if model_type == "resnet":
            if not custom_pretrained:
                feature_extractor = ResnetExtractor.from_pretrained(weights)
            else:
                feature_extractor = ResnetExtractor(PRETRAINED_PATH + custom_pretrained)
        elif model_type == "vit":
            if not custom_pretrained:
                feature_extractor = ViTExtractor.from_pretrained(weights)
            else:
                feature_extractor = ViTExtractor(PRETRAINED_PATH + custom_pretrained)
        else:
            raise NameError("wrong faeture extractor name")

        self.transform, _ = get_transforms_for_pretrained(weights)

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("feature extractor", feature_extractor),
                    ("extra layers", self.parse_layers(extra_layers)),
                ]
            )
        )

    def parse_layers(self, layers):
        layer_list = []
        iteration = 0
        for layer in layers:
            layer_name, layer_params = layer
            layer_list.append(
                (
                    f"custom-layer{iteration}-{layer_name}",
                    getattr(nn, layer_name)(**layer_params),
                )
            )
            iteration += 1
        return nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        output = self.model(x)

        return output
