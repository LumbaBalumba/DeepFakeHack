from models.arch.attached_oml_arch import AttachedNetArch

from models.abstract_model import ABCModel


class AttachedNet(ABCModel):
    def __init__(
        self,
        model_name="resnet",
        weights=None,
        freeze_source=0,
        pretrained=None,
        extra_layers=None,
    ):
        super().__init__()

        self.model = AttachedNetArch(
            model_name, weights, freeze_source, pretrained, extra_layers
        )
        self.transform = self.model.transform
