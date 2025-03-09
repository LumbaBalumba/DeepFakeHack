from models.arch.attached_oml_arch import Attached_Net_Arch

from models.abstract_model import ABCModel


class Attached_Net(ABCModel):
    def __init__(
        self,
        model_name="resnet",
        weights=None,
        freeze_source=0,
        pretrained=None,
        extra_layers=None,
    ):
        super(ABCModel, self).__init__()
        super(Attached_Net, self).__init__()

        self.model = Attached_Net_Arch(
            model_name, weights, freeze_source, pretrained, extra_layers
        )
        self.transform = self.model.transform
