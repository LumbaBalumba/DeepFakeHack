import oml.models as oml_models
from oml.registry import get_transforms_for_pretrained

from deepfakehack.models.abstract_model import ABCModel


class OMLModel(ABCModel):
    def __init__(self, model_name: str, loader: str):
        super().__init__()
        self.model = getattr(oml_models, loader).from_pretrained(model_name)
        self.transform, _ = get_transforms_for_pretrained(model_name)
