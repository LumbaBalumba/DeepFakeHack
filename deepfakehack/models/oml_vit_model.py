from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained

from deepfakehack.models.abstract_model import ABCModel


class OMLVITModel(ABCModel):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = ViTExtractor.from_pretrained(model_name)
        self.transform, _ = get_transforms_for_pretrained(model_name)
