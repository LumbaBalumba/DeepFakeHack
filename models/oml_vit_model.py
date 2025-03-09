from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained

from models.abstract_model import ABCModel


class OML_ViT_Model(ABCModel):
    def __init__(self, model_name: str):
        super(ABCModel, self).__init__()
        super(OML_ViT_Model, self).__init__()
        self.model = ViTExtractor.from_pretrained(model_name)

        self.transform, _ = get_transforms_for_pretrained(model_name)
