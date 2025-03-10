from oml.models import ResnetExtractor
from oml.registry import get_transforms_for_pretrained

from deepfakehack.models.abstract_model import ABCModel


class OMLResnetModel(ABCModel):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = ResnetExtractor.from_pretrained(model_name)

        self.transform, _ = get_transforms_for_pretrained(model_name)
