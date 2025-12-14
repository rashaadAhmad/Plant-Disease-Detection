import timm
import json

from pathlib import Path
from utils import inference, inferencev2, plot_image, get_file_path


AVAILABLE_MODELS = [
    "resnet18",
    "effecientnet-b0"
]

class Dl2_interface:
    def __init__(self, name: str):
        """
        Initialize the model engine with a specified model architecture.
        
        Args:
            name (str): The name of the model to initialize. Must be one of the available 
                       models defined in AVAILABLE_MODELS.
        
        Attributes:
            model: The instantiated model loaded based on the provided name.
            class_names: List of class names loaded from the model's configuration.
            path (str): File path to the pretrained weights checkpoint for ResNet stage 2 
                       fine-tuning model.
        
        Raises:
            ValueError: If the provided model name is not in AVAILABLE_MODELS.
        """
        self.path = get_file_path("checkpoints")
        self.class_names = self._load_class_names()
        self.model = self._create_model(name=name)

    def _load_class_names(self):
        with open(get_file_path("class_names.json")) as f:
            return json.load(f)

    def _create_model(self, name):
        if name == AVAILABLE_MODELS[0]:
            return timm.create_model(
                name,
                pretrained=False,
                num_classes=len(self.class_names),
                checkpoint_path=self.path + "\\" + "resnet_stage2_finetuning.pth"
            )
        elif name == AVAILABLE_MODELS[1]:
            return timm.create_model(
                name,
                pretrained=False,
                num_classes=len(self.class_names),
                checkpoint_path=self.path + "\\" + "effecientnet_stage2_finetuning.pth"
            )
        else:
            raise ValueError(f"name must be in {AVAILABLE_MODELS}")
    
    def inference_basic(self, img_or_frame):
        return inference(
            img=img_or_frame,
            model=self.model
        )
    

    def inference(self, img_or_frame):
        original, heatmap, overlay, pred_idx = inferencev2(img=img_or_frame, model=self.model)
        return plot_image(
            [original, heatmap, overlay, pred_idx],
            ["Input", "Heatmap", f"Inference: {self.class_names[str(pred_idx)]}"],
            cmap="jet"
        )

    
    def whatDevice(self) -> str:
        return str(next(self.model.parameters()).device)
    

    def AVAILABLE_MODELS(self):
        return AVAILABLE_MODELS



test = Dl2_interface("resnet18")