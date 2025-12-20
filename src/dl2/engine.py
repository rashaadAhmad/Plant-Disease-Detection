import timm
import json
import torch
import torch.nn as nn

from pathlib import Path
from .utils import inference, inferencev2, plot_image, get_file_path


AVAILABLE_MODELS = [
    "resnet18",
    "effecientnet-b0"
]

class Dl2_interface:
    def __init__(self, name: str, device: str, use_fp16: bool = False, compile_model: bool = False):
        """
        Initialize the model engine with a specified model architecture.
        
        Args:
            name (str): The name of the model to initialize. Must be one of the available 
                       models defined in AVAILABLE_MODELS.
            device (str): Device to load model on ('cuda', 'cpu', or None for auto-detect).
            use_fp16 (bool): Whether to use half precision (FP16) for faster inference on GPU.
            compile_model (bool): Whether to use torch.compile() for faster inference (PyTorch 2.0+).
        
        Attributes:
            model: The instantiated model loaded based on the provided name.
            class_names: List of class names loaded from the model's configuration.
            path (str): File path to the pretrained weights checkpoint for ResNet stage 2 
                       fine-tuning model.
        
        Raises:
            ValueError: If the provided model name is not in AVAILABLE_MODELS.
        """
        self.model_name = name
        self.use_fp16 = use_fp16
        self.compile_model = compile_model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.path = Path(get_file_path("checkpoints"))
        self.class_names = self._load_class_names()
        self.model = self._create_model(name=name)

    def _load_class_names(self):
        with open(get_file_path("class_names.json")) as f:
            return json.load(f)

    def _create_model(self, name):
        if name == AVAILABLE_MODELS[0]:
            checkpoint_path = self.path / "resnet_stage2_finetuning.pth"
            model = timm.create_model(
                name,
                pretrained=False,
                num_classes=len(self.class_names),
                checkpoint_path=str(checkpoint_path)
            )
        elif name == AVAILABLE_MODELS[1]:
            checkpoint_path = self.path / "effecientnet_stage2_finetuning.pth"
            model = timm.create_model(
                name,
                pretrained=False,
                num_classes=len(self.class_names),
                checkpoint_path=str(checkpoint_path)
            )
        else:
            raise ValueError(f"name must be in [{AVAILABLE_MODELS}]")
        
        model = model.to(self.device)
        model.eval()
        
        if self.use_fp16 and self.device.type == 'cuda':
            model = model.half()
        
        if self.compile_model:
            try:
                from typing import cast
                model = cast(nn.Module, torch.compile(model)) # This is actually a savior, damn
            except Exception:
                pass
        
        return model
    
    def inference_basic(self, img_or_frame):
        return inference(
            img=img_or_frame,
            model=self.model,
            use_fp16=self.use_fp16
        )
    

    def inference(self, img_or_frame, visualize: bool = True):
        original, heatmap, overlay, pred_idx = inferencev2(img=img_or_frame, model=self.model)
        
        if visualize:
            return plot_image(
                [original, heatmap, overlay],
                ["Input", "Heatmap", f"Inference: {self.class_names[str(pred_idx)]}"],
                cmap="jet"
            )
        else:
            return original, heatmap, overlay, pred_idx

    
    def whatDevice(self) -> str:
        return str(next(self.model.parameters()).device)
