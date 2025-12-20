import os
from pathlib import Path

import cv2
import numpy as np
import torch

from torchvision import transforms
import torch.nn.functional as F


# Imagenet stats fo0r normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Precompute as tensors for faster operations
MEAN_TENSOR = torch.tensor(MEAN).view(3, 1, 1)
STD_TENSOR = torch.tensor(STD).view(3, 1, 1)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# File path cache
_file_path_cache = {}


def get_file_path(name: str) -> str:
    """Fast file search: finds the file path in dl2 directory or parent src directory"""
    if name in _file_path_cache:
        return _file_path_cache[name]
    
    dl2_dir = Path(__file__).parent
    src_dir = dl2_dir.parent
    
    # Try direct match in dl2 first (fastest)
    direct_path = dl2_dir / name
    if direct_path.exists():
        _file_path_cache[name] = str(direct_path)
        return _file_path_cache[name]
    
    # Try direct match in src
    src_path = src_dir / name
    if src_path.exists():
        _file_path_cache[name] = str(src_path)
        return _file_path_cache[name]
    
    # Search recursively in dl2
    for path in dl2_dir.rglob(name):
        _file_path_cache[name] = str(path)
        return _file_path_cache[name]
    
    # Search recursively in src
    for path in src_dir.rglob(name):
        _file_path_cache[name] = str(path)
        return _file_path_cache[name]
    
    raise FileNotFoundError(f"File '{name}' not found in {dl2_dir} or {src_dir}")


def get_file_names(directory):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    return file_names


def inference(img, model: torch.nn.Module, use_fp16: bool = False):
    model.eval()
    device = next(model.parameters()).device

    with torch.inference_mode():
        transformed_img = transform(img)
        transformed_img = transformed_img.unsqueeze(0).to(device)  # batch dimension [1, C, H, W]
        
        if use_fp16 and device.type == 'cuda':
            transformed_img = transformed_img.half()
        
        output = model(transformed_img)
        return torch.argmax(output, dim=1).item()



class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_map = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def _save_feature_map(module, input, output):
            self.feature_map = output

        def _save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(_save_feature_map))
        self.hooks.append(self.target_layer.register_backward_hook(_save_gradients))

    def __call__(self, input_image, target_class=None):
        self.model.eval()
        logits = self.model(input_image)

        if target_class is None:
            target_class = logits.argmax(dim=1)

        self.model.zero_grad()

        # Calculate gradients of the predicted class score with respect to the output
        one_hot = torch.zeros_like(logits).to(input_image.device)
        for i in range(input_image.shape[0]):
            one_hot[i, target_class[i]] = 1.0

        logits.backward(gradient=one_hot, retain_graph=True)

        # Retrieve stored feature map and gradients
        feature_map = self.feature_map
        gradients = self.gradients

        # Global average pooling of gradients to get neuron importance weights
        if gradients is None:
            raise RuntimeError("Gradients are None. Ensure backward hook was triggered.")
        if feature_map is None:
            raise RuntimeError("Feature map is None. Ensure forward hook was triggered.")
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Multiply the feature map by the importance weights and sum across channel dimension
        cam = weights * feature_map
        cam = torch.sum(cam, dim=1, keepdim=True)

        # Apply ReLU activation
        cam = F.relu(cam)

        # Resize the heatmap to the original input image size
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)

        # Normalize the heatmap to a range between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.squeeze().detach().cpu().numpy()
    
    def remove_hooks(self):
        # Remove hooks to prevent memory leaks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        # Remove hooks to prevent memory leaks
        self.remove_hooks()


def inferencev2(img, model: torch.nn.Module, target_layer=None):
    """
    Performs inference and generates Grad-CAM visualization.
    
    Args:
        img: PIL Image or tensor
        model: PyTorch model
        target_layer: Layer to use for Grad-CAM (defaults to last conv layer)
    
    Returns:
        tuple: (original_image, heatmap, overlay_image, predicted_class_index)
            - original_image: numpy array [H, W, 3] in RGB format (denormalized)
            - heatmap: numpy array [H, W] grayscale heatmap
            - overlay_image: numpy array [H, W, 3] in RGB format with overlay
            - predicted_class_index: int
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Auto-detect target layer if not provided
    if target_layer is None:
        if hasattr(model, 'layer4'):  # ResNet
            target_layer = list(getattr(model, 'layer4').children())[-1]
        elif hasattr(model, 'features'):  # EfficientNet, VGG, etc.
            target_layer = list(getattr(model, 'features').children())[-1]
        else:
            raise ValueError("Could not auto-detect target layer. Please provide target_layer parameter.")
    
    # Transform and prepare image
    transformed_img = transform(img)
    input_batch = transformed_img.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.inference_mode():
        output = model(input_batch)
        predicted_class = torch.argmax(output, dim=1).item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam(input_batch, target_class=torch.tensor([predicted_class]).to(device))
    gradcam.remove_hooks()
    
    # Denormalize the image for display
    img_for_display = transformed_img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array(MEAN)
    std = np.array(STD)
    img_for_display = std * img_for_display + mean
    img_for_display = np.clip(img_for_display, 0, 1)
    
    # Create overlay
    img_np = (img_for_display * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = (255 * heatmap_resized).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Overlay with alpha blending
    alpha = 0.5
    overlay_bgr = cv2.addWeighted(img_bgr, alpha, heatmap_colored, 1 - alpha, 0)
    
    # Convert back to RGB for display
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    return img_for_display, heatmap, overlay_rgb, predicted_class


def plot_image(images, titles=None, figsize=None, cmap=None):
    """
    Simple plotting function for displaying images side by side.
    
    Args:
        images: Single image array or list of image arrays
        titles: String or list of strings for image titles
        figsize: Tuple (width, height) for figure size
        cmap: Colormap (e.g., 'gray', 'jet') for grayscale images
    """
    import matplotlib.pyplot as plt
    
    # Handle single image case
    if not isinstance(images, list):
        images = [images]
    
    # Handle titles
    if titles is None:
        titles = [None] * len(images)
    elif isinstance(titles, str):
        titles = [titles]
    
    num_images = len(images)
    
    # Auto-set figure size
    if figsize is None:
        figsize = (6 * num_images, 6)
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    # Handle single image case
    if num_images == 1:
        axes = [axes]
    
    # Plot each image
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        if title:
            ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()