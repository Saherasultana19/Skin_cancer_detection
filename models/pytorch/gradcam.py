"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Model Interpretability
PyTorch Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import cv2


class GradCAM:
    """
    Grad-CAM implementation for generating attention maps from CNNs
    
    Visualizes which regions of input contribute most to model predictions
    """
    
    def __init__(self, model: nn.Module, target_layer: str, use_cuda: bool = True):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer for visualization
            use_cuda: Use GPU if available
        """
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Hooks for capturing gradients
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get target layer
        target_module = self._get_module_by_name(self.model, self.target_layer)
        if target_module is None:
            raise ValueError(f"Layer '{self.target_layer}' not found in model")
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    @staticmethod
    def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get module by name from model"""
        for layer_name, module in model.named_modules():
            if layer_name == name:
                return module
        return None
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate CAM for input
        
        Args:
            input_tensor: Input image tensor (B, C, D, H, W) for 3D or (B, C, H, W) for 2D
            target_class: Target class index. If None, uses class with highest score
            
        Returns:
            CAM visualization (normalized to 0-1)
        """
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        with torch.enable_grad():
            output = self.model(input_tensor)
        
        # If output has multiple classes, select target class
        if output.dim() > 1 and output.shape[1] > 1:
            if target_class is None:
                target_class = output.argmax(dim=1)[0].item()
            score = output[:, target_class, ...].sum()
        else:
            score = output.sum()
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Calculate CAM
        gradients = self.gradients[0]  # (C, D, H, W) or (C, H, W)
        activations = self.activations[0]  # (C, D, H, W) or (C, H, W)
        
        # Compute weights
        weights = gradients.mean(dim=tuple(range(1, gradients.dim())), keepdim=True)
        
        # Compute CAM
        cam = (weights * activations).sum(dim=0, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam.squeeze().detach().cpu().numpy()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        
        return cam
    
    def visualize_segmentation(self, input_tensor: torch.Tensor, 
                              cmap: str = 'jet', alpha: float = 0.5) -> np.ndarray:
        """
        Generate visualization overlay for segmentation
        
        Args:
            input_tensor: Input image tensor
            cmap: Colormap name
            alpha: Blending factor (0-1)
            
        Returns:
            Blended visualization
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor)
        
        # Normalize input to 0-1
        input_np = input_tensor[0].permute(1, 2, 3, 0 if input_tensor.dim() == 5 else 1, 2).cpu().numpy()
        if input_np.max() > 1:
            input_np = input_np / input_np.max()
        
        # Handle different dimensions
        if input_np.ndim == 4:  # 3D: (D, H, W, C)
            # Take middle slice
            middle_slice = input_np[input_np.shape[0] // 2]
        else:  # 2D: (H, W, C)
            middle_slice = input_np
        
        # Convert CAM to colormap
        cam_colored = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), 
            getattr(cv2, f'COLORMAP_{cmap.upper()}')
        )
        cam_colored = cam_colored / 255.0
        
        # Blend
        visualization = alpha * cam_colored + (1 - alpha) * middle_slice
        
        return visualization


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation for improved localization"""
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM++ for input
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            
        Returns:
            Grad-CAM++ visualization
        """
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        with torch.enable_grad():
            output = self.model(input_tensor)
        
        # Select target class
        if output.dim() > 1 and output.shape[1] > 1:
            if target_class is None:
                target_class = output.argmax(dim=1)[0].item()
            score = output[:, target_class, ...].sum()
        else:
            score = output.sum()
        
        # First backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Calculate second derivatives
        gradients_1 = self.gradients[0].clone()
        
        # Compute spatial gradients
        spatial_grad2 = (gradients_1 ** 2).mean(dim=0, keepdim=True)
        spatial_grad3 = (gradients_1 ** 3).mean(dim=0, keepdim=True)
        
        # Compute weights
        alpha = spatial_grad2 / (2 * spatial_grad3 + 1e-8)
        relu_grad = F.relu(self.gradients[0])
        weights = (alpha * relu_grad).sum(dim=tuple(range(1, alpha.dim())), keepdim=True)
        
        # Compute Grad-CAM++
        activations = self.activations[0]
        cam = (weights * activations).sum(dim=0, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().detach().cpu().numpy()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        
        return cam


class SaliencyMap:
    """Generate saliency maps using gradients"""
    
    def __init__(self, model: nn.Module, use_cuda: bool = True):
        """
        Initialize Saliency Map generator
        
        Args:
            model: PyTorch model
            use_cuda: Use GPU if available
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def generate_saliency(self, input_tensor: torch.Tensor, 
                         target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate saliency map
        
        Args:
            input_tensor: Input image tensor with gradient enabled
            target_class: Target class
            
        Returns:
            Saliency map
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Select target class
        if target_class is None:
            target_class = output.argmax(dim=1)[0].item()
        
        score = output[0, target_class]
        
        # Backward pass
        score.backward()
        
        # Get gradients
        gradients = input_tensor.grad.data.abs()
        
        # Maximum gradient magnitude across channels
        saliency = gradients.max(dim=1)[0].squeeze().detach().cpu().numpy()
        
        # Normalize
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        return saliency


if __name__ == "__main__":
    from models.pytorch.resnet_classifier import ResNet3DClassifier
    
    # Create dummy model and data
    model = ResNet3DClassifier(num_classes=3)
    model.eval()
    
    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer='layer4')
    
    # Generate dummy input
    x = torch.randn(1, 3, 128, 128, 128)
    
    # Generate CAM
    cam = grad_cam.generate_cam(x)
    print(f"CAM shape: {cam.shape}")
    print(f"CAM range: [{cam.min():.4f}, {cam.max():.4f}]")
