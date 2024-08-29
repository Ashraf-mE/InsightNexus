import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms, utils, models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import base64
from io import BytesIO

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def efficientnet():
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    old_fc = model.classifier[-1]
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features=7, bias=True)
    model.classifier[-1] = new_fc
    return model

def load_finetuned_model(model_path):
    model = efficientnet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_gradcam_visualization(model, image_tensor):
    """
    Generates a GradCAM visualization for a given image tensor using the specified model.
    
    Args:
    - model (torch.nn.Module): The model to be used for GradCAM.
    - image_tensor (torch.Tensor): The input image tensor to visualize.
    
    Returns:
    - visualization (np.ndarray): The GradCAM visualization overlayed on the input image.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Define the target layer for GradCAM
    target_layer = model.features[-1]  # Adjust this as necessary based on your model architecture
    target_layers = [target_layer]
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Perform prediction
    with torch.no_grad():
        prediction = model(image_tensor)
        _, index = torch.max(prediction, 1)
    
    # Define the target class for GradCAM
    targets = [ClassifierOutputTarget(index)]
    
    # Generate the GradCAM heatmap
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]  # Take the first item in the batch
    
    # Convert image tensor to numpy array
    image_tensor_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    image_tensor_np = np.clip(image_tensor_np, 0, 1)
    
    # Generate and return the visualization
    visualization = show_cam_on_image(image_tensor_np, grayscale_cam, use_rgb=True)
    
    return visualization


def image_to_base64(image_array):
    """
    Convert a NumPy array (uint8 RGB image) to a base64 encoded string.
    
    Args:
    - image_array (np.ndarray): The image array to convert.
    
    Returns:
    - base64_str (str): The base64 encoded string of the image.
    """
    # Ensure the image array is in the correct format
    if image_array.dtype != np.uint8:
        raise ValueError("Image array must be of dtype uint8")
    
    # Convert NumPy array to PIL Image (assuming image_array is in RGB format)
    image = Image.fromarray(image_array)
    
    # Save the image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # PNG is commonly used for web images
    
    # Encode the image as base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str