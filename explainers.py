import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision import models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic  # For better spectrogram segmentation
import shap
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from captum.attr import DeepLift
# Correct imports for explainers.py
import torchvision.models as models
import matplotlib.pyplot as plt
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    DeepLiftShap,
    visualization as viz
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image 

'''def load_model(model_path="resnet18_eeg.pt", num_classes=3):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model'''


'''class WrappedResNet(nn.Module):
    def __init__(self, original_model):
        super(WrappedResNet, self).__init__()
        self.model = original_model
        
        # Replace all ReLU layers with custom versions
        self._replace_relus()
        
    def _replace_relus(self):
        for name, module in self.model.named_children():
            if isinstance(module, nn.ReLU):
                setattr(self.model, name, nn.ReLU(inplace=False))
            elif len(list(module.children())) > 0:
                # Recursively replace in child modules
                self._replace_in_submodules(module)
                
    def _replace_in_submodules(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.ReLU(inplace=False))
            elif len(list(child.children())) > 0:
                self._replace_in_submodules(child)
                
    def forward(self, x):
        return self.model(x)'''

'''def load_model(model_path="resnet18_eeg.pt", num_classes=3):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    
    # Wrap the model to handle ReLU issues
    model = WrappedResNet(model)
    model.eval()
    return model'''

'''def load_model(model_path, num_classes=3):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model.eval()'''
def load_model(model_path="resnet18_eeg.pt", num_classes=3):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model.eval()

def generate_gradcam(image_path_clean, image_path_detailed, model, output_dir, spectrogram_bounds):
    img_clean = Image.open(image_path_clean).convert('RGB').resize((224, 224))
    img_np = np.array(img_clean).astype(np.float32) / 255.0
    input_tensor = preprocess_image(img_np, mean=[0.5]*3, std=[0.5]*3)

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Load detailed image
    img_detailed = Image.open(image_path_detailed).convert('RGBA')
    img_width, img_height = img_detailed.size
    
    # Create mask for spectrogram area only
    mask = Image.new('L', img_detailed.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Calculate spectrogram bounds in image coordinates
    x_min = int(spectrogram_bounds['xlim'][0] / spectrogram_bounds['xlim'][1]) * img_width
    x_max = img_width
    y_min = int((1 - spectrogram_bounds['ylim'][1] / spectrogram_bounds['ylim'][1]) * img_height)
    y_max = img_height
    
    draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
    
    # Resize cam overlay to match detailed image
    cam_overlay_resized = Image.fromarray(cam_overlay).resize(img_detailed.size).convert('RGBA')
    
    # Apply mask to only show heatmap on spectrogram area
    cam_overlay_resized.putalpha(mask)
    
    # Blend images
    blended = Image.alpha_composite(img_detailed, cam_overlay_resized)

    out_path = os.path.join(output_dir, "gradcam_interpretable.png")
    blended.save(out_path)
    return out_path


def generate_lime_explanation(image_path_clean, image_path_detailed, model, output_dir, num_samples=500):
    """Generate LIME explanation matching original spectrogram size"""
    try:
        # 1. Load both clean and detailed images
        img_clean = Image.open(image_path_clean).convert('RGB').resize((224, 224))
        img_detailed = Image.open(image_path_detailed)
        target_size = img_detailed.size  # Get size of original spectrogram
        
        # 2. Prepare image for LIME
        img_np = np.array(img_clean) / 255.0
        
        # 3. Create explainer
        explainer = lime_image.LimeImageExplainer()
        
        # 4. Prediction function
        def batch_predict(images):
            images = np.array(images, dtype=np.float32)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
            tensor = torch.stack([transform(img) for img in images])
            with torch.no_grad():
                return model(tensor).numpy()
        
        # 5. Generate explanation
        explanation = explainer.explain_instance(
            img_np,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        
        # 6. Get explanation mask
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5
        )
        
        # 7. Create visualization and resize to match original
        lime_img = mark_boundaries(temp, mask)
        lime_img = (lime_img * 255).astype(np.uint8)
        
        # Convert to PIL Image and resize
        lime_pil = Image.fromarray(lime_img).resize(target_size)
        
        # 8. Save output
        out_path = os.path.join(output_dir, "lime_explanation.png")
        lime_pil.save(out_path)
        
        return out_path
        
    except Exception as e:
        raise RuntimeError(f"LIME failed: {str(e)}")

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import shap

def generate_shap_explanation(image_path_clean, model, output_dir, num_samples=100):
    """Ultra-robust SHAP implementation that handles all edge cases"""
    try:
        # 1. Load and strictly validate image
        if not os.path.exists(image_path_clean):
            raise FileNotFoundError(f"Image not found at {image_path_clean}")
        
        img = Image.open(image_path_clean)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        if not isinstance(img_np, np.ndarray) or img_np.shape != (224, 224, 3):
            raise ValueError(f"Invalid image format. Expected (224,224,3) numpy array")

        # 2. Prepare model
        device = next(model.parameters()).device
        model.eval()

        # 3. Define completely safe prediction function
        def predict(imgs):
            # Convert to numpy array if needed
            if not isinstance(imgs, np.ndarray):
                imgs = np.array(imgs, dtype=np.float32)
            
            # Handle all possible input formats
            if imgs.ndim == 2:  # Flattened
                imgs = imgs.reshape(-1, 224, 224, 3)
            elif imgs.ndim == 3:  # Single image
                imgs = np.expand_dims(imgs, 0)
            elif imgs.ndim != 4:  # Invalid
                raise ValueError(f"Invalid input dim: {imgs.ndim}")
            
            # Convert to tensor with proper normalization
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
            
            with torch.no_grad():
                tensor = torch.stack([transform(x) for x in imgs]).to(device)
                return model(tensor).cpu().numpy()  # Always return numpy array

        # 4. Use the most stable explainer
        masker = shap.maskers.Image("blur(224,224)", img_np.shape)
        explainer = shap.Explainer(
            predict,
            masker,
            output_names=["EEG_Class"],
            algorithm="partition"
        )
        
        # 5. Compute SHAP values with array conversion
        explanation = explainer(
            img_np[np.newaxis, ...],  # Add batch dimension
            max_evals=num_samples,
            outputs=[0]  # First class
        )
        
        # 6. Convert explanation to proper array format
        if not hasattr(explanation, 'values'):
            raise ValueError("Invalid SHAP explanation format")
        shap_values = np.array(explanation.values)  # Force array conversion
        
        # 7. Generate properly sized visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Wider figure
        
        # Original image
        ax1.imshow(img_np)
        ax1.set_title('Original Spectrogram')
        ax1.axis('off')
        # SHAP heatmap
        heatmap = np.mean(np.abs(shap_values[0]), axis=0)  # Combine channels
        im = ax2.imshow(heatmap, cmap='jet', vmin=0)
        ax2.set_title('SHAP Importance')
        ax2.axis('off')
        
        # Add colorbar with proper sizing
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('SHAP Value', rotation=270, labelpad=15)
        
        # 8. Save output
        out_path = os.path.join(output_dir, "shap_explanation.png")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return out_path
        
    except Exception as e:
        raise RuntimeError(f"SHAP failed: {str(e)}")



def generate_integrated_gradients(image_path_clean, model, output_dir):
    """Generate publication-quality Integrated Gradients visualizations with perfect scaling."""
    import numpy as np
    from PIL import Image
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Set up professional plotting parameters
    plt.style.use('default')  # Reset to default first
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'DejaVu Sans',  # Universally available font
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False
    })

    # Enhanced scientific colormap (perceptually uniform)
    colors = [
        (0.0, 0.0, 0.5),  # deep blue
        (0.0, 0.5, 1.0),  # medium blue
        (0.9, 0.9, 0.9),  # neutral white
        (1.0, 0.8, 0.0),  # golden yellow
        (0.8, 0.0, 0.0)   # deep red
    ]
    cmap = LinearSegmentedColormap.from_list('scientific_diverging', colors, N=512)
    
    # Load and preprocess image at higher resolution
    img = Image.open(image_path_clean).convert('RGB').resize((512, 512))
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Model processing
    input_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
    input_tensor = (input_tensor - 0.5) / 0.5
    
    # Compute attributions with more steps
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, target=0, n_steps=200, internal_batch_size=4)
    
    # Process attributions
    attr_np = attributions.squeeze().cpu().detach().numpy()
    attr_np = np.transpose(attr_np, (1, 2, 0))
    attribution_heatmap = np.mean(attr_np, axis=2)
    
    # Create figure with professional layout
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    
    # Panel 1: Original Spectrogram
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(img_np)
    ax1.set_title('Original Spectrogram', pad=20)
    
    # Panel 2: Attribution Heatmap
    ax2 = fig.add_subplot(gs[1])
    vmax = np.percentile(np.abs(attribution_heatmap), 99.5)  # Robust scaling
    im2 = ax2.imshow(attribution_heatmap, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax2.set_title('Feature Attribution Heatmap', pad=20)
    
    # Professional colorbar for heatmap
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label('Attribution Strength', rotation=270, labelpad=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Panel 3: Enhanced Overlay
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(img_np, alpha=0.85)
    im3 = ax3.imshow(attribution_heatmap, cmap=cmap, alpha=0.65, vmin=-vmax, vmax=vmax)
    ax3.set_title('Attribution Overlay', pad=20)
    
    # Second colorbar
    divider = make_axes_locatable(ax3)
    cax2 = divider.append_axes("right", size="5%", pad=0.2)
    cbar2 = plt.colorbar(im3, cax=cax2)
    cbar2.set_label('Attribution Strength', rotation=270, labelpad=20, fontweight='bold')
    cbar2.ax.tick_params(labelsize=10)
    
    # Remove axis ticks for all panels
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Add super title
    plt.suptitle('Integrated Gradients Explanation', y=1.02, fontsize=18, fontweight='bold')
    
    # Save with highest quality
    out_path = os.path.join(output_dir, "integrated_gradients.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close()
    
    return out_path


'''
import torch
from captum.attr import DeepLift
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from captum.attr import DeepLift
import torchvision.transforms as transforms

from captum.attr import LayerDeepLift
import torchvision.transforms as transforms

def generate_deeplift(image_path_clean, image_path_detailed, model, output_dir):
    """Generates DeepLift explanation over the spectrogram image using LayerDeepLift."""

    # Load and preprocess image
    image = Image.open(image_path_clean).convert('RGB').resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor.requires_grad_()

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = torch.argmax(output, dim=1).item()

    # Use LayerDeepLift (more stable with reused layers)
    target_layer = model.layer4[1].relu  # Choose a ReLU layer
    deeplift = LayerDeepLift(model, target_layer)
    baseline = torch.zeros_like(input_tensor)

    attributions = deeplift.attribute(inputs=input_tensor, baselines=baseline, target=pred_label)
    
    # Normalize and prepare heatmap
    attr_np = attributions.squeeze().detach().numpy()
    attr_np = np.transpose(attr_np, (1, 2, 0))  # Convert to HWC
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)

    # Overlay on image
    img_np = np.array(image).astype(np.float32) / 255.0
    heatmap = np.uint8(255 * attr_np)
    heatmap_img = Image.fromarray(heatmap).resize((450, 350)).convert('RGBA')

    img_detailed = Image.open(image_path_detailed).resize((450, 350)).convert('RGBA')
    blended = Image.blend(img_detailed, heatmap_img, alpha=0.5)

    out_path = os.path.join(output_dir, "deeplift_interpretable.png")
    blended.save(out_path)
    return out_path


from captum.attr import DeepLift, GuidedBackprop

import copy
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from captum.attr import DeepLift, GuidedBackprop
import streamlit as st  # Added for streamlit compatibility

def get_deeplift_compatible_model(original_model):
    """Create a DeepLIFT-compatible version of the model"""
    model = copy.deepcopy(original_model)
    
    # Replace all ReLUs with custom versions that store inputs/outputs
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            new_relu = nn.ReLU(inplace=False)
            # Store inputs/outputs for DeepLIFT
            new_relu.register_forward_hook(lambda m, inp, out: setattr(m, "input", inp[0]))
            new_relu.register_forward_hook(lambda m, inp, out: setattr(m, "output", out))
            # Need to use setattr properly for nested modules
            *path, final = name.split('.')
            current = model
            for p in path:
                current = getattr(current, p)
            setattr(current, final, new_relu)
    
    return model

def _generate_deeplift(image_path_clean, image_path_detailed, model, output_dir, target_class):
    """Core DeepLIFT implementation"""
    # Load and preprocess image
    img_clean = Image.open(image_path_clean).convert('RGB').resize((224, 224))
    img_np = np.array(img_clean).astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float()
    
    # Initialize DeepLIFT
    dl = DeepLift(model)
    
    # Create baseline (reference input)
    baseline = torch.zeros_like(input_tensor)
    
    # Compute attributions
    attributions = dl.attribute(input_tensor,
                              baselines=baseline,
                              target=target_class,
                              return_convergence_delta=False)
    
    # Process attributions
    attr_np = attributions.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())
    
    # Create heatmap
    heatmap = np.sum(attr_np, axis=2)
    heatmap = np.uint8(255 * heatmap)
    
    # Load detailed image
    img_detailed = Image.open(image_path_detailed).resize((450, 350)).convert('RGBA')
    
    # Create overlay
    plt.figure(figsize=(6, 4))
    plt.imshow(img_detailed)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f'DeepLIFT Explanation (Class {target_class})')
    
    # Save result
    out_path = os.path.join(output_dir, "deeplift_interpretable.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return out_path, target_class

def _generate_guided_backprop(image_path_clean, image_path_detailed, model, output_dir, target_class):
    """Precise spectrogram-only overlay with clean visualization"""
    try:
        # Load images
        img_clean = Image.open(image_path_clean).convert('RGB').resize((224, 224))
        img_detailed = Image.open(image_path_detailed).convert('RGBA')
        
        # Convert to tensor
        img_np = np.array(img_clean).astype(np.float32) / 255.0
        input_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float()
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = model(input_tensor)
                target_class = torch.argmax(output).item()
        
        # Compute attribution
        gbp = GuidedBackprop(model)
        attribution = gbp.attribute(input_tensor, target=target_class)
        
        # Process attribution
        attr_np = attribution.squeeze().permute(1, 2, 0).abs().cpu().detach().numpy()
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())
        heatmap = np.sum(attr_np, axis=2)
        
        # Create figure with constrained layout
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111)
        
        # Display detailed image
        ax.imshow(img_detailed)
        
        # Spectrogram boundary coordinates (adjust these based on your image)
        spectrogram_bounds = {
            'left': 0.15,    # 15% from left
            'right': 0.85,    # 85% from left
            'bottom': 0.15,   # 15% from bottom
            'top': 0.85       # 85% from bottom
        }
        
        # Convert ratios to pixel coordinates
        img_width, img_height = img_detailed.size
        x0 = spectrogram_bounds['left'] * img_width
        x1 = spectrogram_bounds['right'] * img_width
        y0 = spectrogram_bounds['bottom'] * img_height
        y1 = spectrogram_bounds['top'] * img_height
        
        # Resize heatmap to spectrogram dimensions
        heatmap_resized = np.array(Image.fromarray(heatmap).resize(
            (int(x1 - x0), int(y1 - y0)),
            resample=Image.BILINEAR
        ))
        
        # Apply overlay only to spectrogram region
        ax.imshow(heatmap_resized,
                cmap='jet',
                alpha=0.6,  # Slightly more transparent
                extent=[x0, x1, y1, y0],  # Note y1,y0 for proper orientation
                zorder=10)
        
        # Remove axes for cleaner look
        ax.axis('off')
        
        out_path = os.path.join(output_dir, "spectrogram_attribution.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150, transparent=True)
        plt.close()
        
        return out_path, target_class
        
    except Exception as e:
        raise RuntimeError(f"Visualization failed: {str(e)}")
    
def generate_deeplift_or_fallback(image_path_clean, image_path_detailed, model, output_dir, target_class=None):
    """Robust DeepLIFT implementation with automatic fallback"""
    try:
        # First try with modified model
        dl_model = get_deeplift_compatible_model(model)
        result = _generate_deeplift(image_path_clean, image_path_detailed, dl_model, 
                                  output_dir, target_class)
        return result
    except Exception as e:
        st.warning("DeepLIFT failed, trying Guided Backprop as alternative...")
        try:
            return _generate_guided_backprop(image_path_clean, image_path_detailed, 
                                           model, output_dir, target_class)
        except Exception as e:
            raise RuntimeError(
                "Could not generate DeepLIFT or Guided Backprop explanations. " 
                "This model architecture isn't fully compatible. Try GradCAM instead."
            )'
'''

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import os

def generate_spectrogram_attributions(model, clean_img_path, detailed_img_path, output_dir, target_class=None):
    """Generate precise attribution overlay on spectrogram"""
    try:
        # Load and preprocess images
        clean_img = Image.open(clean_img_path).convert('RGB').resize((224, 224))
        detailed_img = Image.open(detailed_img_path).convert('RGBA')
        
        # Convert to tensor
        img_np = np.array(clean_img).astype(np.float32) / 255.0
        input_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float()
        input_tensor.requires_grad = True
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = model(input_tensor)
                target_class = torch.argmax(output).item()
        
        # Compute attributions
        ig = IntegratedGradients(model)
        attributions = ig.attribute(input_tensor,
                                  baselines=torch.zeros_like(input_tensor),
                                  target=target_class,
                                  n_steps=50)
        
        # Process attributions
        attr_np = attributions.squeeze().permute(1, 2, 0).abs().cpu().detach().numpy()
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())
        heatmap = np.sum(attr_np, axis=2)
        
        # Create figure with exact dimensions of detailed image
        dpi = 100
        fig_width = detailed_img.width / dpi
        fig_height = detailed_img.height / dpi
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Display detailed spectrogram
        ax.imshow(detailed_img)
        
        # Define spectrogram bounds (adjust these values for your images)
        spec_left = 0.15  # 15% from left
        spec_right = 0.85  # 85% from left
        spec_top = 0.15   # 15% from top
        spec_bottom = 0.85 # 85% from top
        
        # Calculate pixel coordinates
        img_width, img_height = detailed_img.size
        left_px = int(spec_left * img_width)
        right_px = int(spec_right * img_width)
        top_px = int(spec_top * img_height)
        bottom_px = int(spec_bottom * img_height)
        
        # Resize heatmap to spectrogram dimensions
        heatmap_resized = np.array(Image.fromarray(heatmap).resize((right_px - left_px, bottom_px - top_px),resample=Image.BILINEAR))
        
        # Apply overlay only to spectrogram region
        ax.imshow(heatmap_resized,
                cmap='jet',
                alpha=0.6,  # Optimal transparency
                extent=[left_px, right_px, bottom_px, top_px],
                zorder=10)
        
        ax.axis('off')
        
        # Save result
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"attribution_{target_class}.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
        print(f"[DEBUG] Attribution saved at {out_path}")
        
        return out_path, target_class
        
    except Exception as e:
        raise RuntimeError(f"Visualization failed: {str(e)}")

def find_spectrogram_bounds(image_path):
    """Helper to determine exact spectrogram coordinates"""
    img = Image.open(image_path)
    plt.imshow(img)
    print("Click on spectrogram corners (top-left then bottom-right)")
    coords = plt.ginput(2)
    plt.close()
    return {
        'left': coords[0][0]/img.width,
        'right': coords[1][0]/img.width,
        'top': coords[0][1]/img.height,
        'bottom': coords[1][1]/img.height
    }

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from captum.attr import IntegratedGradients
import os

def generate_deeplift_style_ig(image_path_clean, model, output_dir):
    """Generate IG explanation with DeepLIFT-style visualization"""
    try:
        # Load and preprocess image
        img = Image.open(image_path_clean).convert('RGB').resize((512, 512))
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor
        input_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float()
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            target_class = torch.argmax(output).item()
        
        # Compute Integrated Gradients
        ig = IntegratedGradients(model)
        attributions = ig.attribute(input_tensor,
                                  baselines=torch.zeros_like(input_tensor),
                                  target=target_class,
                                  n_steps=100)
        
        # Process attributions with DeepLIFT-style coloring
        attr_np = attributions.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())
        heatmap = np.sum(attr_np, axis=2)
        
        # Create DeepLIFT-style colormap
        colors = [(0, 0, 0.5), (0, 0, 1), (0.5, 0.5, 0.5), (1, 0.8, 0), (1, 0, 0)]  # Blue to Red
        cmap = LinearSegmentedColormap.from_list('deeplift_style', colors)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        
        # Display original image
        ax.imshow(img_np)
        
        # Define spectrogram bounds (adjust these)
        width, height = img.size
        spec_left = int(width * 0.15)
        spec_right = int(width * 0.85)
        spec_top = int(height * 0.15)
        spec_bottom = int(height * 0.85)
        
        # Resize and overlay heatmap
        heatmap_resized = np.array(Image.fromarray(heatmap).resize(
            (spec_right-spec_left, spec_bottom-spec_top),
            resample=Image.BILINEAR
        ))
        
        # Apply DeepLIFT-style overlay
        overlay = ax.imshow(heatmap_resized,
                          cmap=cmap,
                          alpha=0.6,
                          extent=[spec_left, spec_right, spec_bottom, spec_top],
                          zorder=10)
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(overlay, cax=cax, label='Attribution Strength')
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save result
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "deeplift_style_explanation.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1, dpi=120)
        plt.close()
        
        return out_path
        
    except Exception as e:
        raise RuntimeError(f"DeepLIFT-style visualization failed: {str(e)}")