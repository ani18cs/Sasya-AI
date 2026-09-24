"""
Grad-CAM Explainability Layer using pytorch-grad-cam.
"""
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

def get_gradcam_overlay(model, input_tensor, rgb_img, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    visualization = show_cam_on_image(rgb_img / 255.0, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam
