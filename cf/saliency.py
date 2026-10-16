"""Target-class Grad-CAM support maps for frozen pixel classifiers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cf.classifier import classifier_logits


def resolve_gradcam_layer(model, layer_name=""):
    """Resolve an explicit module path, or a final spatial feature block.

    DenseNet and ResNet use their final dense/residual block. Other CNNs
    fall back to their last Conv2d. Paths are relative to bundle['model'].
    """
    modules = dict(model.named_modules())
    if layer_name:
        if layer_name not in modules:
            raise ValueError(f"Grad-CAM layer {layer_name!r} was not found in the classifier.")
        return modules[layer_name]
    for suffix in ("features.denseblock4", "layer4"):
        candidates = [module for name, module in modules.items()
                      if name == suffix or name.endswith("." + suffix)]
        if candidates:
            return candidates[-1]
    candidates = [module for module in modules.values() if isinstance(module, nn.Conv2d)]
    if not candidates:
        raise ValueError("Grad-CAM requires a spatial CNN layer; set CF_GRADCAM_LAYER explicitly.")
    return candidates[-1]


@torch.enable_grad()
def target_gradcam(bundle, x_img, y_target, layer_name=""):
    """Compute positive, normalized target-logit Grad-CAM, [B,1,H,W].

    This uses a separate classifier pass on the detached current view. It
    does not change the CE gradient used for counterfactual guidance or
    accumulate parameter gradients. Hooks are removed even if forward fails.
    """
    layer = resolve_gradcam_layer(bundle["model"], layer_name)
    activations = []

    def capture(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("The Grad-CAM layer must return a [B,C,H,W] tensor.")
        activations.append(output)

    hook = layer.register_forward_hook(capture)
    try:
        x = x_img.detach().requires_grad_(True)
        logits = classifier_logits(bundle, x)
        if len(activations) != 1:
            raise ValueError("The Grad-CAM layer must run exactly once per classifier forward.")
        features = activations[0]
        score = logits.gather(1, y_target.reshape(-1, 1).long()).sum()
        gradients = torch.autograd.grad(score, features)[0]
        weights = gradients.float().mean(dim=(-2, -1), keepdim=True)
        cam = (weights * features.detach().float()).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=x_img.shape[-2:], mode="bilinear", align_corners=False)
        peak = cam.flatten(1).amax(dim=1).reshape(-1, 1, 1, 1)
        return (cam / peak.clamp_min(1e-8)).detach()
    finally:
        hook.remove()
