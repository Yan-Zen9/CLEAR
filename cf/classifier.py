"""
Unified classifier interface for CLEAR (vendored from FiRe/CFgenerating).

Same calling convention as FiRe so checkpoints and usage are consistent:
    bundle = get_classifier_bundle(ckpt, device, query_label=..., classifier_mode=...)
    logits = classifier_logits(bundle, x)          # x in [-1, 1]
    prob   = classifier_target_prob(bundle, x, y_target)   # y_target = class INDEX (long)

The model outputs MULTI-class logits (2 for binary, via cat([-q, q]) for the
CelebA binary heads, or num_classes for a standard DenseNet). Guidance uses
cross_entropy(logits, y_target); the saliency uses the target-class logit.

`normalize_for_classifier` maps the diffusion image space [-1,1] -> [0,1] ->
standardize with the checkpoint's mean/std.
"""

import torch
import torch.nn as nn
from torchvision import models

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_LEGACY_MEAN = [0.5, 0.5, 0.5]
_LEGACY_STD = [0.5, 0.5, 0.5]
_LEGACY_QUERY_LABEL = 31
_CELEBAHQ_QUERY_LABEL_TO_INDEX = {20: 0, 31: 1, 39: 2}


class Identity(nn.Module):
    def forward(self, x):
        return x


class LegacyDenseNet121(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = models.densenet121(weights=None)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024

    def forward(self, x):
        return self.feat_extract(x)


class LegacyCelebABinaryClassifier(nn.Module):
    def __init__(self, checkpoint_path, query_label=_LEGACY_QUERY_LABEL, *, state_dict_payload=None):
        super().__init__()
        self.feat_extract = LegacyDenseNet121()
        self.classifier = nn.Linear(self.feat_extract.output_size, 40)
        self.query_label = int(query_label)
        payload = state_dict_payload if state_dict_payload is not None else _load_checkpoint(checkpoint_path)
        self.feat_extract.load_state_dict(payload["feat_extract"])
        self.classifier.load_state_dict(payload["classifier"])

    def forward(self, x):
        logits40 = self.classifier(self.feat_extract(x))
        q = logits40[:, self.query_label:self.query_label + 1]
        return torch.cat([-q, q], dim=1)


class CelebAHQBinaryClassifier(nn.Module):
    def __init__(self, checkpoint_path, query_label=_LEGACY_QUERY_LABEL, *, state_dict=None):
        super().__init__()
        if int(query_label) not in _CELEBAHQ_QUERY_LABEL_TO_INDEX:
            raise ValueError("CelebA-HQ classifier supports query labels 20, 31, or 39.")
        self.feat_extract = LegacyDenseNet121()
        self.classifier = nn.Linear(self.feat_extract.output_size, len(_CELEBAHQ_QUERY_LABEL_TO_INDEX))
        self.query_index = _CELEBAHQ_QUERY_LABEL_TO_INDEX[int(query_label)]
        payload = state_dict if state_dict is not None else _strip_prefixes(
            _load_checkpoint(checkpoint_path)["model_state_dict"])
        self.load_state_dict(payload, strict=False)

    def forward(self, x):
        logits3 = self.classifier(self.feat_extract(x))
        q = logits3[:, self.query_index:self.query_index + 1]
        return torch.cat([-q, q], dim=1)


class StandardDenseNetClassifier(nn.Module):
    """Generic DenseNet-121 classifier (used for medical datasets)."""

    def __init__(self, num_classes):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class StandardResNetClassifier(nn.Module):
    """torchvision ResNet-50 (ImageNet-1k, 1000-way) — the ACE ImageNet classifier
    (`models.resnet50(pretrained=True)` == ResNet50_Weights.IMAGENET1K_V1). The
    checkpoint is a BARE state_dict (features.*/fc.*), not the wrapped payloads the
    CelebA/medical loaders use. Normalized with the standard ImageNet mean/std."""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = models.resnet50(weights=None)
        if num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def _is_bare_resnet_sd(payload):
    """A raw torchvision ResNet state_dict (tensors at top level), NOT one of the
    wrapped CelebA/medical payloads."""
    if not isinstance(payload, dict):
        return False
    if "model_state_dict" in payload or ("feat_extract" in payload and "classifier" in payload):
        return False
    return "conv1.weight" in payload and "fc.weight" in payload


def _strip_prefixes(state_dict, prefixes=("module.", "model.")):
    norm = dict(state_dict)
    for p in prefixes:
        if norm and all(k.startswith(p) for k in norm):
            norm = {k[len(p):]: v for k, v in norm.items()}
    return norm


def _load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _detect_classifier_mode(payload, checkpoint_path, classifier_mode, query_label):
    if classifier_mode not in {"auto", "legacy_celeba", "celebahq"}:
        raise ValueError(f"Unsupported classifier_mode: {classifier_mode}")
    if classifier_mode != "auto":
        return classifier_mode
    if "feat_extract" in payload and "classifier" in payload:
        return "legacy_celeba"
    if "model_state_dict" in payload:
        sd = _strip_prefixes(payload["model_state_dict"])
        w = sd.get("classifier.weight")
        if (w is not None and int(w.shape[0]) == len(_CELEBAHQ_QUERY_LABEL_TO_INDEX)
                and int(query_label) in _CELEBAHQ_QUERY_LABEL_TO_INDEX
                and "celebamaskhq" in checkpoint_path.lower()):
            return "celebahq"
    return "auto"


def load_model(checkpoint_path, device, *, query_label=_LEGACY_QUERY_LABEL,
               classifier_mode="auto", checkpoint_payload=None):
    payload = checkpoint_payload if checkpoint_payload is not None else _load_checkpoint(checkpoint_path)

    if _is_bare_resnet_sd(payload):                              # torchvision ResNet-50 (ImageNet)
        num_classes = int(payload["fc.weight"].shape[0])
        model = StandardResNetClassifier(num_classes=num_classes)
        # strict=False tolerates the missing num_batches_tracked buffers in the raw hub file
        missing, unexpected = model.model.load_state_dict(payload, strict=False)
        non_bn = [k for k in missing if "num_batches_tracked" not in k]
        if non_bn or unexpected:
            print(f"  ⚠️ ResNet load: {len(non_bn)} missing (non-BN), {len(unexpected)} unexpected")
        model = model.to(device).eval()
        return model, list(range(num_classes)), None, 256, None

    mode = _detect_classifier_mode(payload, checkpoint_path, classifier_mode, int(query_label))

    if mode == "legacy_celeba":
        model = LegacyCelebABinaryClassifier(checkpoint_path, query_label=query_label,
                                             state_dict_payload=payload).to(device).eval()
        return model, ["negative", "positive"], None, 128, int(query_label)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        sd = _strip_prefixes(payload["model_state_dict"])
        if mode == "celebahq":
            model = CelebAHQBinaryClassifier(checkpoint_path, query_label=query_label,
                                             state_dict=sd).to(device).eval()
            return model, ["negative", "positive"], None, 256, int(query_label)
        w = sd.get("classifier.weight")
        if w is None:
            raise ValueError(f"Cannot infer classifier head size from: {checkpoint_path}")
        num_classes = int(w.shape[0])
        model = StandardDenseNetClassifier(num_classes=num_classes)
        # sd keys are the inner densenet's (features.*/classifier.*, after _strip_prefixes);
        # load into model.model (the densenet), not the wrapper.
        missing, unexpected = model.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  ⚠️ StandardDenseNet load: {len(missing)} missing, {len(unexpected)} unexpected")
        model = model.to(device).eval()
        return model, list(range(num_classes)), None, 128, None

    raise ValueError(f"Unsupported classifier checkpoint format: {checkpoint_path}")


def _infer_norm_from_payload(payload, *, source_path=None, classifier_mode="auto",
                             query_label=_LEGACY_QUERY_LABEL):
    if _is_bare_resnet_sd(payload):                              # ImageNet ResNet-50: ImageNet mean/std
        return torch.tensor(_MEAN), torch.tensor(_STD)
    if "feat_extract" in payload and "classifier" in payload:
        return torch.tensor(_LEGACY_MEAN), torch.tensor(_LEGACY_STD)
    if "model_state_dict" in payload:
        mode = _detect_classifier_mode(payload, source_path or "", classifier_mode, int(query_label))
        if mode == "celebahq":
            return torch.tensor(_LEGACY_MEAN), torch.tensor(_LEGACY_STD)
        return torch.tensor(_MEAN), torch.tensor(_STD)
    raise ValueError(f"Unsupported classifier checkpoint format: {source_path}")


def get_classifier_bundle(checkpoint_path, device, query_label=_LEGACY_QUERY_LABEL,
                          classifier_mode="auto"):
    """Load a classifier into a unified bundle: {model, mean, std, preprocess}."""
    payload = _load_checkpoint(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported classifier checkpoint type: {type(payload)!r}")
    mean, std = _infer_norm_from_payload(payload, source_path=checkpoint_path,
                                         classifier_mode=classifier_mode, query_label=int(query_label))
    model, _, preprocess, _, _ = load_model(checkpoint_path, device, query_label=int(query_label),
                                            classifier_mode=classifier_mode, checkpoint_payload=payload)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"model": model, "preprocess": preprocess, "mean": mean.to(device), "std": std.to(device)}


def normalize_for_classifier(x_pmf, bundle):
    """Image in [-1,1] -> [0,1] -> standardize with bundle mean/std."""
    x01 = ((x_pmf + 1.0) / 2.0).clamp(0.0, 1.0)
    mean = bundle["mean"].to(device=x_pmf.device, dtype=x_pmf.dtype).view(1, 3, 1, 1)
    std = bundle["std"].to(device=x_pmf.device, dtype=x_pmf.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def classifier_logits(bundle, x_pmf):
    """Multi-class logits for an image in [-1,1]."""
    return bundle["model"](normalize_for_classifier(x_pmf, bundle))


def classifier_target_prob(bundle, x_pmf, y_target):
    """Softmax probability of the target class index (no grad)."""
    with torch.no_grad():
        logits = classifier_logits(bundle, x_pmf)
        probs = torch.softmax(logits, dim=1)
        return probs.gather(1, y_target.view(-1, 1)).squeeze(1)
