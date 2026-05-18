import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, models
from collections import Counter

MODEL_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
    "efficientnet_b3": models.efficientnet_b3,
    "convnext_tiny": models.convnext_tiny,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "vit_b_16": models.vit_b_16,
}



def try_load_model(model_path, device, num_classes):
    """
    Load a classifier checkpoint and rebuild the correct model architecture.
    """
    
    checkpoint = torch.load(model_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary with model metadata and weights.")

    
    model_name = checkpoint.get("model_name")
    state_dict = checkpoint.get("model_state")
    class_names = checkpoint.get("class_names")

    if model_name is None:
        raise ValueError("Checkpoint missing 'model_name'.")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'model_state'.")
    
    saved_num_classes = len(class_names) if class_names else num_classes

    if saved_num_classes != num_classes:
        raise ValueError(
            f"Checkpoint expects {saved_num_classes} classes, but received {num_classes}."
        )
    
    model = build_model_architecture(model_name, saved_num_classes)
    model.load_state_dict(state_dict)
    
    model.to(device).eval()
    print(f"[camera_model] loaded {model_name} with {saved_num_classes} classes")
    return model

def set_classifier_head(model_name, model, num_classes):
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "convnext_tiny":
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    return model

def build_model_architecture(model_name, num_classes):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_name in checkpoint: {model_name}")
    
    model = MODEL_BUILDERS[model_name](weights = None)
    model = set_classifier_head(model_name, model, num_classes)
    return model

def build_preprocess():
    """
    Build preprocessing pipeline for the camera classifier.
    """
    ops = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
    ]
    return transforms.Compose(ops)


def get_video_duration_ms(cap):
    """
    Estimate video duration from OpenCV frame count and FPS.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    try:
        return int((n / fps) * 1000)
    except Exception:
        return 0


def read_frame_at_ms(cap, t_ms):
    """
    Read a single frame at an approximate timestamp in milliseconds.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = max(0, int((t_ms / 1000.0) * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    return frame_bgr if ok else None

def sample_frames_for_shot_with_times(cap, a_ms, b_ms, num_samples=7):
    """
    Sample multiple frames evenly across a shot.
    Returns a list of (timestamp_ms, frame_bgr).
    """
    a_ms = int(a_ms)
    b_ms = int(b_ms)

    if b_ms <= a_ms:
        return []

    if num_samples <= 1:
        mid = (a_ms + b_ms) // 2
        fr = read_frame_at_ms(cap, mid)
        return [(mid, fr)] if fr is not None else []

    out = []
    span = b_ms - a_ms

    # Sample inside the segment, not exactly at the edges.
    for i in range(num_samples):
        frac = (i + 1) / (num_samples + 1)
        t_ms = a_ms + int(span * frac)
        fr = read_frame_at_ms(cap, t_ms)
        if fr is not None:
            out.append((t_ms, fr))

    return out


def majority_label(labels, default="main_camera_center"):
    """
    Return the most common label from a list.
    """
    if not labels:
        return default
    return Counter(labels).most_common(1)[0][0]



@torch.no_grad()
def classify_frames_batch(model, frames_bgr, device, preprocess, labels):
    """
    Classify a batch of BGR frames.
    Return list of (label, confidence).
    """
    xs = []
    for frame_bgr in frames_bgr:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        xs.append(preprocess(frame_rgb))
    x = torch.stack(xs, dim=0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    conf, idx = torch.max(probs, dim=1)
    return [(labels[int(i)], float(c)) for i, c in zip(idx, conf)]


@torch.no_grad()
def classify_shots(cap, model, device, preprocess, labels, shots, batch_size=64):
    """
    Classify each shot by sampling a representative frame near its midpoint.
    """
    frames = []
    idx_map = []
    for i, (a, b) in enumerate(shots):
        a = int(a)
        b = int(b)
        mid = (a + b) // 2
        fr = read_frame_at_ms(cap, mid)
        if fr is None:
            fr = read_frame_at_ms(cap, a)
        if fr is not None:
            idx_map.append(i)
            frames.append(fr)

    out_labels = ["main_camera_center"] * len(shots)
    if not frames:
        return out_labels

    start = 0
    while start < len(frames):
        chunk = frames[start:start + batch_size]
        preds = classify_frames_batch(model, chunk, device, preprocess, labels)
        for j, (lbl, _conf) in enumerate(preds):
            out_labels[idx_map[start + j]] = lbl
        start += batch_size

    return out_labels
