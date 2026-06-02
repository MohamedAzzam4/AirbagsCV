import torch
from anomalib.engine import Engine
from anomalib.models import Patchcore, EfficientAd
from PIL import Image
import numpy as np
import glob
import os

_MODEL_CACHE = {}

def get_model(model_name, dataset_name):
    key = f"{model_name.lower()}_{dataset_name.lower()}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
        
    base_dir = f"./results/{model_name.lower()}_{dataset_name.lower()}"
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Model directory not found: {base_dir}. Please train it first.")
        
    ckpt_paths = glob.glob(f"{base_dir}/**/*.ckpt", recursive=True)
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint found in {base_dir}.")
    
    ckpt_path = ckpt_paths[-1]
    
    if model_name.lower() == "patchcore":
        model = Patchcore.load_from_checkpoint(ckpt_path)
    elif model_name.lower() == "efficientad":
        model = EfficientAd.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    model.eval()
    engine = Engine(accelerator="auto")
    
    _MODEL_CACHE[key] = (engine, model)
    return engine, model

def predict(image_path, model_name, dataset_name):
    engine, model = get_model(model_name, dataset_name)
    
    predictions = engine.predict(model=model, data_path=image_path)
    batch = predictions[0]
    
    score = batch.pred_score[0].item()
    label = "ANOMALOUS" if batch.pred_label[0].item() else "NORMAL"
    
    heatmap_tensor = batch.anomaly_map[0].squeeze().cpu().numpy()
    heatmap_normalized = ((heatmap_tensor - heatmap_tensor.min()) / (heatmap_tensor.max() - heatmap_tensor.min() + 1e-8) * 255).astype(np.uint8)
    
    mask_tensor = batch.pred_mask[0].squeeze().cpu().numpy()
    mask = (mask_tensor * 255).astype(np.uint8)
    
    return score, label, heatmap_normalized, mask
