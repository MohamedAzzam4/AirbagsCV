from anomalib.engine import Engine
from anomalib.models import Patchcore, EfficientAd
from anomalib.deploy import ExportType
import glob
import os

def export_model(model_name, dataset_name):
    base_dir = f"./results/{model_name.lower()}_{dataset_name.lower()}"
    if not os.path.exists(base_dir):
        print(f"Skipping export for {model_name} on {dataset_name} (not found).")
        return
        
    ckpt_paths = glob.glob(f"{base_dir}/**/*.ckpt", recursive=True)
    if not ckpt_paths:
        print(f"No checkpoint found in {base_dir}.")
        return
        
    ckpt_path = ckpt_paths[-1]
    
    if model_name.lower() == "patchcore":
        model = Patchcore.load_from_checkpoint(ckpt_path)
    elif model_name.lower() == "efficientad":
        model = EfficientAd.load_from_checkpoint(ckpt_path)
    
    engine = Engine(accelerator="auto")
    
    # Export to OpenVINO
    export_root = f"./exports/openvino/{model_name.lower()}_{dataset_name.lower()}"
    os.makedirs(export_root, exist_ok=True)
    try:
        engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root
        )
        print(f"Exported {model_name} on {dataset_name} to OpenVINO format.")
    except Exception as e:
        print(f"Failed to export OpenVINO: {e}")
        
    # Export to ONNX
    export_root_onnx = f"./exports/onnx/{model_name.lower()}_{dataset_name.lower()}"
    os.makedirs(export_root_onnx, exist_ok=True)
    try:
        engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=export_root_onnx
        )
        print(f"Exported {model_name} on {dataset_name} to ONNX format.")
    except Exception as e:
        print(f"Failed to export ONNX: {e}")

def main():
    datasets = ["aitex"]
    models = ["PatchCore", "EfficientAD"]
    
    for dataset in datasets:
        for model in models:
            export_model(model, dataset)

if __name__ == '__main__':
    main()
