from anomalib.data import MVTecAD, Folder
from anomalib.models import Patchcore, EfficientAd
from anomalib.engine import Engine
from anomalib import TaskType
import os
import time
import pandas as pd



def train_efficientad(datamodule, dataset_name):
    print(f"Training EfficientAD on {dataset_name}...")
    model = EfficientAd(
        model_size="small",
        teacher_out_channels=384,
        lr=0.0001,
        weight_decay=0.00001,
        padding=False,
        pad_maps=True,
    )
    engine = Engine(
        max_epochs=10,
        accelerator="auto",
        default_root_dir=f"./results/efficientad_{dataset_name}"
    )
    engine.fit(datamodule=datamodule, model=model)
    
    start_time = time.time()
    test_results = engine.test(model=model, datamodule=datamodule)[0]
    latency = (time.time() - start_time) / len(datamodule.test_dataloader().dataset) * 1000
    
    return {
        'Model': 'EfficientAD',
        'Dataset': dataset_name,
        'Image AUROC': test_results.get('image_AUROC', test_results.get('test_image_AUROC', None)),
        'Pixel AUROC': test_results.get('pixel_AUROC', test_results.get('test_pixel_AUROC', None)),
        'Latency (ms)': latency
    }

def main():
    os.makedirs("./results", exist_ok=True)
    results = []
    
    # 1. AITEX
    if os.path.exists("./datasets/aitex"):
        aitex_data = Folder(
            name="aitex",
            root="./datasets/aitex",
            normal_dir="train/good",
            abnormal_dir="test/anomaly",
            normal_test_dir="test/good",
            mask_dir="ground_truth/anomaly",
            train_batch_size=1,
            eval_batch_size=32,
            num_workers=0,
        )
        results.append(train_efficientad(aitex_data, "aitex"))
    else:
        print("AITEX dataset not found. Skipping...")
        
    df = pd.DataFrame(results)
    df.to_csv("./results/benchmark_results.csv", index=False)
    print("\nBenchmark Results:")
    print(df.to_string())

if __name__ == '__main__':
    main()
