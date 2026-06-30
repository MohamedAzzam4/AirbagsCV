"""
Gradio demo for the AirbagsCV anomaly-detection PoC.

This is a LOCAL proof-of-concept only. It is not a production system.

Usage:
    python demo/app.py
    # then open http://127.0.0.1:7860/ in your browser

The demo will gracefully report a clear error message if no trained checkpoint
is available, instead of crashing on import.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

# Allow `python demo/app.py` and `python -m demo.app` to both work.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import predict, resolve_run_dir, find_latest_checkpoint  # noqa: E402

logger = logging.getLogger("demo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
BENCHMARK_CSV = RESULTS_DIR / "benchmark_results.csv"

# Models actually trained in this repo. PatchCore was previously listed but is
# NOT implemented. Do not add it back until it is.
AVAILABLE_MODELS = ["EfficientAD"]
AVAILABLE_DATASETS = ["AITEX"]


# --------------------------------------------------------------------------- #
# Core processing                                                             #
# --------------------------------------------------------------------------- #
def _resize_to_match(src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize `src` (HxWxC or HxW) to target (H, W)."""
    th, tw = target_shape[:2]
    if src.ndim == 2:
        return cv2.resize(src, (tw, th), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(src, (tw, th), interpolation=cv2.INTER_LINEAR)


def process_image(image: np.ndarray, model_name: str, dataset_name: str):
    """Run inference and produce an overlay image + status string + score."""
    if image is None:
        return None, "Please upload an image first.", 0.0

    # Use a process-safe tempfile (no fixed filename, no race condition).
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
    try:
        Image.fromarray(image).save(temp_path)
        try:
            score, label, heatmap, mask = predict(
                temp_path, model_name, dataset_name, results_dir=RESULTS_DIR
            )
        except FileNotFoundError as e:
            return None, f"Model not trained: {e}", 0.0
        except ValueError as e:
            return None, f"Invalid input or model: {e}", 0.0
        except Exception as e:
            logger.exception("Inference failed")
            return None, f"Error: {type(e).__name__}: {e}", 0.0
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    # Build a coloured heatmap and overlay it on the ORIGINAL uploaded image.
    # The heatmap is always 256x256 (Anomalib pre-processor default); we resize
    # it up to the uploaded image size so they can be blended with cv2.addWeighted.
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = _resize_to_match(heatmap_color, image.shape[:2])

    alpha = 0.5
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    status = f"Status: {label}\nAnomaly score: {score:.4f}\n(0 = normal, higher = more anomalous)"
    return overlay, status, float(score)


def load_benchmarks():
    if not BENCHMARK_CSV.exists():
        return pd.DataFrame(
            {"Message": ["No benchmark_results.csv yet. Train a model first."]}
        )
    try:
        return pd.read_csv(BENCHMARK_CSV)
    except Exception as e:
        return pd.DataFrame({"Message": [f"Failed to read {BENCHMARK_CSV}: {e}"]})


def available_models_status():
    """Return a short string describing what models are actually loadable."""
    lines = []
    for m in AVAILABLE_MODELS:
        for d in AVAILABLE_DATASETS:
            run_dir = resolve_run_dir(m.lower(), d.lower(), RESULTS_DIR)
            ckpt = find_latest_checkpoint(run_dir)
            if ckpt is not None:
                lines.append(f"- {m} on {d}: READY  ({ckpt.name})")
            else:
                lines.append(
                    f"- {m} on {d}: NOT TRAINED  "
                    f"(expected checkpoint under {run_dir})"
                )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# UI                                                                          #
# --------------------------------------------------------------------------- #
CSS = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif; }
.status-normal { color: #15803d; font-weight: bold; }
.status-anomalous { color: #b91c1c; font-weight: bold; }
.caveat { color: #6b7280; font-size: 0.85em; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as app:
    gr.Markdown("# Airbag Fabric Defect Detection — Local PoC")
    gr.Markdown(
        "<span class='caveat'>"
        "Proof-of-concept only. Trained on AITEX (a textile proxy), not real "
        "airbag fabric. AUROC is reported, not accuracy. Not a production system."
        "</span>"
    )

    with gr.Tabs():
        with gr.Tab("Live Inspection"):
            gr.Markdown(
                "Upload an image (any size). The model returns an anomaly heatmap "
                "overlay and a score in [0, 1]. Higher = more anomalous."
            )
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="Input image")
                    model_dd = gr.Dropdown(
                        AVAILABLE_MODELS, value=AVAILABLE_MODELS[0], label="Model"
                    )
                    dataset_dd = gr.Dropdown(
                        AVAILABLE_DATASETS, value=AVAILABLE_DATASETS[0], label="Trained on"
                    )
                    btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    img_output = gr.Image(label="Anomaly heatmap overlay")
                    status_output = gr.Textbox(label="Verdict", lines=3)
                    score_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.001,
                        label="Anomaly score", interactive=False,
                    )
            btn.click(
                process_image,
                inputs=[img_input, model_dd, dataset_dd],
                outputs=[img_output, status_output, score_slider],
            )
            gr.Markdown("### Available models")
            gr.Textbox(value=available_models_status(), lines=4, interactive=False,
                       show_label=False)

        with gr.Tab("Benchmark Dashboard"):
            gr.Markdown(
                "Contents of `results/benchmark_results.csv`. Honest metrics only: "
                "image_AUROC, pixel_AUROC, eval_seconds (NOT inference latency). "
                "Use `scripts/benchmark_inference.py` for per-image latency."
            )
            df_out = gr.Dataframe(value=load_benchmarks)
            refresh = gr.Button("Refresh")
            refresh.click(load_benchmarks, outputs=[df_out])

        with gr.Tab("How It Works"):
            gr.Markdown(
                """
                ### Unsupervised anomaly detection
                The system trains ONLY on normal (defect-free) images. At inference
                time, anything that does not look "normal" produces a high anomaly
                score.

                **EfficientAD** (the only model currently trained in this repo) is a
                student-teacher network:
                - A pre-trained *teacher* network extracts features.
                - A *student* network learns to mimic the teacher on normal data.
                - On defective regions the student diverges from the teacher ->
                  anomaly signal.
                - A small autoencoder adds a second anomaly signal (reconstruction
                  error).

                ### Why this matters for airbags
                Real defect samples are rare and the factory has not shared data.
                Unsupervised methods need only normal samples, which the factory
                can capture in minutes of normal production.

                ### What is NOT in this demo
                - PatchCore is referenced in the research notes but is NOT trained
                  in this repo.
                - No real airbag data was used.
                - The latency reported here is not industrial line-scan throughput.
                - No threshold calibration against real defect rates has been done.
                """
            )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
