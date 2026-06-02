import gradio as gr
import cv2
import numpy as np
import pandas as pd
from inference import predict
import os
from PIL import Image

def process_image(image, model_name, dataset_name):
    if image is None:
        return None, "Please upload an image.", 0.0
        
    # Save temporarily for prediction
    temp_path = "temp_predict.png"
    Image.fromarray(image).save(temp_path)
    
    try:
        score, label, heatmap, mask = predict(temp_path, model_name, dataset_name)
    except FileNotFoundError as e:
        return None, f"Model not trained yet: {str(e)}", 0.0
    except Exception as e:
        return None, f"Error: {str(e)}", 0.0
        
    # Create colormap heatmap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    
    status_text = f"Status: {label}\nAnomaly Score: {score:.4f}"
    
    return overlay, status_text, score

def load_benchmarks():
    try:
        df = pd.read_csv("../results/benchmark_results.csv")
        return df
    except:
        return pd.DataFrame({"Message": ["Benchmarks not found. Train models first."]})

css = """
.gradio-container { font-family: 'Inter', sans-serif; }
.status-normal { color: green; font-weight: bold; }
.status-anomalous { color: red; font-weight: bold; }
"""

with gr.Blocks() as app:
    gr.Markdown("# 🛡️ Airbag Defect Detection AI Prototype")
    
    with gr.Tabs():
        with gr.Tab("🔍 Live Inspection"):
            gr.Markdown("Upload an image of an airbag fabric or proxy textile to detect defects.")
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="Input Image")
                    model_dropdown = gr.Dropdown(["EfficientAD"], value="EfficientAD", label="Model")
                    dataset_dropdown = gr.Dropdown(["AITEX"], value="AITEX", label="Trained On")
                    btn = gr.Button("Analyze", variant="primary")
                    
                with gr.Column():
                    img_output = gr.Image(label="Anomaly Heatmap Overlay")
                    status_output = gr.Textbox(label="Verdict")
                    score_bar = gr.Slider(minimum=0, maximum=1, label="Anomaly Confidence")
                    
            btn.click(process_image, inputs=[img_input, model_dropdown, dataset_dropdown], outputs=[img_output, status_output, score_bar])
            
        with gr.Tab("📊 Benchmark Dashboard"):
            gr.Markdown("Compare model performance across datasets.")
            df_output = gr.Dataframe(value=load_benchmarks)
            refresh_btn = gr.Button("Refresh Data")
            refresh_btn.click(load_benchmarks, outputs=[df_output])
            
        with gr.Tab("ℹ️ How It Works"):
            gr.Markdown("""
            ### Unsupervised Anomaly Detection
            This system uses 1-class learning. It only needs **normal** images to train.
            - **PatchCore**: Extracts features using a ResNet backbone and stores them in a memory bank. Compares new images using K-Nearest Neighbors.
            - **EfficientAD**: Uses a student-teacher network. The student learns to mimic the teacher on normal data. On defective data, their outputs diverge.
            
            **Why this matters for airbags:**
            You don't need to provide thousands of defective samples. Just a few minutes of normal production is enough to train a robust model.
            """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, css=css, theme=gr.themes.Soft())
