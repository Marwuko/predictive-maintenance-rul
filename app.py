import gradio as gr
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import spaces

@spaces.GPU
def gpu_placeholder():
    return "ok"

model = joblib.load("rul_model.joblib")
FEATURES = joblib.load("features.joblib")
df = pd.read_parquet("val_data.parquet")

ALERT = 20

def show_engine(engine_id, current_cycle):
    e = df[df.engine_id == engine_id].sort_values("cycle")
    e = e[e.cycle <= current_cycle]
    if len(e) < 2:
        return "Move the cycle slider to start", None

    preds = model.predict(e[FEATURES])
    rul_now = preds[-1]

    status = "🔴 MAINTENANCE REQUIRED" if rul_now < ALERT else "🟢 Healthy"
    summary = (f"## Engine {int(engine_id)} at cycle {int(current_cycle)}\n"
               f"Predicted RUL: **{rul_now:.0f} cycles** | Status: {status}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e["cycle"], y=preds, name="predicted RUL", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=e["cycle"], y=e["RUL_clipped"], name="actual RUL",
                             line=dict(dash="dot"), opacity=0.6))
    fig.add_hline(y=ALERT, line_color="red", line_dash="dash",
                  annotation_text="maintenance alert")
    fig.update_layout(title="Remaining Useful Life over time",
                      xaxis_title="cycle", yaxis_title="RUL (cycles)", height=420)
    return summary, fig

with gr.Blocks(title="Predictive Maintenance: RUL") as demo:
    gr.Markdown("# 🔧 Predictive Maintenance — Remaining Useful Life")
    gr.Markdown("Trained on NASA C-MAPSS turbofan data. Pick an engine, advance its life, watch the failure prediction.")
    with gr.Row():
        engine = gr.Dropdown(choices=sorted(df.engine_id.unique().tolist()), value=85, label="Engine")
        cycle = gr.Slider(10, 350, value=100, step=5, label="Current cycle")
    summary = gr.Markdown()
    plot = gr.Plot()
    for ctrl in (engine, cycle):
        ctrl.change(show_engine, inputs=[engine, cycle], outputs=[summary, plot])
    demo.load(show_engine, inputs=[engine, cycle], outputs=[summary, plot])

demo.launch()
