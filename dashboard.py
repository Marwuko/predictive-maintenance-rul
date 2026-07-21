import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Predictive Maintenance: RUL", layout="wide")
st.title("🔧 Predictive Maintenance — Remaining Useful Life")
st.caption("Trained on NASA C-MAPSS turbofan data. Pick an engine, advance its life, watch the failure prediction.")

@st.cache_resource
def load():
    model = joblib.load("rul_model.joblib")
    features = joblib.load("features.joblib")
    df = pd.read_parquet("val_data.parquet")
    return model, features, df

model, FEATURES, df = load()
ALERT = 20

col1, col2 = st.columns([1, 2])
with col1:
    engine_id = st.selectbox("Engine", sorted(df.engine_id.unique()))
with col2:
    max_c = int(df[df.engine_id == engine_id].cycle.max())
    current_cycle = st.slider("Current cycle", 10, max_c, min(100, max_c), step=5)

e = df[df.engine_id == engine_id].sort_values("cycle")
e = e[e.cycle <= current_cycle]

if len(e) >= 2:
    preds = model.predict(e[FEATURES])
    rul_now = preds[-1]

    if rul_now < ALERT:
        st.error(f"🔴 MAINTENANCE REQUIRED — predicted RUL: {rul_now:.0f} cycles")
    else:
        st.success(f"🟢 Healthy — predicted RUL: {rul_now:.0f} cycles")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e["cycle"], y=preds, name="predicted RUL", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=e["cycle"], y=e["RUL_clipped"], name="actual RUL",
                             line=dict(dash="dot"), opacity=0.6))
    fig.add_hline(y=ALERT, line_color="red", line_dash="dash", annotation_text="maintenance alert")
    fig.update_layout(title=f"Engine {engine_id} — RUL over time",
                      xaxis_title="cycle", yaxis_title="RUL (cycles)", height=450)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Move the cycle slider to start")
