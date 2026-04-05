import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from data_prep import prepare_data
from models import build_arima, build_prophet, build_lstm

# -------------------------------------------------------------
# Configuration & Aesthetics
# -------------------------------------------------------------
st.set_page_config(
    page_title="Dublin Unemployment Forecaster", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 300;
        color: #888;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(78, 205, 196, 0.2);
    }
    
    .report-box {
        background: linear-gradient(135deg, rgba(29, 38, 113, 0.8) 0%, rgba(195, 55, 100, 0.8) 100%);
        border-radius: 20px;
        padding: 30px;
        color: white;
        margin-top: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    if not os.path.exists("cleaned_unemployment.csv"):
        if os.path.exists("dataset.csv"):
            df = prepare_data("dataset.csv", "cleaned_unemployment.csv")
        else:
            st.error("No dataset.csv found. Please ensure the dataset is in the working directory.")
            return pd.DataFrame()
    else:
        df = pd.read_csv("cleaned_unemployment.csv")
        df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_and_prep_data()

# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3206/3206015.png", width=100)
    st.markdown("### Model Configuration")
    steps_ahead = st.slider("Forecast Horizon (Quarters)", min_value=1, max_value=8, value=4, step=1)
    target_metric = st.selectbox(
        "Target Variable for Forecast",
        ["Dublin_Unemployment_Rate_Pct", "Dublin_Unemployed_Thousands", "National_Unemployment_Rate_Pct"]
    )
    
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown("🐍 Python\n\n🐼 Pandas\n\n🔮 Prophet & ARIMA\n\n🧠 TensorFlow (LSTM)\n\n📊 Plotly\n\n✨ Streamlit")

# -------------------------------------------------------------
# Main Content
# -------------------------------------------------------------
st.markdown('<div class="main-header">Dublin Unemployment Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced predictive modelling for Ireland\'s capital</div>', unsafe_allow_html=True)

if df.empty:
    st.stop()

# -------------------------------------------------------------
# 1. Exploratory Data Analysis (EDA)
# -------------------------------------------------------------
st.markdown("### 📊 Historical Trends (EDA)")

# High-level Metrics
latest_date = df['Date'].max()
latest_data = df[df['Date'] == latest_date].iloc[0]
prev_data = df.iloc[-2] if len(df) > 1 else latest_data

col1, col2, col3, col4 = st.columns(4)
with col1:
    delta = latest_data['Dublin_Unemployment_Rate_Pct'] - prev_data['Dublin_Unemployment_Rate_Pct']
    st.metric("Dublin Unemployment Rate", f"{latest_data['Dublin_Unemployment_Rate_Pct']}%", f"{delta:.2f}% YoY/QoQ", delta_color="inverse")
with col2:
    delta2 = latest_data['National_Unemployment_Rate_Pct'] - prev_data['National_Unemployment_Rate_Pct']
    st.metric("National Unemployment Rate", f"{latest_data['National_Unemployment_Rate_Pct']}%", f"{delta2:.2f}% YoY/QoQ", delta_color="inverse")
with col3:
    delta3 = latest_data['Dublin_Unemployed_Thousands'] - prev_data['Dublin_Unemployed_Thousands']
    st.metric("Dublin Unemployed", f"{latest_data['Dublin_Unemployed_Thousands']}k", f"{delta3:.1f}k YoY/QoQ", delta_color="inverse")
with col4:
    delta4 = latest_data['Dublin_Employed_Thousands'] - prev_data['Dublin_Employed_Thousands']
    st.metric("Dublin Employed", f"{latest_data['Dublin_Employed_Thousands']}k", f"{delta4:.1f}k YoY/QoQ")

st.markdown("<br>", unsafe_allow_html=True)

# Comparisons Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Dublin_Unemployment_Rate_Pct'], mode='lines', name='Dublin Rate (%)', line=dict(color='#FF6B6B', width=3)))
fig.add_trace(go.Scatter(x=df['Date'], y=df['National_Unemployment_Rate_Pct'], mode='lines', name='National Rate (%)', line=dict(color='#4ECDC4', width=3)))

fig.update_layout(
    title='Dublin vs National Unemployment Rate Over Time',
    xaxis_title='Date',
    yaxis_title='Unemployment Rate (%)',
    template='plotly_dark',
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 2. Forecasting
# -------------------------------------------------------------
st.markdown("---")
st.markdown(f"### 🤖 AI Forecasting Models (`{target_metric}`)")

series = df[target_metric]

if st.button("🚀 Train Models & Generate Forecast"):
    with st.spinner("Training ARIMA, Prophet, and LSTM models..."):
        # ARIMA
        start_time = time.time()
        arima_fc, arima_insample, arima_rmse = build_arima(series, steps=steps_ahead)
        arima_time = time.time() - start_time
        
        # Prophet
        start_time = time.time()
        prophet_fc, prophet_insample, prophet_rmse, future_dates = build_prophet(df, 'Date', target_metric, steps=steps_ahead)
        prophet_time = time.time() - start_time
        
        # LSTM
        start_time = time.time()
        lstm_fc, lstm_insample, lstm_rmse = build_lstm(series, steps=steps_ahead, epochs=100)
        lstm_time = time.time() - start_time
        
        st.success("All models trained successfully!")
        
        # Determine the best model
        metrics = {
            "ARIMA": {"RMSE": arima_rmse, "Time (s)": arima_time},
            "Prophet": {"RMSE": prophet_rmse, "Time (s)": prophet_time},
            "LSTM": {"RMSE": lstm_rmse, "Time (s)": lstm_time}
        }
        
        best_model = min(metrics.keys(), key=lambda k: metrics[k]["RMSE"])
        
        # Compare Performances
        metric_df = pd.DataFrame(metrics).T
        st.markdown(f"**Best Model**: `{best_model}` (Lowest RMSE)")
        st.dataframe(metric_df.style.highlight_min(subset=["RMSE"], color="#FF6B6B"))

        # Visualizing Forecasts
        st.markdown("#### Forecast Comparison vs Actuals")
        
        # Create future dates manually for ARIMA/LSTM
        last_date = df['Date'].max()
        future_dt = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=steps_ahead, freq='QS')
        
        fig2 = go.Figure()
        # Actual Data
        fig2.add_trace(go.Scatter(x=df['Date'], y=series, mode='lines', name='Actual Data', line=dict(color='white', width=2)))
        
        # Best model's forecast
        if best_model == "ARIMA":
            best_fc = arima_fc
        elif best_model == "Prophet":
            best_fc = prophet_fc
        else:
            best_fc = lstm_fc
            
        fig2.add_trace(go.Scatter(x=future_dt, y=best_fc, mode='lines+markers', name=f'{best_model} Forecast (Best)', line=dict(color='#FFE66D', width=4, dash='dot')))
        
        # Other models just for reference
        fig2.add_trace(go.Scatter(x=future_dt, y=arima_fc if best_model != "ARIMA" else prophet_fc, mode='lines', name='Alt Model', line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dot')))

        fig2.update_layout(
            title=f'Next {steps_ahead} Quarters Forecast',
            xaxis_title='Date',
            yaxis_title=target_metric,
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig2, use_container_width=True)

        # -------------------------------------------------------------
        # 3. Business Report
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown(f"### 💼 Business Report for Dublin Employers")
        
        # Dynamic advice based on trend
        start_val = series.iloc[-1]
        end_val = best_fc[-1]
        trend_diff = end_val - start_val
        
        if trend_diff < -0.5:
            trend_text = "significantly tight labor market"
            advice = "Employers should anticipate heightened competition for talent. A shrinking unemployment rate typically translates to upward wage pressures. Prioritize retention strategies, internal upskilling, and consider expanding recruitment channels."
        elif trend_diff > 0.5:
            trend_text = "loosening labor market"
            advice = "Employers might experience a larger pool of available candidates. It may become easier to fill roles, but this could also indicate broader macroeconomic cooling affecting consumer demand. Focus on optimizing workforce efficiency and strategic hiring."
        else:
            trend_text = "relatively stable labor market"
            advice = "Conditions are expected to remain steady. Employers should maintain their standard hiring rhythms and focus on sustainable, long-term talent acquisition without the need for aggressive compensation inflation."

        report_html = f"""
        <div class="report-box">
            <h2 style="margin-top:0; color: white;">What should Dublin employers expect? (Next {steps_ahead} Quarters)</h2>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                Based on our <strong>{best_model}</strong> AI forecasting model, the <code>{target_metric}</code> 
                is projected to move from <strong>{start_val:.1f}</strong> to approximately <strong>{end_val:.1f}</strong> over the forecast horizon.
            </p>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                This indicates a <strong>{trend_text}</strong> in the medium term.
            </p>
            <hr style="border-top: 1px solid rgba(255,255,255,0.3);">
            <h3 style="color: #FFE66D;">Strategic Recommendation:</h3>
            <p style="font-size: 1.2rem; font-weight: 300;">
                {advice}
            </p>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)
else:
    st.info("👆 Click the button above to train models and generate insights.")
