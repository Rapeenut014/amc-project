"""
Streamlit Dashboard for Automatic Modulation Classification
Interactive visualization and real-time prediction demo
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="AMC Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.5);
    }
    
    .prediction-bar {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 8px;
        height: 24px;
        transition: all 0.3s ease;
    }
    
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    
    h1 {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .stSelectbox label, .stSlider label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    
    .model-status {
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-top: 8px;
    }
    
    .model-loaded {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid rgba(34, 197, 94, 0.5);
        color: #22c55e;
    }
    
    .model-sample {
        background: rgba(234, 179, 8, 0.2);
        border: 1px solid rgba(234, 179, 8, 0.5);
        color: #eab308;
    }
</style>
""", unsafe_allow_html=True)


# ============ Helper Functions ============

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def list_trained_models():
    """List all trained models in the models directory"""
    models_dir = get_project_root() / "models"
    if not models_dir.exists():
        return []
    
    models = list(models_dir.glob("*.keras"))
    return sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)


def list_history_files():
    """List all training history files"""
    models_dir = get_project_root() / "models"
    if not models_dir.exists():
        return []
    
    histories = list(models_dir.glob("history_*.json"))
    return sorted(histories, key=lambda x: x.stat().st_mtime, reverse=True)


@st.cache_data
def load_sample_data():
    """Load sample/cached data for demo when no real data available"""
    # Sample classes
    classes = [
        'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK',
        '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
        '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
        'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
    ]
    
    # Sample SNR accuracy (simulated realistic values)
    snr_range = list(range(-20, 32, 2))
    snr_accuracy = {}
    for snr in snr_range:
        # Realistic S-curve: low at low SNR, high at high SNR
        base_acc = 1 / (1 + np.exp(-(snr + 5) / 5))
        noise = np.random.uniform(-0.03, 0.03)
        snr_accuracy[snr] = min(0.98, max(0.04, base_acc + noise))
    
    # Sample confusion matrix (realistic diagonal-dominant pattern)
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                cm[i, j] = np.random.uniform(0.5, 0.9)
            else:
                cm[i, j] = np.random.uniform(0, 0.1)
    # Normalize rows
    cm = cm / cm.sum(axis=1, keepdims=True)
    
    # Sample training history
    epochs = list(range(1, 51))
    train_loss = [2.5 * np.exp(-e/15) + 0.3 + np.random.uniform(-0.05, 0.05) for e in epochs]
    val_loss = [2.5 * np.exp(-e/15) + 0.4 + np.random.uniform(-0.08, 0.08) for e in epochs]
    train_acc = [1 - 0.95 * np.exp(-e/12) + np.random.uniform(-0.02, 0.02) for e in epochs]
    val_acc = [1 - 0.98 * np.exp(-e/12) + np.random.uniform(-0.03, 0.03) for e in epochs]
    
    history = {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    
    return classes, snr_accuracy, cm, history


def load_real_results(results_dir=None):
    """Try to load real evaluation results"""
    if results_dir is None:
        results_dir = get_project_root() / "results"
    else:
        results_dir = Path(results_dir)
    
    report_path = results_dir / 'evaluation_report.json'
    if report_path.exists():
        with open(report_path, 'r') as f:
            return json.load(f)
    return None


def load_training_history(history_path):
    """Load training history from JSON file"""
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    # Convert to expected format
    history = {
        'epochs': list(range(1, len(data['loss']) + 1)),
        'train_loss': data['loss'],
        'val_loss': data['val_loss'],
        'train_acc': data['accuracy'],
        'val_acc': data['val_accuracy']
    }
    return history, data.get('classes', None)


def generate_sample_signal(modulation_type, snr=10, n_samples=1024):
    """Generate sample I/Q signal for demo"""
    t = np.linspace(0, 1, n_samples)
    
    # Base signal based on modulation type
    if 'PSK' in modulation_type or 'QPSK' in modulation_type:
        phase = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], n_samples)
        I = np.cos(2 * np.pi * 10 * t + phase)
        Q = np.sin(2 * np.pi * 10 * t + phase)
    elif 'QAM' in modulation_type:
        I = np.random.choice([-3, -1, 1, 3], n_samples) / 3
        Q = np.random.choice([-3, -1, 1, 3], n_samples) / 3
    elif 'FM' in modulation_type:
        mod_signal = np.sin(2 * np.pi * 1 * t)
        phase = np.cumsum(mod_signal) / n_samples * 10
        I = np.cos(2 * np.pi * 10 * t + phase)
        Q = np.sin(2 * np.pi * 10 * t + phase)
    else:  # ASK, OOK, AM
        amplitude = 1 + 0.5 * np.sin(2 * np.pi * 1 * t)
        I = amplitude * np.cos(2 * np.pi * 10 * t)
        Q = amplitude * np.sin(2 * np.pi * 10 * t)
    
    # Add noise based on SNR
    noise_power = 10 ** (-snr / 10)
    I += np.random.normal(0, np.sqrt(noise_power), n_samples)
    Q += np.random.normal(0, np.sqrt(noise_power), n_samples)
    
    return np.stack([I, Q], axis=1)


def create_metric_card(value, label, prefix="", suffix=""):
    """Create styled metric card"""
    display_value = f"{prefix}{value}{suffix}"
    return f"""
    <div class="metric-card">
        <div class="metric-value">{display_value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# ============ Main Dashboard ============

def main():
    # Check for real data
    available_models = list_trained_models()
    available_histories = list_history_files()
    real_results = load_real_results()
    
    # Determine data source
    using_real_data = False
    selected_history = None
    
    # Load data
    classes, snr_accuracy, cm, history = load_sample_data()
    
    # If real history exists, use it
    if available_histories:
        # Load the most recent history by default
        history, loaded_classes = load_training_history(available_histories[0])
        if loaded_classes:
            classes = loaded_classes
        using_real_data = True
    
    # If real evaluation results exist, use them
    if real_results:
        if 'snr_accuracy' in real_results:
            snr_accuracy = {int(k): v for k, v in real_results['snr_accuracy'].items()}
        using_real_data = True
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/radio-tower.png", width=80)
        st.title("AMC Dashboard")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Overview", "📊 Performance", "🔥 Confusion Matrix", 
             "📈 SNR Analysis", "🎮 Live Demo"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model Selection
        st.markdown("### 🤖 Model Selection")
        if available_models:
            model_names = [m.name for m in available_models]
            selected_model = st.selectbox(
                "Choose trained model:",
                options=model_names,
                label_visibility="collapsed"
            )
            st.session_state['selected_model'] = available_models[model_names.index(selected_model)]
            st.markdown(
                '<div class="model-status model-loaded">✓ Model loaded</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No trained models found. Train a model first using `python src/train.py`")
            st.markdown(
                '<div class="model-status model-sample">⚠ Using sample data</div>',
                unsafe_allow_html=True
            )
        
        # History selection
        if len(available_histories) > 1:
            st.markdown("### 📜 Training History")
            history_names = [h.name for h in available_histories]
            selected_history_name = st.selectbox(
                "Choose history:",
                options=history_names,
                label_visibility="collapsed"
            )
            idx = history_names.index(selected_history_name)
            history, loaded_classes = load_training_history(available_histories[idx])
            if loaded_classes:
                classes = loaded_classes
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown(f"**Classes:** {len(classes)}")
        st.markdown(f"**Architecture:** CNN")
        st.markdown(f"**Input Shape:** (1024, 2)")
        
        st.markdown("---")
        
        # Data source indicator
        if using_real_data:
            st.success("📊 Using real training data")
        else:
            st.warning("📊 Using simulated demo data")
        
        st.markdown("##### Built with ❤️ using Streamlit")
    
    # Main content
    if page == "🏠 Overview":
        render_overview(classes, snr_accuracy, history)
    elif page == "📊 Performance":
        render_performance(history)
    elif page == "🔥 Confusion Matrix":
        render_confusion_matrix(cm, classes)
    elif page == "📈 SNR Analysis":
        render_snr_analysis(snr_accuracy)
    elif page == "🎮 Live Demo":
        render_live_demo(classes)


def render_overview(classes, snr_accuracy, history):
    """Overview page with key metrics"""
    st.title("📡 Automatic Modulation Classification")
    st.markdown("### AI-Powered Signal Recognition System")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    overall_acc = np.mean(list(snr_accuracy.values()))
    high_snr_acc = np.mean([v for k, v in snr_accuracy.items() if k >= 10])
    
    with col1:
        st.markdown(create_metric_card(f"{overall_acc*100:.1f}", "Overall Accuracy", suffix="%"), 
                    unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(f"{high_snr_acc*100:.1f}", "High SNR Acc.", suffix="%"), 
                    unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(len(classes), "Modulation Classes"), 
                    unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card(len(history['epochs']), "Training Epochs"), 
                    unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Training Progress")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['epochs'], y=history['train_acc'],
            name='Train', line=dict(color='#818cf8', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=history['epochs'], y=history['val_acc'],
            name='Validation', line=dict(color='#f472b6', width=2)
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 SNR vs Accuracy")
        snrs = sorted(snr_accuracy.keys())
        accs = [snr_accuracy[s] for s in snrs]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=snrs, y=accs,
            mode='lines+markers',
            line=dict(color='#818cf8', width=3),
            marker=dict(size=8, color='#c084fc')
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#f472b6", opacity=0.5)
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='SNR (dB)',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1]),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Modulation classes
    st.markdown("---")
    st.markdown("### 🎯 Supported Modulation Types")
    
    # Group classes by type
    groups = {
        'PSK': [c for c in classes if 'PSK' in c],
        'QAM': [c for c in classes if 'QAM' in c],
        'ASK': [c for c in classes if 'ASK' in c or 'OOK' in c],
        'AM/FM': [c for c in classes if 'AM' in c or 'FM' in c or 'GMSK' in c]
    }
    
    cols = st.columns(4)
    for i, (group_name, group_classes) in enumerate(groups.items()):
        with cols[i]:
            st.markdown(f"**{group_name}**")
            for c in group_classes:
                st.markdown(f"• {c}")


def render_performance(history):
    """Training performance page"""
    st.title("📊 Training Performance")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Loss Curves")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['epochs'], y=history['train_loss'],
            name='Train Loss', 
            line=dict(color='#818cf8', width=2),
            fill='tozeroy', fillcolor='rgba(129, 140, 248, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=history['epochs'], y=history['val_loss'],
            name='Val Loss', 
            line=dict(color='#f472b6', width=2),
            fill='tozeroy', fillcolor='rgba(244, 114, 182, 0.1)'
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Accuracy Curves")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['epochs'], y=history['train_acc'],
            name='Train Acc', 
            line=dict(color='#818cf8', width=2),
            fill='tozeroy', fillcolor='rgba(129, 140, 248, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=history['epochs'], y=history['val_acc'],
            name='Val Acc', 
            line=dict(color='#f472b6', width=2),
            fill='tozeroy', fillcolor='rgba(244, 114, 182, 0.1)'
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Training stats
    st.markdown("---")
    st.markdown("### 📋 Training Summary")
    
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Train Acc", f"{final_train_acc*100:.2f}%")
    col2.metric("Final Val Acc", f"{final_val_acc*100:.2f}%")
    col3.metric("Best Val Acc", f"{best_val_acc*100:.2f}%")
    col4.metric("Best Epoch", best_epoch)


def render_confusion_matrix(cm, classes):
    """Confusion matrix page"""
    st.title("🔥 Confusion Matrix")
    st.markdown("---")
    
    # Normalization toggle
    normalize = st.checkbox("Normalize by row", value=True)
    
    if normalize:
        cm_display = cm
    else:
        # Scale for display
        cm_display = (cm * 100).astype(int)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=classes,
        y=classes,
        colorscale='Viridis',
        text=np.round(cm_display, 2),
        texttemplate='%{text:.2f}' if normalize else '%{text}',
        textfont={"size": 8},
        hoverongaps=False
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Predicted',
        yaxis_title='True',
        height=700,
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange='reversed')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Class-wise accuracy
    st.markdown("---")
    st.markdown("### Per-Class Accuracy")
    
    class_acc = np.diag(cm)
    df = pd.DataFrame({
        'Class': classes,
        'Accuracy': class_acc
    }).sort_values('Accuracy', ascending=False)
    
    fig = px.bar(
        df, x='Accuracy', y='Class',
        orientation='h',
        color='Accuracy',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        yaxis=dict(categoryorder='total ascending')
    )
    st.plotly_chart(fig, use_container_width=True)


def render_snr_analysis(snr_accuracy):
    """SNR analysis page"""
    st.title("📈 SNR Analysis")
    st.markdown("---")
    
    snrs = sorted(snr_accuracy.keys())
    accs = [snr_accuracy[s] for s in snrs]
    
    # Main chart
    fig = go.Figure()
    
    # Color gradient based on accuracy
    colors = ['#ef4444' if a < 0.3 else '#f59e0b' if a < 0.6 else '#22c55e' for a in accs]
    
    fig.add_trace(go.Bar(
        x=snrs, y=accs,
        marker_color=colors,
        text=[f'{a*100:.1f}%' for a in accs],
        textposition='outside'
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="#f472b6", 
                  annotation_text="50% Baseline")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='SNR (dB)',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1.1]),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats table
    st.markdown("---")
    st.markdown("### SNR Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    low_snr = np.mean([v for k, v in snr_accuracy.items() if k < 0])
    mid_snr = np.mean([v for k, v in snr_accuracy.items() if 0 <= k < 15])
    high_snr = np.mean([v for k, v in snr_accuracy.items() if k >= 15])
    
    col1.metric("Low SNR (<0 dB)", f"{low_snr*100:.1f}%", 
                delta=f"{(low_snr-0.5)*100:.1f}% vs baseline")
    col2.metric("Mid SNR (0-15 dB)", f"{mid_snr*100:.1f}%",
                delta=f"{(mid_snr-0.5)*100:.1f}% vs baseline")
    col3.metric("High SNR (≥15 dB)", f"{high_snr*100:.1f}%",
                delta=f"{(high_snr-0.5)*100:.1f}% vs baseline")


def render_live_demo(classes):
    """Live prediction demo page"""
    st.title("🎮 Live Demo")
    st.markdown("Generate sample signals and see model predictions!")
    st.markdown("---")
    
    # Check for loaded model
    model = None
    if 'selected_model' in st.session_state:
        try:
            import tensorflow as tf
            model_path = st.session_state['selected_model']
            if 'loaded_model' not in st.session_state or st.session_state.get('loaded_model_path') != model_path:
                with st.spinner("Loading model..."):
                    st.session_state['loaded_model'] = tf.keras.models.load_model(model_path)
                    st.session_state['loaded_model_path'] = model_path
            model = st.session_state['loaded_model']
        except Exception as e:
            st.error(f"Could not load model: {e}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Signal Parameters")
        
        selected_mod = st.selectbox("Modulation Type", classes)
        snr = st.slider("SNR (dB)", -20, 30, 10)
        
        if st.button("🎲 Generate Signal", use_container_width=True):
            st.session_state.demo_signal = generate_sample_signal(selected_mod, snr)
            st.session_state.demo_true_class = selected_mod
        
        if model is not None:
            st.success("✓ Using real model predictions")
        else:
            st.info("ℹ️ Using simulated predictions (no model loaded)")
    
    with col2:
        if 'demo_signal' in st.session_state:
            signal = st.session_state.demo_signal
            
            st.markdown("### Generated I/Q Signal")
            
            # Plot I/Q
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=('I Component', 'Q Component'))
            
            time = np.arange(len(signal))
            
            fig.add_trace(
                go.Scatter(y=signal[:, 0], mode='lines', 
                          line=dict(color='#818cf8', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=signal[:, 1], mode='lines',
                          line=dict(color='#f472b6', width=1)),
                row=2, col=1
            )
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction Results
            st.markdown("### Prediction Results")
            
            if model is not None:
                # Real prediction
                signal_input = signal.reshape(1, 1024, 2).astype(np.float32)
                # Normalize
                mean = np.mean(signal_input, axis=1, keepdims=True)
                std = np.std(signal_input, axis=1, keepdims=True) + 1e-8
                signal_input = (signal_input - mean) / std
                
                probs = model.predict(signal_input, verbose=0)[0]
            else:
                # Simulated prediction (weighted towards true class)
                probs = np.random.dirichlet(np.ones(len(classes)) * 0.5)
                true_idx = classes.index(st.session_state.demo_true_class)
                probs[true_idx] += 0.4  # Bias towards true class
                probs = probs / probs.sum()
            
            top_k = 5
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            for idx in top_indices:
                prob = probs[idx]
                class_name = classes[idx]
                is_true = class_name == st.session_state.demo_true_class
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    color = '#22c55e' if is_true else '#6366f1'
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; margin-bottom:8px;">
                        <div style="flex:1; background:rgba(99,102,241,0.2); 
                                    border-radius:8px; height:24px; margin-right:10px;">
                            <div style="width:{prob*100:.1f}%; background:{color}; 
                                        height:100%; border-radius:8px;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    emoji = "✅" if is_true else ""
                    st.markdown(f"**{class_name}** {emoji}")
                    st.caption(f"{prob*100:.1f}%")
        else:
            st.info("👆 Click 'Generate Signal' to start the demo")


if __name__ == "__main__":
    main()
