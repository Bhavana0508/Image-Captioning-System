import streamlit as st
from pathlib import Path
import time
from app.models import EnhancedCaptioningSystem
from app.utils import is_valid_image, process_image_file
from app.config import UPLOAD_DIR, SUPPORTED_FORMATS
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import logging
import json
import os
import shutil
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the captioning system
@st.cache_resource(show_spinner=False)
def get_captioning_system() -> EnhancedCaptioningSystem:
    """Initialize and cache the captioning system"""
    try:
        return EnhancedCaptioningSystem()
    except Exception as e:
        logger.error(f"Error initializing captioning system: {e}")
        st.error("Error initializing the system. Please try refreshing the page.")
        return None

def clean_system():
    """Clean all cache, history, and uploaded files"""
    directories = [
        'data/uploads',
        'data/history',
        'data/audio',
        'data/models',
        'data/cache',
        'data/samples',
        '.streamlit'  # Streamlit cache directory
    ]
    
    # Clear Streamlit cache first
    st.cache_resource.clear()
    
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            try:
                # Remove all files in the directory instead of removing the directory
                for file in dir_path.glob('*'):
                    try:
                        if file.is_file():
                            file.unlink()
                        elif file.is_dir():
                            shutil.rmtree(file)
                    except Exception as e:
                        logger.warning(f"Could not remove {file}: {e}")
            except Exception as e:
                logger.error(f"Error cleaning directory {directory}: {e}")
    
    # Recreate necessary directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

    # Reset the history files
    history_file = Path('data/history/caption_history.json')
    metrics_file = Path('data/history/metrics_history.json')
    
    try:
        # Create empty history files
        for file in [history_file, metrics_file]:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('[]')
    except Exception as e:
        logger.error(f"Error resetting history files: {e}")

def create_metrics_chart(metrics: Dict) -> Optional[go.Figure]:
    """Create a bar chart for metrics visualization"""
    if not metrics:
        return None
        
    metrics_data = {
        'Basic': {
            'Accuracy': metrics["accuracy"],
            'Precision': metrics["precision"],
            'Recall': metrics["recall"],
            'F1 Score': metrics["f1_score"]
        },
        'Advanced': {
            'BLEU': metrics["bleu"],
            'METEOR': metrics["meteor"],
            'ROUGE-L': metrics["rouge_l"],
            'CIDEr': metrics["cider"]
        }
    }
    
    metrics_df = pd.DataFrame([
        {'Metric': metric, 'Value': value, 'Category': category}
        for category, metrics_dict in metrics_data.items()
        for metric, value in metrics_dict.items()
    ])
    
    fig = px.bar(
        metrics_df,
        x='Metric',
        y='Value',
        color='Category',
        title='Model Performance Metrics',
        range_y=[0, 1],
        barmode='group',
        color_discrete_map={
            'Basic': '#1f77b4',
            'Advanced': '#ff7f0e'
        }
    )
    
    fig.update_layout(
        height=500,
        yaxis_title='Score',
        xaxis_title='',
        legend_title='Metric Type',
        xaxis={'categoryorder': 'total descending'},
        template='plotly_white',
        title_x=0.5
    )
    
    return fig

def create_loss_chart(
    epochs: List[int],
    training_losses: List[float],
    validation_losses: List[float]
) -> go.Figure:
    """Create a line chart for training and validation losses"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=training_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=validation_losses,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title={
            'text': 'Training and Validation Loss Over Time',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Epochs',
        yaxis_title='Loss',
        height=400,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )
    
    return fig

def display_sidebar_info():
    """Display sidebar information and controls"""
    with st.sidebar:
        st.title("🖼️ Enhanced Image Captioning")
        st.markdown("---")
        
        # System information
        st.subheader("ℹ️ System Information")
        st.markdown("""
        - **Model**: Enhanced Captioning System
        - **Version**: 1.0.0
        - **Status**: Active
        """)
        
        # Reset system
        st.markdown("---")
        st.subheader("🔄 System Reset")
        if st.button("Reset System"):
            try:
                clean_system()
                st.success("System reset successful! All caches and history cleared.")
                time.sleep(1)  # Give some time for the cleanup to complete
                st.rerun()
            except Exception as e:
                st.error(f"Error during reset: {str(e)}")
                logger.error(f"Reset error: {e}")
        
        # About section
        st.markdown("---")
        st.subheader("📝 About")
        st.markdown("""
        This system uses advanced AI models to:
        - Generate detailed image captions
        - Enhance basic captions
        - Provide performance metrics
        - Convert captions to speech
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("Made with ❤️ by Your Team")

def display_image_upload_section() -> Tuple[st.columns, Optional[Path]]:
    """Display the image upload section and return the processed image path"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=list(SUPPORTED_FORMATS),
            help="Supported formats: " + ", ".join(SUPPORTED_FORMATS)
        )
        
        if uploaded_file:
            file_path = UPLOAD_DIR / uploaded_file.name
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if is_valid_image(str(file_path)):
                    processed_path = process_image_file(str(file_path))
                    st.image(
                        processed_path,
                        use_container_width=True,
                        caption="Uploaded Image"
                    )
                    return (col1, col2), processed_path
                else:
                    st.error("Invalid image file. Please upload a valid image.")
            except Exception as e:
                logger.error(f"Error processing upload: {e}")
                st.error(f"Error processing upload: {str(e)}")
        
        return (col1, col2), None

def display_results(result: Dict, col2: st.columns):
    """Display the captioning results"""
    with col2:
        st.header("📝 Results")
        
        # Base Caption
        with st.expander("Base Caption", expanded=True):
            st.write(result["base_caption"])
        
        # Enhanced Caption
        st.subheader("✨ Enhanced Caption")
        st.write(result["improved_caption"])
        
        # Audio Version
        if "audio_file" in result:
            with st.expander("🔊 Audio Version", expanded=True):
                st.audio(result["audio_file"])
                st.download_button(
                    "Download Audio",
                    open(result["audio_file"], "rb"),
                    file_name="caption_audio.mp3"
                )

def display_metrics_tab(system: EnhancedCaptioningSystem):
    """Display the metrics tab content"""
    st.header("📊 Performance Metrics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        metrics = system.get_comparison_metrics()
        if metrics:
            metrics_chart = create_metrics_chart(metrics)
            if metrics_chart:  # Check if chart was created successfully
                st.plotly_chart(metrics_chart, use_container_width=True)
            
            with st.expander("📘 Metrics Explanation"):
                st.markdown("""
                **Basic Metrics:**
                - **Accuracy**: Overall prediction accuracy (0-1)
                - **Precision**: Exactness of the predictions
                - **Recall**: Completeness of the predictions
                - **F1 Score**: Harmonic mean of precision and recall
                
                **Advanced Metrics:**
                - **BLEU**: Bilingual Evaluation Understudy
                - **METEOR**: Metric for Evaluation of Translation with Explicit ORdering
                - **ROUGE-L**: Longest Common Subsequence based metric
                - **CIDEr**: Consensus-based Image Description Evaluation
                """)
        else:
            st.info("No metrics data available yet.")
    
    with col2:
        try:
            # Get loss history data
            epochs, training_losses, validation_losses = system.get_loss_history()
            
            # Check if we have valid data
            if (epochs and training_losses and validation_losses and 
                len(epochs) > 0 and len(epochs) == len(training_losses) == len(validation_losses)):
                
                # Create and display the loss chart
                loss_chart = create_loss_chart(epochs, training_losses, validation_losses)
                st.plotly_chart(loss_chart, use_container_width=True)
                
                with st.expander("📘 Loss Curves Explanation"):
                    st.markdown("""
                    **Understanding the Loss Curves:**
                    
                    - **Training Loss** (Blue): Shows model's learning progress during training
                    - **Validation Loss** (Orange): Shows model's performance on unseen data
                    
                    **Interpretation:**
                    - Converging curves: Good balance between training and validation
                    - Diverging curves: Possible overfitting (validation loss increases)
                    - High fluctuation: Unstable training or learning rate too high
                    - Plateau: Model might have reached optimal performance
                    """)
            else:
                st.info("No loss history data available yet.")
        except Exception as e:
            logger.error(f"Error displaying loss curves: {e}")
            st.error("Error displaying loss curves. Please check the system logs.")

def create_loss_chart(
    epochs: List[int],
    training_losses: List[float],
    validation_losses: List[float]
) -> go.Figure:
    """Create a line chart for training and validation losses"""
    fig = go.Figure()
    
    # Add training loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=training_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    # Add validation loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=validation_losses,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8)
    ))
    
    # Update layout with more detailed configuration
    fig.update_layout(
        title={
            'text': 'Training and Validation Loss Over Time',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Epochs',
        yaxis_title='Loss',
        height=400,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,  # Show every epoch
            gridcolor='lightgray'
        ),
        yaxis=dict(
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='white'
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate="<b>Epoch</b>: %{x}<br>" +
                      "<b>Loss</b>: %{y:.4f}<br>" +
                      "<extra></extra>"  # This removes the secondary box
    )
    
    return fig

def display_history_tab(system: EnhancedCaptioningSystem):
    """Display the history tab content"""
    st.header("📜 Caption History")
    
    if system.history:
        # Add filter and sort options
        col1, col2 = st.columns([1, 1])
        with col1:
            sort_order = st.selectbox(
                "Sort by",
                ["Newest First", "Oldest First"]
            )
        with col2:
            filter_metric = st.selectbox(
                "Filter by Metric",
                ["All", "Accuracy > 0.7", "BLEU > 0.5", "METEOR > 0.4"]
            )
        
        # Get history and convert to list
        history = list(system.history)
        
        # Sort the history based on timestamp
        history.sort(key=lambda x: datetime.fromisoformat(x['timestamp']), 
                    reverse=(sort_order == "Newest First"))
        
        # Apply filtering
        filtered_history = []
        for entry in history:
            metrics = entry.get('metrics', {})
            if filter_metric == "All":
                filtered_history.append(entry)
            elif filter_metric == "Accuracy > 0.7":
                if metrics.get('accuracy', 0) > 0.7:
                    filtered_history.append(entry)
            elif filter_metric == "BLEU > 0.5":
                if metrics.get('bleu', 0) > 0.5:
                    filtered_history.append(entry)
            elif filter_metric == "METEOR > 0.4":
                if metrics.get('meteor', 0) > 0.4:
                    filtered_history.append(entry)
        
        # Show number of entries after filtering
        st.markdown(f"**Showing {len(filtered_history)} entries**")
        
        if not filtered_history:
            st.info("No entries match the selected filter criteria.")
            return
        
        # Display entries
        for entry in filtered_history:
            st.markdown("---")
            timestamp = datetime.fromisoformat(entry['timestamp'])
            st.subheader(f"📸 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Create three columns for image, captions, and metrics
            img_col, cap_col, met_col = st.columns([1, 1, 1])
            
            with img_col:
                if Path(entry["image_path"]).exists():
                    st.image(entry["image_path"], use_container_width=True)
                else:
                    st.warning("Image file not found")
                
                # Add audio player if audio file exists
                if "audio_file" in entry and Path(entry["audio_file"]).exists():
                    st.audio(entry["audio_file"])
                    st.download_button(
                        "Download Audio",
                        open(entry["audio_file"], "rb"),
                        file_name=f"caption_audio_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp3"
                    )
            
            with cap_col:
                st.markdown("**Base Caption:**")
                st.write(entry["base_caption"])
                st.markdown("**Enhanced Caption:**")
                st.write(entry["improved_caption"])
            
            with met_col:
                if "metrics" in entry:
                    st.markdown("**📊 Performance Metrics**")
                    
                    # Create metrics dataframe
                    metrics_data = {
                        'Metric': [],
                        'Value': []
                    }
                    
                    # Basic metrics
                    basic_metrics = {
                        'Accuracy': entry['metrics'].get("accuracy", 0),
                        'Precision': entry['metrics'].get("precision", 0),
                        'Recall': entry['metrics'].get("recall", 0),
                        'F1': entry['metrics'].get("f1_score", 0)
                    }
                    
                    # Advanced metrics
                    advanced_metrics = {
                        'BLEU': entry['metrics'].get("bleu", 0),
                        'METEOR': entry['metrics'].get("meteor", 0),
                        'ROUGE-L': entry['metrics'].get("rouge_l", 0),
                        'CIDEr': entry['metrics'].get("cider", 0)
                    }
                    
                    # Combine metrics
                    for metric, value in {**basic_metrics, **advanced_metrics}.items():
                        metrics_data['Metric'].append(metric)
                        metrics_data['Value'].append(f"{value:.2f}")
                    
                    # Create and display dataframe
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(
                        metrics_df,
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Display loss values if available
                    if "training_loss" in entry['metrics'] and "validation_loss" in entry['metrics']:
                        st.markdown("**📉 Loss Values**")
                        loss_df = pd.DataFrame({
                            'Type': ['Training Loss', 'Validation Loss'],
                            'Value': [
                                f"{entry['metrics']['training_loss']:.4f}",
                                f"{entry['metrics']['validation_loss']:.4f}"
                            ]
                        })
                        st.dataframe(
                            loss_df,
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    # Add expandable detailed view
                    with st.expander("View Raw Metrics"):
                        display_metrics = {k: v for k, v in entry['metrics'].items() 
                                        if k not in ['timestamp', 'base_length', 'improved_length']}
                        st.json(display_metrics)
    else:
        st.info("No caption history available yet. Process some images to see the history.")

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Enhanced Image Captioning",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🖼️ Enhanced Image Captioning System")
    st.markdown("""
    This advanced system uses state-of-the-art AI models to generate detailed
    captions for your images and converts them to natural-sounding speech.
    """)

    # Display sidebar information
    display_sidebar_info()

    # Initialize the captioning system
    system = get_captioning_system()
    if system is None:
        st.stop()  # Stop execution if system initialization failed
        return

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📸 Upload & Results", "📊 Metrics", "📜 History"])
    
    # Rest of your main() function remains the same...

    with tab1:
        columns, processed_path = display_image_upload_section()
        
        if processed_path:
            try:
                with st.spinner("🤖 Analyzing image and generating captions..."):
                    result = system.process_image(processed_path)
                    
                if result:
                    display_results(result, columns[1])
            except Exception as e:
                logger.error(f"Error in image processing: {e}")
                st.error(f"Error processing image: {str(e)}")

    with tab2:
        display_metrics_tab(system)

    with tab3:
        display_history_tab(system)

if __name__ == "__main__":
    main()