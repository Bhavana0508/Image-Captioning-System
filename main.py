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
        st.title("🖼 Enhanced Image Captioning")
        st.markdown("---")

        # System information
        st.subheader("ℹ System Information")
        st.markdown("""
        - Model: Enhanced Captioning System
        - Version: 1.0.0
        - Status: Active
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
        st.markdown("Made with ❤ by Your Team")


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

        # Add Performance Metrics and Loss Values sections
        if "metrics" in result:
            # Performance Metrics
            st.markdown("📊 Performance Metrics**")

            # Create metrics dataframe
            metrics_data = {
                'Metric': [],
                'Value': []
            }

            # Basic metrics
            basic_metrics = {
                'Accuracy': result['metrics'].get("accuracy", 0),
                'Precision': result['metrics'].get("precision", 0),
                'Recall': result['metrics'].get("recall", 0),
                'F1': result['metrics'].get("f1_score", 0)
            }

            # Advanced metrics
            advanced_metrics = {
                'BLEU': result['metrics'].get("bleu", 0),
                'METEOR': result['metrics'].get("meteor", 0),
                'ROUGE-L': result['metrics'].get("rouge_l", 0),
                'CIDEr': result['metrics'].get("cider", 0)
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
            if "training_loss" in result['metrics'] and "validation_loss" in result['metrics']:
                st.markdown("📉 Loss Values**")
                loss_df = pd.DataFrame({
                    'Type': ['Training Loss', 'Validation Loss'],
                    'Value': [
                        f"{result['metrics']['training_loss']:.4f}",
                        f"{result['metrics']['validation_loss']:.4f}"
                    ]
                })
                st.dataframe(
                    loss_df,
                    hide_index=True,
                    use_container_width=True
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
                Basic Metrics:
                - Accuracy: Overall prediction accuracy (0-1)
                - Precision: Exactness of the predictions
                - Recall: Completeness of the predictions
                - F1 Score: Harmonic mean of precision and recall

                Advanced Metrics:
                - BLEU: Bilingual Evaluation Understudy
                - METEOR: Metric for Evaluation of Translation with Explicit ORdering
                - ROUGE-L: Longest Common Subsequence based metric
                - CIDEr: Consensus-based Image Description Evaluation
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
                    Understanding the Loss Curves:

                    - Training Loss (Blue): Shows model's learning progress during training
                    - Validation Loss (Orange): Shows model's performance on unseen data

                    Interpretation:
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
        hovermode='x unified'
    )

    return fig


def display_history_tab(system: EnhancedCaptioningSystem):
    """Display the history tab content"""
    st.header("📜 Caption History")

    try:
        # Try to get history data if the method exists
        if hasattr(system, 'get_caption_history'):
            history = system.get_caption_history()
        else:
            # Fallback: Try to load history directly from file
            history_file = Path('data/history/caption_history.json')
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading history file: {e}")
                    history = []
            else:
                history = []

        if not history:
            st.info("No caption history available yet.")
            return

        # Add filtering options
        st.subheader("🔍 Filter Options")
        col1, col2 = st.columns(2)

        with col1:
            if history and all("timestamp" in entry for entry in history):
                try:
                    min_date = datetime.strptime(min([entry["timestamp"] for entry in history]), "%Y-%m-%dT%H:%M:%S.%f")
                    max_date = datetime.strptime(max([entry["timestamp"] for entry in history]), "%Y-%m-%dT%H:%M:%S.%f")

                    date_range = st.date_input(
                        "Date Range",
                        [min_date.date(), max_date.date()],
                        min_value=min_date.date(),
                        max_value=max_date.date()
                    )
                except (ValueError, KeyError) as e:
                    logger.error(f"Error processing timestamps: {e}")
                    st.error("Error processing history timestamps.")
                    return
            else:
                st.text("No timestamp data available")
                date_range = []

        with col2:
            sort_by = st.selectbox(
                "Sort By",
                ["Newest First", "Oldest First", "Highest Accuracy", "Highest BLEU Score"]
            )

        # Filter and sort history
        filtered_history = []

        for entry in history:
            if "timestamp" not in entry:
                continue

            try:
                entry_date = datetime.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%S.%f").date()

                if len(date_range) == 2:
                    if date_range[0] <= entry_date <= date_range[1]:
                        filtered_history.append(entry)
                else:
                    filtered_history.append(entry)
            except ValueError:
                # Skip entries with invalid timestamp format
                continue

        # Sort the filtered history
        if sort_by == "Newest First":
            filtered_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        elif sort_by == "Oldest First":
            filtered_history.sort(key=lambda x: x.get("timestamp", ""))
        elif sort_by == "Highest Accuracy":
            filtered_history.sort(key=lambda x: x.get("metrics", {}).get("accuracy", 0), reverse=True)
        elif sort_by == "Highest BLEU Score":
            filtered_history.sort(key=lambda x: x.get("metrics", {}).get("bleu", 0), reverse=True)

        # Display the filtered history
        st.subheader(f"📊 Results ({len(filtered_history)} entries)")

        for i, entry in enumerate(filtered_history):
            try:
                timestamp_display = datetime.strptime(entry['timestamp'], '%Y-%m-%dT%H:%M:%S.%f').strftime(
                    '%Y-%m-%d %H:%M:%S')
            except (ValueError, KeyError):
                timestamp_display = "Unknown date"

            with st.expander(f"Entry {i + 1} - {timestamp_display}"):
                col1, col2 = st.columns([1, 1])

                with col1:
                    image_path = entry.get("image_path", "")
                    if image_path and Path(image_path).exists():
                        st.image(image_path, use_container_width=True)
                    else:
                        st.error("Image not found")

                with col2:
                    st.markdown("Base Caption:")
                    st.write(entry.get("base_caption", "N/A"))

                    st.markdown("Enhanced Caption:")
                    st.write(entry.get("improved_caption", "N/A"))

                    audio_file = entry.get("audio_file", "")
                    if audio_file and Path(audio_file).exists():
                        st.audio(audio_file)

                # Display metrics for this entry
                if "metrics" in entry:
                    st.markdown("📊 Performance Metrics**")

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
                    if "training_loss" in entry.get('metrics', {}) and "validation_loss" in entry.get('metrics', {}):
                        st.markdown("📉 Loss Values**")
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
    except Exception as e:
        logger.error(f"Error displaying history: {e}")
        st.error("Error displaying caption history. Please check the system logs.")


def main():
    """Main application function"""
    st.set_page_config(
        page_title="Enhanced Image Captioning",
        page_icon="🖼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Display sidebar
    display_sidebar_info()

    # Initialize the captioning system
    system = get_captioning_system()
    if not system:
        st.error("Failed to initialize the captioning system. Please refresh the page.")
        return

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Upload & Results", "Metrics", "History"])

    # Tab 1: Upload & Results
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

    # Tab 2: Metrics
    with tab2:
        try:
            display_metrics_tab(system)
        except Exception as e:
            logger.error(f"Error displaying metrics tab: {e}")
            st.error("Error displaying metrics. Please check the system logs.")

    # Tab 3: History
    with tab3:
        try:
            display_history_tab(system)
        except Exception as e:
            logger.error(f"Error displaying history tab: {e}")
            st.error("Error displaying history. Please check the system logs.")


if __name__ == "__main__":
    main()
