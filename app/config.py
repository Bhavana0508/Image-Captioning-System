from pathlib import Path
import os

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
HISTORY_DIR = DATA_DIR / "history"
AUDIO_DIR = DATA_DIR / "audio"
MODEL_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

# Create necessary directories
for directory in [UPLOAD_DIR, HISTORY_DIR, AUDIO_DIR, MODEL_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File formats and limitations
SUPPORTED_FORMATS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes

# Image processing configurations
IMAGE_PROCESSING = {
    'max_size': 1024,     # Maximum dimension for processed images
    'quality': 85,        # JPEG quality for processed images
    'formats': SUPPORTED_FORMATS
}

# Model configurations
MODEL_CONFIG = {
    'base_model': 'microsoft/git-base-coco',
    'improved_model': 'Salesforce/blip-image-captioning-base',
    'cache_dir': str(CACHE_DIR)
}

# Audio configurations
AUDIO_CONFIG = {
    'format': 'mp3',
    'language': 'en',
    'slow': False
}

# History configurations
HISTORY_CONFIG = {
    'max_entries': 100,  # Maximum number of entries to keep in history
    'save_interval': 5   # Save history every N entries
}

# Logging configurations
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': str(DATA_DIR / 'app.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Development configurations
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# UI configurations
UI_CONFIG = {
    'theme': {
        'primaryColor': '#1f77b4',
        'backgroundColor': '#ffffff',
        'secondaryBackgroundColor': '#f0f2f6',
        'textColor': '#262730',
        'font': 'sans-serif'
    },
    'layout': {
        'wide_mode': True,
        'sidebar_state': 'expanded'
    }
}
