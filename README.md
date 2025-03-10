# Image Captioning System Using Transfer Learning and Attention Mechanism

## Prerequisites

- Python 3.12.1
- Git
- Sufficient disk space for models and data

## Installation Steps

1. **Clone the Repository**
 ```bash
 git clone https://github.com/yourusername/Image-Captioning-System.git
 cd Image-Captioning-System
 ```

2. **Create and Activate Virtual Environment**
 ```bash
 # Windows
 python -m venv venv
 .\venv\Scripts\activate

 # Linux/MacOS
 python -m venv venv
 source venv/bin/activate
 ```

3. **Install Requirements**
 ```bash
 pip install -r requirements.txt
 ```

 Requirements list:
 ```
 torch>=2.2.0
 torchvision>=0.17.0
 transformers>=4.37.2
 Pillow>=10.2.0
 streamlit>=1.31.0
 gTTS>=2.5.0
 pandas>=2.2.0
 plotly>=5.18.0
 python-dotenv>=1.0.1
 numpy>=1.26.3
 requests>=2.31.0
 protobuf>=4.25.2
 nltk>=3.8.1
 rouge-score>=0.1.2
 pycocoevalcap
 ```

## First-Time Setup

These steps are required only for the first time you run the project:

1. **Run NLTK Setup**
 ```bash
 python nltk_setup.py
 ```

2. **Verify NLTK Installation**
 ```bash
 python test_nltk.py
 ```

## Running the Application

After completing the setup, run the main application:
```bash
streamlit run main.py
```

## Project Structure
```
enhanced-image-captioning/
├── app/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   └── utils.py
├── data/
│   ├── audio/
│   ├── cache/
│   ├── history/
│   ├── models/
│   ├── samples/
│   ├── uploads/
│   └── visualizations/
├── .streamlit/
├── main.py
├── nltk_setup.py
├── test_nltk.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Important Notes

1. Make sure to run `nltk_setup.py` and `test_nltk.py` only once during the initial setup
2. The application requires an active internet connection for the first run to download necessary NLTK data
3. All generated content will be stored in respective folders under the `data` directory

## Troubleshooting

If you encounter any issues:

1. Verify Python version:
 ```bash
 python --version  # Should show Python 3.12.1
 ```

2. Ensure all requirements are installed:
 ```bash
 pip list
 ```

3. Check if NLTK data is properly downloaded by running `test_nltk.py` again

