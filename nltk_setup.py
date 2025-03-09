import nltk
import ssl
import os
from pathlib import Path
import shutil
import time

def setup_nltk():
    # Set specific NLTK data path and ensure it exists
    nltk_data_dir = Path.home() / 'nltk_data'
    nltk_data_dir.mkdir(parents=True, exist_ok=True)

    # Configure SSL for download
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Set environment variable
    os.environ['NLTK_DATA'] = str(nltk_data_dir)
    
    # Reset NLTK's data path
    nltk.data.path = [str(nltk_data_dir)]

    print(f"NLTK data will be stored in: {nltk_data_dir}")

    # Define required NLTK packages
    required_packages = [
        'punkt',
        'punkt_tab',
        'wordnet',
        'averaged_perceptron_tagger',
        'universal_tagset',
        'omw-1.4',
        'words'
    ]

    print("\nDownloading NLTK data...")
    
    # First, try downloading everything
    nltk.download('all', download_dir=str(nltk_data_dir), quiet=False)
    
    # Then verify specific packages
    print("\nVerifying specific package installations:")
    for package in required_packages:
        try:
            # Force download of specific package
            nltk.download(package, download_dir=str(nltk_data_dir), quiet=False, force=True)
            time.sleep(1)  # Add small delay between downloads
            
            # Verify the package
            if package == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif package == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif package in ['wordnet', 'omw-1.4']:
                nltk.data.find(f'corpora/{package}')
            else:
                nltk.data.find(f'taggers/{package}')
            print(f"✓ {package}: Successfully installed and verified")
        except Exception as e:
            print(f"✗ {package}: Installation or verification failed - {str(e)}")

    # Test basic functionality
    print("\nTesting basic NLTK functionality:")
    try:
        # Test tokenization
        text = "Testing NLTK functionality."
        tokens = nltk.word_tokenize(text)
        print(f"✓ Tokenization working: {tokens}")
        
        # Test POS tagging
        tagged = nltk.pos_tag(tokens)
        print(f"✓ POS tagging working: {tagged}")
        
        # Test WordNet
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets('test')
        print(f"✓ WordNet working: Found {len(synsets)} synsets for 'test'")
        
    except Exception as e:
        print(f"✗ Functionality test failed: {str(e)}")

    print("\nEnvironment Information:")
    print(f"NLTK Version: {nltk.__version__}")
    print(f"NLTK Data Path: {nltk.data.path}")
    print(f"NLTK_DATA environment variable: {os.getenv('NLTK_DATA')}")

if __name__ == "__main__":
    # Clean up existing installation first
    nltk_data_dir = Path.home() / 'nltk_data'
    if nltk_data_dir.exists():
        print(f"Removing existing NLTK data directory: {nltk_data_dir}")
        shutil.rmtree(nltk_data_dir)
    
    setup_nltk()