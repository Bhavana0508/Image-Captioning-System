import nltk
from pathlib import Path
import os

def test_nltk_installation():
    print("Testing NLTK Installation")
    print("-" * 50)
    
    # Print environment info
    print(f"NLTK Version: {nltk.__version__}")
    print(f"NLTK Data Path: {nltk.data.path}")
    print(f"NLTK_DATA env: {os.getenv('NLTK_DATA')}")
    
    tests = {
        'Tokenization': {
            'func': lambda: nltk.word_tokenize("Testing NLTK tokenization functionality."),
            'expected_type': list
        },
        'WordNet': {
            'func': lambda: nltk.corpus.wordnet.synsets('test'),
            'expected_type': list
        },
        'POS Tagging': {
            'func': lambda: nltk.pos_tag(nltk.word_tokenize("Testing the tagger.")),
            'expected_type': list
        }
    }
    
    print("\nRunning Tests:")
    all_passed = True
    
    for test_name, test_info in tests.items():
        print(f"\n{test_name} Test:")
        try:
            result = test_info['func']()
            if isinstance(result, test_info['expected_type']):
                print(f"✓ {test_name} working")
                print(f"Sample output: {result}")
            else:
                print(f"✗ {test_name} returned unexpected type: {type(result)}")
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} failed: {str(e)}")
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed. Please check the output above.")

if __name__ == "__main__":
    test_nltk_installation()