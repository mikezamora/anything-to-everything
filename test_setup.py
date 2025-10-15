#!/usr/bin/env python
"""
Test script to verify the EPUB to Audiobook setup
"""
import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        import ebooklib
        print("  ✓ ebooklib")
    except ImportError:
        print("  ✗ ebooklib - run: pip install ebooklib")
        return False
    
    try:
        from bs4 import BeautifulSoup
        print("  ✓ beautifulsoup4")
    except ImportError:
        print("  ✗ beautifulsoup4 - run: pip install beautifulsoup4")
        return False
    
    try:
        import requests
        print("  ✓ requests")
    except ImportError:
        print("  ✗ requests - run: pip install requests")
        return False
    
    try:
        import torch
        print("  ✓ torch")
    except ImportError:
        print("  ✗ torch - install PyTorch from pytorch.org")
        return False
    
    try:
        import torchaudio
        print("  ✓ torchaudio")
    except ImportError:
        print("  ✗ torchaudio - install with PyTorch")
        return False
    
    print("\n✓ All required modules are available\n")
    return True

def test_ollama():
    """Test Ollama availability"""
    print("Testing Ollama integration...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("  ✓ Ollama is available and running")
            models = response.json().get('models', [])
            if models:
                print(f"  Available models: {', '.join([m['name'] for m in models])}")
            return True
        else:
            print("  ⚠ Ollama responded but with an error")
            return False
    except Exception as e:
        print(f"  ⚠ Ollama not available: {e}")
        print("  Note: Ollama is optional. Install from https://ollama.ai")
        return False

def test_indextts():
    """Test IndexTTS2 availability"""
    print("\nTesting IndexTTS2 availability...")
    
    # Check for required model files
    model_dir = "../../../checkpoints"
    required_files = [
        "config.yaml",
        "gpt.pth",
        "s2mel.pth",
        "bpe.model",
    ]
    
    all_found = True
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            all_found = False
    
    if all_found:
        print("\n✓ IndexTTS2 model files are available")
    else:
        print("\n⚠ Some IndexTTS2 model files are missing")
        print("  Make sure you've downloaded the models to the checkpoints directory")
    
    return all_found

def test_modules():
    """Test individual project modules"""
    print("\nTesting project modules...")
    
    modules = [
        "epub_extractor",
        "text_segmenter",
        "ollama_processor",
        "audio_merger",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}.py")
        except ImportError as e:
            print(f"  ✗ {module}.py - {e}")
            all_ok = False
    
    if all_ok:
        print("\n✓ All project modules are available")
    else:
        print("\n⚠ Some project modules have issues")
    
    return all_ok

def main():
    print("="*70)
    print("EPUB to Audiobook Converter - Setup Test")
    print("="*70)
    print()
    
    results = []
    
    # Test imports
    results.append(("Required modules", test_imports()))
    
    # Test project modules
    results.append(("Project modules", test_modules()))
    
    # Test Ollama (optional)
    ollama_ok = test_ollama()
    # Don't include in pass/fail since it's optional
    
    # Test IndexTTS2
    results.append(("IndexTTS2 models", test_indextts()))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<50} {status}")
        if not result:
            all_passed = False
    
    if ollama_ok:
        print(f"{'Ollama (optional)':.<50} ✓ AVAILABLE")
    else:
        print(f"{'Ollama (optional)':.<50} ⚠ NOT AVAILABLE")
    
    print("="*70)
    
    if all_passed:
        print("\n✓ Setup is complete! You're ready to convert EPUB files.")
        print("\nTry the quick start:")
        print("  python main.py book.epub speaker.wav -o audiobook.wav")
    else:
        print("\n⚠ Setup is incomplete. Please resolve the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
