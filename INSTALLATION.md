# Installation Guide - EPUB to Audiobook Converter

## Quick Install (Recommended)

```powershell
# Navigate to the project directory
cd F:\experiments\index-tts\lit_cov\epub_to_audiobook

# Install only the required dependencies
pip install ebooklib beautifulsoup4 lxml requests tqdm
```

## About DeepSpeed

The error you saw about `deepspeed` is related to the optional dependency in the main IndexTTS project. **DeepSpeed is NOT required** for the EPUB to Audiobook converter to work.

### Why the Error Occurred

The main `pyproject.toml` includes DeepSpeed as an optional dependency with:
```toml
[project.optional-dependencies]
deepspeed = ["deepspeed==0.17.1"]
```

When you ran `uv sync --all-extras`, it tried to install ALL optional dependencies, including DeepSpeed, which has complex build requirements.

## Solutions

### Option 1: Install Without All Extras (Recommended)

```powershell
# Install only base dependencies
uv sync

# Then install EPUB converter dependencies
cd lit_cov\epub_to_audiobook
pip install -r requirements.txt
```

### Option 2: Install Without DeepSpeed

```powershell
# Install all extras EXCEPT deepspeed
uv sync --extra gpu --extra training
# (or whatever extras you need, just not deepspeed)

# Then install EPUB converter dependencies
cd lit_cov\epub_to_audiobook
pip install -r requirements.txt
```

### Option 3: Skip DeepSpeed in the Audiobook Project

The EPUB to Audiobook converter has DeepSpeed disabled by default:

```python
# In main.py, DeepSpeed is NOT enabled by default
parser.add_argument("--use-deepspeed", action="store_true", help="Use DeepSpeed")

# In tts_processor.py, it's disabled by default
use_deepspeed=False
```

So you can use the tool without DeepSpeed completely.

## Installing DeepSpeed (Optional, Advanced Users Only)

If you really need DeepSpeed for faster inference on large batches:

### Prerequisites
1. **Visual Studio 2019 or 2022** with C++ build tools
2. **CUDA Toolkit** (matching your PyTorch CUDA version)

### Installation Steps

```powershell
# 1. Check your CUDA version
python -c "import torch; print(torch.version.cuda)"

# 2. Install Visual Studio Build Tools (if not installed)
# Download from: https://visualstudio.microsoft.com/downloads/
# Select: "Desktop development with C++"

# 3. Try installing DeepSpeed
pip install deepspeed

# Or build from source if pre-built fails
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip install -e .
```

### If DeepSpeed Installation Fails

**Don't worry!** The EPUB to Audiobook converter works perfectly without DeepSpeed. Simply:
- Don't use the `--use-deepspeed` flag
- The tool will automatically detect that DeepSpeed is unavailable and fall back to normal inference

## Verifying Your Installation

```powershell
# Navigate to project
cd F:\experiments\index-tts\lit_cov\epub_to_audiobook

# Run the test
python test_setup.py
```

This will show:
- ✓ Required modules (ebooklib, beautifulsoup4, etc.)
- ✓ Project modules (all .py files)
- ✓ IndexTTS2 models
- ⚠ Ollama (optional)
- ⚠ DeepSpeed (optional)

## Full Installation from Scratch

If you're setting up everything from scratch:

```powershell
# 1. Clone/navigate to index-tts
cd F:\experiments\index-tts

# 2. Install base PyTorch (CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install IndexTTS base requirements
pip install -r requirements.txt

# 4. Install EPUB converter requirements
cd lit_cov\epub_to_audiobook
pip install -r requirements.txt

# 5. Test the setup
python test_setup.py
```

## Minimal Installation for Testing

Just want to try it out quickly?

```powershell
pip install ebooklib beautifulsoup4 lxml requests
```

Then run the test:
```powershell
python test_setup.py
```

## Troubleshooting

### "No module named 'setuptools'"

This is the error you saw. It means DeepSpeed tried to build but couldn't find setuptools. Solution:

```powershell
# Don't install deepspeed - skip it!
# Or if you must:
pip install setuptools wheel
```

### "ImportError: DLL load failed"

This usually means CUDA libraries are missing. Solution:
- Make sure CUDA Toolkit is installed
- Match CUDA version with PyTorch
- Or just don't use `--use-deepspeed`

### "ModuleNotFoundError: No module named 'ebooklib'"

```powershell
pip install ebooklib beautifulsoup4 lxml
```

## What You Actually Need

For the EPUB to Audiobook converter, you only need:

### Required
- ✅ Python 3.8+
- ✅ PyTorch + torchaudio
- ✅ ebooklib
- ✅ beautifulsoup4
- ✅ lxml
- ✅ requests
- ✅ IndexTTS2 models (in checkpoints/)

### Optional
- ⚠ Ollama (for text cleanup)
- ⚠ DeepSpeed (for faster inference, but complex to install)

### Not Needed for Basic Use
- ❌ Visual Studio (unless building DeepSpeed)
- ❌ CUDA Toolkit installation (PyTorch includes CUDA libraries)
- ❌ DeepSpeed (the tool works fine without it)

## Recommended Setup

```powershell
# 1. Basic Python dependencies
pip install ebooklib beautifulsoup4 lxml requests tqdm

# 2. (Optional) Install Ollama from https://ollama.ai

# 3. Test it
cd F:\experiments\index-tts\lit_cov\epub_to_audiobook
python test_setup.py

# 4. Done! Now you can convert EPUBs
python main.py book.epub speaker.wav -o audiobook.wav --use-fp16
```

## Summary

**You DO NOT need DeepSpeed** for the EPUB to Audiobook converter. The error you encountered was from trying to install all optional dependencies. Simply install the requirements from `requirements.txt` and you're good to go!
