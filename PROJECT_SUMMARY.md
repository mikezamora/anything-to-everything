# EPUB to Audiobook Converter - Project Summary

## 📁 Project Structure

```
lit_cov/epub_to_audiobook/
├── __init__.py                 # Package initialization
├── main.py                     # Main CLI entry point
├── epub_extractor.py          # EPUB text extraction module
├── text_segmenter.py          # Text segmentation module (500 words)
├── ollama_processor.py        # Ollama LLM integration (with artifact saving)
├── tts_processor.py           # IndexTTS2 integration for speech synthesis
├── audio_merger.py            # Audio segment merging module
├── config_template.py         # Configuration template
├── test_setup.py              # Setup verification script
├── requirements.txt           # Python dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md              # Quick start guide
├── OLLAMA_ARTIFACTS.md        # Ollama artifacts guide
├── INSTALLATION.md            # Installation & troubleshooting guide
├── PROJECT_SUMMARY.md         # This file
├── example.sh                 # Bash example script
├── example.ps1                # PowerShell example script
├── convert_epub.bat           # Simple Windows batch script
└── .gitignore                 # Git ignore rules
```

## 🎯 Key Features Implemented

### 1. **EPUB Text Extraction** (`epub_extractor.py`)
- Reads and parses EPUB files
- Extracts text content from HTML
- Cleans formatting artifacts
- Extracts metadata (title, author, language)

### 2. **Smart Text Segmentation** (`text_segmenter.py`)
- Splits text into 500-word segments (configurable)
- Respects sentence boundaries
- Configurable min/max word limits
- Provides segment statistics

### 3. **Ollama Integration** (`ollama_processor.py`)
- Optional text cleanup using LLM
- Checks Ollama availability
- Processes segments individually
- Falls back gracefully if unavailable
- **Saves processing artifacts**:
  - Prompts sent to Ollama
  - Original text before processing
  - Processed text after cleanup
  - Side-by-side comparison files
  - Session metadata and summary

### 4. **TTS Processing** (`tts_processor.py`)
- Wraps IndexTTS2 for batch processing
- Processes segments sequentially
- Handles voice cloning with reference audio
- Supports emotion control (audio reference or vectors)
- Provides progress tracking

### 5. **Audio Merging** (`audio_merger.py`)
- Merges all audio segments
- Adds configurable silence between segments
- Handles different sampling rates
- Saves metadata alongside audio

### 6. **Main CLI** (`main.py`)
- Complete command-line interface
- Comprehensive argument parsing
- Progress tracking through all stages
- Error handling and validation
- Optional cleanup of temporary files

## 🔧 Core Workflow

```
EPUB File → Extract Text → Segment (500 words) → [Optional: Ollama Cleanup]
                                                           ↓
Final Audiobook ← Merge Segments ← Generate Audio (IndexTTS2)
```

### Processing Stages:

1. **[1/6] Extract**: Read EPUB and extract text content
2. **[2/6] Segment**: Split into 500-word chunks with sentence boundaries
3. **[3/6] Process**: Optional Ollama text cleanup
4. **[4/6] Initialize**: Load IndexTTS2 models
5. **[5/6] Generate**: Create audio for each segment using TTS
6. **[6/6] Merge**: Combine all segments into final audiobook

## 📋 Usage Examples

### Basic Usage
```powershell
python main.py book.epub speaker.wav -o audiobook.wav
```

### With All Features
```powershell
python main.py book.epub speaker.wav -o audiobook.wav `
    --use-ollama `
    --emo-audio emotion.wav `
    --emo-alpha 0.7 `
    --use-fp16 `
    --keep-segments `
    --verbose
```

### Custom Segmentation
```powershell
python main.py book.epub speaker.wav -o audiobook.wav `
    --segment-words 400 `
    --max-words 500 `
    --min-words 100
```

## 🎛️ Key Parameters

### Text Processing
- `--segment-words 500`: Target words per segment
- `--use-ollama`: Enable text cleanup
- `--ollama-model llama2`: LLM model to use

### Voice & Emotion
- `--emo-audio emotion.wav`: Emotion reference audio
- `--emo-alpha 0.7`: Emotion blend strength
- `--use-emo-text`: Auto-detect emotion from text

### Generation Quality
- `--temperature 0.8`: Controls randomness
- `--top-p 0.8`: Nucleus sampling
- `--repetition-penalty 10.0`: Reduces repetition
- `--num-beams 3`: Beam search width

### Performance
- `--use-fp16`: Use half precision (faster on GPU)
- `--device cuda:0`: Specify device
- `--use-deepspeed`: Enable DeepSpeed

## 📦 Dependencies

### Required
- **ebooklib**: EPUB file reading
- **beautifulsoup4**: HTML parsing
- **lxml**: XML/HTML processing
- **requests**: HTTP client for Ollama
- **torch/torchaudio**: Audio processing
- **IndexTTS2**: Text-to-speech engine

### Optional
- **Ollama**: Local LLM for text cleanup
- **DeepSpeed**: Accelerated inference

## 🚀 Installation

```powershell
# Navigate to project
cd F:\experiments\index-tts\lit_cov\epub_to_audiobook

# Install dependencies
pip install -r requirements.txt

# Optional: Install Ollama
# Download from https://ollama.ai

# Test setup
python test_setup.py
```

## ✅ Testing

Run the setup test to verify everything is working:

```powershell
python test_setup.py
```

This checks:
- ✓ Required Python modules
- ✓ Project modules
- ✓ IndexTTS2 model files
- ⚠ Ollama availability (optional)

## 📝 Output Files

After processing, you'll get:
- **audiobook.wav**: The complete merged audiobook
- **audiobook_metadata.txt**: Generation metadata
- **work/segments/**: Individual segment files (if `--keep-segments`)

## 🎨 Customization

Edit `config_template.py` to set default values for:
- Text processing parameters
- TTS generation settings
- Audio configuration
- Emotion settings
- Model preferences

## 🔍 Module Details

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `epub_extractor.py` | EPUB parsing | Text extraction, metadata, HTML cleaning |
| `text_segmenter.py` | Text splitting | 500-word segments, sentence boundaries |
| `ollama_processor.py` | LLM integration | Text cleanup, optional processing |
| `tts_processor.py` | TTS generation | IndexTTS2 wrapper, batch processing |
| `audio_merger.py` | Audio combining | Segment merging, silence insertion |
| `main.py` | CLI interface | Complete workflow orchestration |

## 💡 Tips for Best Results

1. **Audio Quality**: Use clear, high-quality reference audio (WAV, 16kHz+)
2. **Segment Size**: Adjust based on GPU memory (300-600 words)
3. **Text Cleanup**: Use `--use-ollama` for cleaner text formatting
4. **Performance**: Enable `--use-fp16` on CUDA devices for 2x speedup
5. **Emotion**: Experiment with `--emo-alpha` (0.5-0.9) for best results

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `--segment-words` |
| Slow generation | Add `--use-fp16` |
| Bad text quality | Use `--use-ollama` |
| Missing modules | Run `pip install -r requirements.txt` |
| Ollama errors | Skip with: remove `--use-ollama` flag |

## 📖 Documentation Files

- **README.md**: Complete documentation with all options
- **QUICKSTART.md**: Fast getting started guide
- **config_template.py**: Configuration template
- **example.ps1**: PowerShell example script
- **example.sh**: Bash example script

## 🎯 Project Status

✅ **Complete and Ready to Use**

All core features implemented:
- ✅ EPUB text extraction
- ✅ Smart 500-word segmentation
- ✅ Ollama integration (optional)
- ✅ IndexTTS2 integration
- ✅ Audio segment merging
- ✅ Full CLI interface
- ✅ Comprehensive documentation
- ✅ Example scripts
- ✅ Setup verification

## 🎬 Next Steps

1. **Test the setup**: `python test_setup.py`
2. **Try an example**: Follow QUICKSTART.md
3. **Customize**: Edit config_template.py for your defaults
4. **Process your book**: Use main.py with your EPUB file

## 📞 Support

For issues or questions:
1. Check README.md for detailed documentation
2. Run test_setup.py to verify installation
3. See QUICKSTART.md for common use cases
4. Review example scripts for working configurations

---

**Created**: 2025-10-14  
**Version**: 1.0.0  
**Status**: Production Ready ✅
