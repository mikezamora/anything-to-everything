# Quick Start Guide - EPUB to Audiobook

## Installation

1. **Install Python dependencies:**
   ```powershell
   cd lit_cov\epub_to_audiobook
   pip install -r requirements.txt
   ```

2. **Optional - Install Ollama for text cleanup:**
   - Download from https://ollama.ai
   - Install and run: `ollama pull llama2`

## Basic Usage

### Step 1: Prepare Your Files

You need:
- An EPUB file (your book)
- A speaker reference audio (WAV format, 10-30 seconds of clear speech)
- Optional: An emotion reference audio

### Step 2: Run the Converter

**Simple example:**
```powershell
uv run -m main "./inputs/mybook.epub" "./inputs/voice-ref.mp3" -o "./outputs/audiobook.wav"
```

**With all features:**
```powershell
uv run -m main "./inputs/TheBet.epub" "./inputs/voice-ref.mp3" -o "./outputs/audiobook.wav" --emo-audio "./inputs/emo-ref.mp3" --emo-alpha 1 --keep-segments
```

```powershell
uv run -m main "./inputs/TheBet.epub" "./inputs/voice-ref.mp3" -o "./outputs/audiobook.wav" --emo-audio "./inputs/emo-ref.mp3" --emo-alpha 1 --keep-segments --use-ollama --ollama-model "aratan/DeepSeek-R1-32B-Uncensored:latest" --ollama-url "http://localhost:11434"
```


## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--segment-words` | Words per segment | `--segment-words 400` |
| `--use-ollama` | Enable text cleanup | `--use-ollama` |
| `--emo-audio` | Emotion reference | `--emo-audio emotion.wav` |
| `--use-fp16` | Faster generation | `--use-fp16` |
| `--keep-segments` | Keep temp files | `--keep-segments` |
| `--verbose` | Detailed output | `--verbose` |

## Tips

1. **Performance**: Add `--use-fp16` for 2x faster generation on GPU
2. **Quality**: Use high-quality, clear reference audio (WAV format)
3. **Long books**: Reduce `--segment-words` if you run out of memory
4. **Text cleanup**: Use `--use-ollama` for better text quality

## Troubleshooting

**"ModuleNotFoundError: No module named 'ebooklib'"**
```powershell
pip install ebooklib beautifulsoup4
```

**"CUDA out of memory"**
```powershell
python main.py book.epub speaker.wav -o output.wav --segment-words 300
```

**"Ollama not available"**
- Make sure Ollama is installed and running
- Or skip with: don't use `--use-ollama` flag

## Example Workflow

```powershell
# 1. Navigate to the project
cd F:\experiments\index-tts\lit_cov\epub_to_audiobook

# 2. Prepare your files in a folder
mkdir my_audiobook
# Place: my_audiobook\book.epub and my_audiobook\speaker.wav

# 3. Run conversion
python main.py my_audiobook\book.epub my_audiobook\speaker.wav `
    -o my_audiobook\audiobook.wav `
    --use-fp16 `
    --verbose

# 4. Find your audiobook at: my_audiobook\audiobook.wav
```

## What Happens During Processing?

1. **[1/6] Extract**: Reads text from EPUB
2. **[2/6] Segment**: Splits into 500-word chunks
3. **[3/6] Process**: Optional Ollama cleanup
4. **[4/6] Initialize**: Loads IndexTTS2 models
5. **[5/6] Generate**: Creates audio for each segment
6. **[6/6] Merge**: Combines into final audiobook

## Output Files

After completion, you'll have:
- `audiobook.wav` - Your complete audiobook
- `audiobook_metadata.txt` - Information about the generation
- `work/segments/` - Individual segment files (if `--keep-segments` used)

## Need Help?

See the full README.md for detailed documentation and advanced options.
