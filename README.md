# EPUB to Audiobook Converter

Convert EPUB files to audiobooks using IndexTTS2 with support for custom voice cloning and emotion control.

## Features

- **EPUB Text Extraction**: Automatically extracts and cleans text from EPUB files
- **Smart Segmentation**: Splits text into manageable segments (default: 500 words) while respecting sentence boundaries
- **Ollama Integration**: Optional text cleanup and processing using Ollama LLM
- **Voice Cloning**: Clone any voice using a reference audio sample
- **Emotion Control**: Control emotion using reference audio or emotion vectors
- **Automatic Merging**: Combines all segments into a single audiobook file

## Installation

### Prerequisites

1. Install the base IndexTTS requirements (see main README)
2. Install additional dependencies:

```bash
pip install ebooklib beautifulsoup4 lxml requests
```

### Optional: Ollama Setup

For text cleanup functionality, install Ollama:

1. Download and install Ollama from https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Ensure Ollama is running (it runs as a service by default)

## Usage

### Basic Usage

Convert an EPUB to an audiobook using a speaker reference audio:

```bash
python main.py book.epub speaker_voice.wav -o audiobook.wav
```

### With Emotion Reference

Add emotional control using a separate emotion reference audio:

```bash
python main.py book.epub speaker.wav -o audiobook.wav --emo-audio emotion_ref.wav --emo-alpha 0.8
```

### With Ollama Text Processing

Enable text cleanup using Ollama:

```bash
python main.py book.epub speaker.wav -o audiobook.wav --use-ollama --ollama-model llama2
```

### Advanced Options

```bash
python main.py book.epub speaker.wav -o audiobook.wav \
  --segment-words 400 \
  --use-ollama \
  --emo-audio emotion.wav \
  --emo-alpha 0.7 \
  --use-fp16 \
  --temperature 0.9 \
  --keep-segments
```

## Command Line Arguments

### Required Arguments

- `epub_file`: Path to the EPUB file to convert
- `speaker_audio`: Path to speaker reference audio file (WAV format recommended)
- `-o, --output`: Path for the output audiobook WAV file

### Text Processing Options

- `--segment-words`: Target words per segment (default: 500)
- `--max-words`: Maximum words per segment (default: 600)
- `--min-words`: Minimum words per segment (default: 100)
- `--use-ollama`: Enable Ollama text cleanup
- `--ollama-model`: Ollama model to use (default: llama2)
- `--ollama-url`: Ollama API URL (default: http://localhost:11434)

### Voice & Emotion Options

- `--emo-audio`: Path to emotion reference audio file
- `--emo-alpha`: Emotion blend strength (0.0-1.0, default: 1.0)
- `--emo-vector`: Manual emotion vector (8 values)
- `--use-emo-text`: Automatically detect emotion from text
- `--interval-silence`: Silence between sentences in ms (default: 200)
- `--segment-silence`: Silence between segments in ms (default: 500)

### Model Options

- `--config`: Path to config file (default: checkpoints/config.yaml)
- `--model-dir`: Path to model directory (default: checkpoints)
- `--use-fp16`: Use FP16 precision for faster generation
- `--device`: Device to use (e.g., cuda:0, cpu)
- `--no-cuda-kernel`: Disable CUDA kernel for BigVGAN
- `--use-deepspeed`: Use DeepSpeed (if available)

### Generation Options

- `--max-text-tokens`: Max text tokens per TTS segment (default: 120)
- `--temperature`: Generation temperature (default: 0.8)
- `--top-p`: Top-p sampling (default: 0.8)
- `--top-k`: Top-k sampling (default: 30)
- `--repetition-penalty`: Repetition penalty (default: 10.0)
- `--length-penalty`: Length penalty (default: 0.0)
- `--num-beams`: Number of beams for beam search (default: 3)

### Other Options

- `--work-dir`: Working directory for temporary files (default: ./work)
- `--keep-segments`: Keep individual segment audio files
- `-v, --verbose`: Verbose output

## Output

The tool generates:

1. **Main audiobook file**: The complete merged audiobook (specified by `-o`)
2. **Metadata file**: Text file with audiobook information (same name as output with `_metadata.txt` suffix)
3. **Segment files** (optional): Individual audio files for each segment (if `--keep-segments` is used)
4. **Ollama artifacts** (if `--use-ollama` is used):
   - `work/ollama/prompts/` - Prompts sent to Ollama
   - `work/ollama/original_text/` - Original text segments
   - `work/ollama/processed_text/` - Processed text segments
   - `work/ollama/segment_XXXX_comparison.txt` - Side-by-side comparisons
   - `work/ollama/session_metadata.txt` - Session information
   - `work/ollama/processing_summary.txt` - Processing summary

## Workflow

1. **Extract**: Reads EPUB file and extracts text content
2. **Segment**: Splits text into segments (default: 500 words each)
3. **Process** (optional): Cleans text using Ollama LLM
   - Saves prompts to `work/ollama/prompts/`
   - Saves original text to `work/ollama/original_text/`
   - Saves processed text to `work/ollama/processed_text/`
   - Creates comparison files for review
4. **Generate**: Converts each segment to speech using IndexTTS2
5. **Merge**: Combines all audio segments into final audiobook
6. **Cleanup**: Removes temporary files (unless `--keep-segments` is used)

## Tips

### Performance

- Use `--use-fp16` for faster generation on CUDA devices
- Adjust `--segment-words` based on your GPU memory
- Smaller segments = more stable but slower overall

### Quality

- Use high-quality reference audio (clear, minimal background noise)
- For long books, consider using `--use-ollama` to clean up text formatting issues
- Experiment with `--temperature` and sampling parameters for different voice characteristics

### Emotion Control

Three ways to control emotion:

1. **Emotion audio reference**: `--emo-audio emotion.wav --emo-alpha 0.7`
2. **Automatic text detection**: `--use-emo-text`
3. **Manual vector**: `--emo-vector 0.5 0.2 0.1 0.0 0.0 0.1 0.0 0.6`

## Troubleshooting

### "No text extracted from EPUB"

- Ensure the EPUB file is valid and not corrupted
- Try opening it in an EPUB reader first
- Some DRM-protected EPUBs cannot be processed

### "Ollama not available"

- Make sure Ollama is installed and running
- Check that the Ollama API is accessible at the specified URL
- The tool will skip text processing if Ollama is unavailable

### Reviewing Ollama processing

When using `--use-ollama`, all processing artifacts are saved to `work/ollama/`:
- **prompts/**: The prompts sent to Ollama for each segment
- **original_text/**: The original text before processing
- **processed_text/**: The cleaned text after Ollama processing
- **segment_XXXX_comparison.txt**: Side-by-side comparison files
- **session_metadata.txt**: Processing session information
- **processing_summary.txt**: Summary of the processing session

Review these files to understand how Ollama modified your text.

### Out of memory errors

- Reduce `--segment-words` to process smaller chunks
- Use `--use-fp16` to reduce memory usage
- Close other GPU-intensive applications

### Generation is too slow

- Use `--use-fp16` on CUDA devices
- Reduce `--num-beams` for faster generation
- Consider using a faster GPU or reducing segment size

## Examples

### Simple audiobook from EPUB

```bash
python main.py "my_book.epub" "my_voice.wav" -o "audiobook.wav"
```

### High-quality with emotion and cleanup

```bash
python main.py "novel.epub" "narrator.wav" -o "novel_audiobook.wav" \
  --use-ollama \
  --emo-audio "emotional_reading.wav" \
  --emo-alpha 0.6 \
  --use-fp16 \
  --keep-segments
```

### Fast generation with custom settings

```bash
python main.py "short_story.epub" "voice.wav" -o "story.wav" \
  --segment-words 300 \
  --temperature 0.9 \
  --top-p 0.85 \
  --use-fp16
```

## Module Documentation

### epub_extractor.py

Handles EPUB file reading and text extraction.

### text_segmenter.py

Intelligently splits text into segments while respecting sentence boundaries.

### ollama_processor.py

Interfaces with Ollama for text cleanup and processing.

### tts_processor.py

Wraps IndexTTS2 for batch processing of text segments.

### audio_merger.py

Merges multiple audio files with configurable silence intervals.

## License

This project follows the same license as IndexTTS2. See the main LICENSE file.

## Credits

Built on top of IndexTTS2 by the Index-TTS team.
