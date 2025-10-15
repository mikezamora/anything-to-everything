# EPUB to Audiobook Converter

Convert EPUB files to audiobooks using IndexTTS2 with support for custom voice cloning and emotion control.

## Features

- **Web UI** 🆕: 
  - Browser-based interface for all features
  - Job creation and batch processing
  - Real-time job monitoring
  - Character detection and management
  - File upload handling
  - Live terminal output
  - See [WEBUI_GUIDE.md](WEBUI_GUIDE.md) for details
- **EPUB Text Extraction**: Automatically extracts and cleans text from EPUB files with metadata support
- **Character-Aware Processing** 🆕: 
  - Automatic character detection with gender and demeanor analysis
  - Character-specific voice mapping
  - Dialogue, thought, and narration detection
  - Emotional state analysis per segment
  - Interactive character review and merging tool
- **Smart Segmentation**: Splits text into manageable segments while respecting:
  - Sentence boundaries
  - Character changes
  - Emotional shifts
- **Job Queue System** 🆕:
  - Batch processing for multiple audiobooks
  - Priority-based queue management
  - Per-job configuration overrides
  - Full job tracking and status monitoring
  - Automated processing workflow
- **Ollama Integration**: Optional text cleanup and processing using Ollama LLM
- **Voice Cloning**: Clone any voice using a reference audio sample
- **Multi-Voice Support** 🆕: Assign different voices to different characters
- **Emotion Control**: 
  - Automatic emotion detection from text
  - Per-character emotion reference audio
  - Dynamic emotion vectors based on content
- **Output Formats**: 
  - WAV (uncompressed)
  - M4B (compressed audiobook format with embedded metadata)
- **Automatic Merging**: Combines all segments into a single audiobook file

## Installation

### Prerequisites

1. Install the base IndexTTS requirements (see main README)
2. Install additional dependencies:

```bash
pip install ebooklib beautifulsoup4 lxml requests gradio pandas
```

### Optional: Ollama Setup

For text cleanup functionality, install Ollama:

1. Download and install Ollama from https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Ensure Ollama is running (it runs as a service by default)

### Optional: FFmpeg for M4B Support

To create M4B audiobook files, install FFmpeg:

- **Windows**: Download from https://ffmpeg.org/download.html or use `winget install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` or `sudo yum install ffmpeg`

## Usage

### Web UI (Recommended)

Launch the browser-based interface:

```bash
# Basic launch
python webui.py

# Or use the launcher
python launch_webui.py

# Windows
launch_webui.bat

# Custom host/port
python webui.py --host 0.0.0.0 --port 7860

# Create public share link
python webui.py --share
```

Access the UI at `http://localhost:7860`

See [WEBUI_GUIDE.md](WEBUI_GUIDE.md) for complete Web UI documentation.

### Command Line Interface

Convert an EPUB to an audiobook using a speaker reference audio:

```bash
# WAV format (uncompressed)
python main.py book.epub speaker_voice.wav -o audiobook.wav

# M4B format (compressed, with metadata)
python main.py book.epub speaker_voice.wav -o audiobook.m4b --format m4b
```

### With Emotion Reference

Add emotional control using a separate emotion reference audio:

```bash
python main.py book.epub speaker.wav -o audiobook.m4b --format m4b --emo-audio emotion_ref.wav --emo-alpha 0.8
```

### With Ollama Integration

#### Text Processing
Enable text cleanup using Ollama:

```bash
python main.py book.epub speaker.wav -o audiobook.m4b --format m4b --use-ollama --ollama-model llama2
```

#### Advanced Character Detection (Recommended)
Use Ollama for more accurate character detection:

```bash
python main.py book.epub speaker.wav -o audiobook.m4b \
  --detect-characters --ollama-character-detection
```

**Benefits of Ollama character detection:**
- Context-aware character identification
- Better gender and personality detection
- Pronoun resolution (connects "he/she/I" to characters)
- Filters out false positives (places, objects)
- More accurate for complex narratives

### Advanced Options

```bash
python main.py book.epub speaker.wav -o audiobook.m4b \
  --format m4b \
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
- `-o, --output`: Path for the output audiobook file

### Output Options

- `--format`: Output format - `wav` (uncompressed) or `m4b` (compressed audiobook format with metadata). Default: wav
  - **WAV**: Uncompressed, larger file size, no metadata embedding
  - **M4B**: Compressed AAC audio in MP4 container, smaller file size (~10-15x compression), embedded metadata (title, author, album)
- `--work-dir`: Working directory for temporary files (default: ./work)
- `--keep-segments`: Keep individual segment audio files

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

- `-v, --verbose`: Verbose output

## Output

The tool generates:

1. **Main audiobook file**: The complete merged audiobook (specified by `-o`)
   - **WAV format**: Uncompressed audio, larger file size
   - **M4B format**: Compressed AAC audio with embedded metadata (title, author, album, genre, comments)
2. **Metadata file**: Text file with audiobook information (same name as output with `_metadata.txt` suffix)
3. **Segment files** (optional): Individual audio files for each segment (if `--keep-segments` is used)
4. **Ollama artifacts** (if `--use-ollama` is used):
   - `work/ollama/prompts/` - Prompts sent to Ollama
   - `work/ollama/original_text/` - Original text segments
   - `work/ollama/processed_text/` - Processed text segments
   - `work/ollama/segment_XXXX_comparison.txt` - Side-by-side comparisons
   - `work/ollama/session_metadata.txt` - Session information
   - `work/ollama/processing_summary.txt` - Processing summary

### M4B Format Benefits

- **Smaller file size**: Typically 10-15x smaller than WAV
- **Embedded metadata**: Title, author, album, genre automatically embedded
- **Audiobook format**: Standard format recognized by audiobook players and devices
- **AAC compression**: High quality at 64kbps (optimal for voice)

## Character-Aware Mode 🆕

The character-aware mode automatically detects characters, analyzes their traits and emotional states, and generates audio with character-specific voices and emotions.

### Quick Start with Character Mode

#### Step 1: Detect Characters

```bash
python main.py book.epub dummy.wav -o output.m4b --detect-characters
```

This will:
- Analyze the EPUB and detect all characters
- Identify gender and demeanor for each character
- Create configuration templates:
  - `work/detected_characters.json` - Detected characters with traits
  - `work/character_voices_template.json` - Voice mapping template
  - `work/emotion_library_template.json` - Emotion reference template

#### Step 2: Review and Configure Characters (Optional but Recommended)

```bash
python character_review_tool.py work/detected_characters.json
```

Interactive options:
1. Display characters
2. Merge characters (e.g., "John" and "Johnny")
3. Edit character traits (gender, demeanor)
4. Remove false positives
5. Save and create voice config

Or review during conversion:

```bash
python main.py book.epub dummy.wav -o output.m4b --character-mode --review-characters
```

#### Step 3: Configure Voice Mappings

Edit `work/character_voices.json`:

```json
{
  "narrator_voice": {
    "speaker_audio": "voices/narrator.wav",
    "emotion_audio": null,
    "emotion_alpha": 0.7,
    "use_emo_text": true
  },
  "character_voices": {
    "John": {
      "speaker_audio": "voices/john_male_deep.wav",
      "emotion_audio": null,
      "emotion_alpha": 1.0,
      "use_emo_text": true
    },
    "Mary": {
      "speaker_audio": "voices/mary_female_soft.wav",
      "emotion_audio": "emotions/calm_female.wav",
      "emotion_alpha": 0.8,
      "use_emo_text": true
    }
  },
  "default_voice": {
    "speaker_audio": "voices/default.wav",
    "emotion_alpha": 0.8,
    "use_emo_text": true
  }
}
```

#### Step 4: Configure Emotion Library (Optional)

Edit `work/emotion_library.json`:

```json
{
  "happy": {"emotion_name": "happy", "audio_path": "emotions/happy.wav", "intensity": 1.0},
  "sad": {"emotion_name": "sad", "audio_path": "emotions/sad.wav", "intensity": 1.0},
  "angry": {"emotion_name": "angry", "audio_path": "emotions/angry.wav", "intensity": 1.0},
  "afraid": {"emotion_name": "afraid", "audio_path": "emotions/afraid.wav", "intensity": 1.0},
  "surprised": {"emotion_name": "surprised", "audio_path": "emotions/surprised.wav", "intensity": 1.0},
  "disgusted": {"emotion_name": "disgusted", "audio_path": "emotions/disgusted.wav", "intensity": 1.0},
  "calm": {"emotion_name": "calm", "audio_path": "emotions/calm.wav", "intensity": 0.8},
  "melancholic": {"emotion_name": "melancholic", "audio_path": "emotions/melancholic.wav", "intensity": 0.9}
}
```

#### Step 5: Generate Audiobook with Character Voices

```bash
python main.py book.epub dummy.wav -o audiobook.m4b \
  --format m4b \
  --character-mode \
  --character-config work/character_voices.json \
  --emotion-library work/emotion_library.json
```

### Character Mode Features

- **Automatic Detection**: Identifies characters based on proper nouns and context
- **Gender Analysis**: Detects male/female/neutral/unknown based on pronouns and context
- **Demeanor Analysis**: Identifies character personality traits (calm, energetic, serious, etc.)
- **Dialogue Detection**: Extracts quoted speech and attributes it to speakers
- **Thought Detection**: Identifies internal thoughts (typically in parentheses or italics)
- **Emotion Analysis**: Analyzes emotional content per segment using keyword matching
- **Dynamic Segmentation**: Creates segments based on:
  - Character changes (new speaker)
  - Emotion shifts (mood changes)
  - Content type (dialogue vs narration)

### Character Mode Command Line Options

```
--character-mode          Enable character-aware processing
--character-config FILE   Path to character voice configuration JSON
--emotion-library FILE    Path to emotion reference library JSON
--detect-characters       Detect characters and create config template (then exit)
--review-characters       Interactive character review before processing
```

### Character Analysis Output

In character mode, the tool creates additional files:

- `work/detected_characters.json` - Character data with traits and statistics
- `work/character_voices.json` - Voice mapping configuration
- `work/emotion_library.json` - Emotion reference library

## Workflow

### Standard Mode

1. **Extract**: Reads EPUB file and extracts text content
2. **Segment**: Splits text into segments (default: 500 words each)
3. **Process** (optional): Cleans text using Ollama LLM
4. **Generate**: Converts each segment to speech using IndexTTS2
5. **Merge**: Combines all audio segments into final audiobook

### Character-Aware Mode

1. **Extract**: Reads EPUB file and extracts text content with metadata
2. **Analyze Characters**: Detects characters, gender, demeanor, and dialogue patterns
3. **Review** (optional): Interactive character review and configuration
4. **Segment**: Creates character-aware segments based on:
   - Who is speaking/thinking
   - Emotional state
   - Content type (dialogue/thought/narration)
5. **Generate**: Converts each segment with appropriate:
   - Character voice
   - Emotion reference
   - Dynamic emotion vector
6. **Merge**: Combines all audio segments into final audiobook with metadata

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

## Batch Processing with Job Queue 🆕

For processing multiple audiobooks efficiently, use the job queue system:

### Quick Start

```bash
# Create jobs
python create_job.py book1.epub voice.wav -o book1.m4b --priority 10
python create_job.py book2.epub voice.wav -o book2.m4b --priority 5

# Process all jobs
python job_processor.py
```

### Interactive Job Creation

```bash
python create_job.py
```

### View Queue Status

```bash
# List pending jobs
python job_processor.py --list pending

# Check specific job
python job_processor.py --status <job-id>

# Cancel a pending job
python job_processor.py --cancel <job-id>
```

### Advanced Processing

```bash
# Process only 5 jobs
python job_processor.py --max-jobs 5

# Stop on first error
python job_processor.py --stop-on-error
```

**See `JOB_QUEUE_GUIDE.md` for complete documentation** and `JOB_QUEUE_QUICK_REFERENCE.md` for a command reference.

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

### Batch processing multiple books

```bash
# Create multiple jobs
python create_job.py book1.epub voice.wav -o book1.m4b --priority 10
python create_job.py book2.epub voice.wav -o book2.m4b --priority 10
python create_job.py book3.epub voice.wav -o book3.m4b --priority 5

# Process all at once
python job_processor.py
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
