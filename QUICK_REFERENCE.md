# Quick Reference: EPUB to Audiobook Converter

## Installation

```bash
pip install ebooklib beautifulsoup4 lxml requests tqdm
```

Optional:
- FFmpeg (for M4B format)
- Ollama (for text processing)

## Basic Commands

### Standard Single-Voice Audiobook

```bash
# WAV output
python main.py book.epub narrator.wav -o audiobook.wav

# M4B output (compressed)
python main.py book.epub narrator.wav -o audiobook.m4b --format m4b
```

### Character-Aware Multi-Voice Audiobook

```bash
# Step 1: Detect characters (with Ollama for better accuracy)
uv run -m main ./inputs/TheBet.epub ./inputs/voice-ref.mp3 -o output.m4b \
  --detect-characters --ollama-character-detection

# Step 1 Alternative: Detect characters (heuristic-based, faster)
uv run -m main ./inputs/TheBet.epub ./inputs/voice-ref.mp3 -o output.m4b --detect-characters --ollama-character-detection --keep-segments --ollama-model "aratan/DeepSeek-R1-32B-Uncensored:latest" --ollama-url "http://localhost:11434"

# Step 2: Review characters (optional)
uv run -m character_review_tool work/detected_characters.json

# Step 3: Edit configuration files
# - work/character_voices.json
# - work/emotion_library.json (optional)

# Step 4: Generate audiobook
uv run main ./inputs/TheBet.epub ./inputs/voice-ref.mp3 -o audiobook.m4b \
  --format m4b \
  --character-mode \
  --character-config work/character_voices.json \
  --emotion-library work/emotion_library.json
```

## Common Options

### Output
- `--format {wav,m4b}` - Output format (default: wav)
- `--keep-segments` - Keep individual audio segments
- `--work-dir DIR` - Working directory (default: ./work)

### Text Processing
- `--segment-words N` - Words per segment (default: 500)
- `--max-words N` - Maximum words per segment (default: 600)
- `--use-ollama` - Use Ollama for text cleanup

### Character Mode
- `--character-mode` - Enable character-aware processing
- `--character-config FILE` - Character voice configuration
- `--emotion-library FILE` - Emotion reference library
- `--detect-characters` - Detect and create config template
- `--review-characters` - Interactive character review
- `--ollama-character-detection` - Use Ollama for more accurate character detection (recommended)

### TTS Settings
- `--emo-audio FILE` - Emotion reference audio
- `--emo-alpha VALUE` - Emotion strength (0.0-1.0)
- `--use-emo-text` - Auto-detect emotion from text
- `--temperature VALUE` - Generation temperature (default: 0.8)
- `--use-fp16` - Use FP16 for faster generation

### Audio Settings
- `--interval-silence MS` - Silence between sentences (default: 200ms)
- `--segment-silence MS` - Silence between segments (default: 500ms)

## Configuration Files

### Character Voice Mapping (`character_voices.json`)

```json
{
  "narrator_voice": {
    "speaker_audio": "voices/narrator.wav",
    "emotion_alpha": 0.7,
    "use_emo_text": true
  },
  "character_voices": {
    "Alice": {
      "speaker_audio": "voices/alice.wav",
      "emotion_alpha": 1.0,
      "use_emo_text": true
    },
    "Bob": {
      "speaker_audio": "voices/bob.wav",
      "emotion_audio": "emotions/calm.wav",
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

### Emotion Library (`emotion_library.json`)

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

## Character Review Tool Commands

```bash
# Run interactive review
python character_review_tool.py work/detected_characters.json

# Options in review tool:
# 1. Display characters
# 2. Merge characters (combine duplicates)
# 3. Edit character traits (gender, demeanor)
# 4. Remove characters (false positives)
# 5. Save and create voice config
# 6. Exit without saving
```

## Test Scripts

```bash
# Test character detection
python test_character_detection.py

# Test individual modules
python character_analyzer.py
python character_segmenter.py
python character_voice_config.py
```

## Typical Workflows

### Workflow 1: Simple Audiobook
```bash
python main.py book.epub voice.wav -o book.m4b --format m4b --use-fp16
```

### Workflow 2: Multi-Voice with Auto-Detection
```bash
# Detect
python main.py book.epub voice.wav -o book.m4b --detect-characters

# Configure (edit JSON files)

# Generate
python main.py book.epub voice.wav -o book.m4b \
  --format m4b --character-mode \
  --character-config work/character_voices.json
```

### Workflow 3: Full Interactive Workflow
```bash
# Detect and review in one go
python main.py book.epub voice.wav -o book.m4b \
  --character-mode --review-characters

# After review, configure voices, then re-run:
python main.py book.epub voice.wav -o book.m4b \
  --format m4b --character-mode \
  --character-config work/character_voices.json \
  --emotion-library work/emotion_library.json
```

## Character Detection Methods

### Heuristic Detection (Default)
- Fast, no external dependencies
- Looks for capitalized words appearing 3+ times
- Gender detection based on pronouns in context
- Demeanor based on emotion keywords
- **May miss**: Characters with few appearances, detect false positives

### Ollama Detection (Recommended)
Use `--ollama-character-detection` for:
- **Context-aware analysis**: Understands narrative context
- **Better accuracy**: Distinguishes characters from places/things
- **Smarter gender detection**: Based on character description and role
- **Personality insights**: More nuanced demeanor detection
- **Pronoun resolution**: Connects "he/she/I" to actual characters
- **Fewer false positives**: Filters out non-character entities

**Example:**
```bash
# Standard detection
python main.py book.epub voice.wav -o book.m4b --detect-characters

# Ollama-enhanced detection (better results)
python main.py book.epub voice.wav -o book.m4b \
  --detect-characters --ollama-character-detection
```

### Pronoun Resolution
The system automatically:
- Tracks last mentioned character per gender
- Maps "he/she/I" pronouns to character names
- Resolves unnamed dialogue/thoughts to speakers
- Uses context from previous sentences

**Example text:**
> "John walked into the room. He saw Mary sitting by the window."

- "He" is resolved to "John" (last mentioned male character)
- Dialogue following "He said" is attributed to John

## Supported Emotions

1. **happy** - joy, delight, cheerful, smiled, laughed
2. **sad** - sorrow, grief, depressed, cried, tears
3. **angry** - rage, furious, irritated, shouted
4. **afraid** - fear, scared, terrified, anxious
5. **surprised** - shocked, amazed, astonished
6. **disgusted** - revolted, repulsed, grimaced
7. **calm** - peaceful, serene, gentle
8. **melancholic** - wistful, nostalgic, pensive

## Troubleshooting

### FFmpeg not found
```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

### Character not detected
Edit `work/detected_characters.json` manually and add the character.

### Wrong gender detected
Use `--review-characters` or edit JSON file directly.

### Voice file not found
Check paths in `character_voices.json` are correct relative to where you run the command.

### Out of memory
- Use `--use-fp16`
- Reduce `--max-words` (try 400)
- Process smaller chunks

### Segments too short
- Increase `--min-words` (try 200)
- Characters will be merged if similar and under max_words

## Performance Tips

1. **Use FP16**: `--use-fp16` for 2x faster generation
2. **Adjust segments**: Larger segments = fewer iterations
3. **M4B format**: Use for 90% size reduction
4. **Character mode**: Process small books first to test
5. **Keep segments**: `--keep-segments` for debugging only

## File Structure

```
project/
├── inputs/
│   └── book.epub
├── voices/
│   ├── narrator.wav
│   ├── alice.wav
│   └── bob.wav
├── emotions/          # Optional
│   ├── happy.wav
│   ├── sad.wav
│   └── ...
├── work/              # Auto-created
│   ├── detected_characters.json
│   ├── character_voices.json
│   ├── emotion_library.json
│   └── segments/
└── outputs/
    ├── audiobook.m4b
    └── audiobook_metadata.txt
```

## Output Files

### Standard Mode
- `audiobook.wav` or `audiobook.m4b` - Final audiobook
- `audiobook_metadata.txt` - Metadata text file

### Character Mode (Additional)
- `work/detected_characters.json` - Character analysis
- `work/character_voices.json` - Voice configuration
- `work/emotion_library.json` - Emotion references

### With Ollama
- `work/ollama/processing_summary.txt`
- `work/ollama/segment_XXXX_comparison.txt`

## Quick Checks

### Verify Installation
```bash
python main.py --help
```

### Test Character Detection
```bash
python test_character_detection.py
```

### Check FFmpeg
```bash
ffmpeg -version
```

## Getting Help

1. Check `README.md` for full documentation
2. Check `CHARACTER_MODE_GUIDE.md` for detailed character mode guide
3. Check `FEATURE_SUMMARY.md` for feature overview
4. Run with `--verbose` flag for detailed output
5. Test with small EPUB first

## Version Info

- Standard mode: Single voice, WAV/M4B output
- Character mode: Multi-voice, emotion-aware, WAV/M4B output
- Both modes: Fully backwards compatible
