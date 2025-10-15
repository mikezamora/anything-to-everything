# Feature Summary: Character-Aware Audiobook Generation + M4B Support

## Overview

Two major features have been added to the EPUB to Audiobook converter:

1. **M4B Format Support** - Compressed audiobook format with embedded metadata
2. **Character-Aware Processing** - Multi-voice audiobooks with automatic character detection and emotion analysis

## New Features

### 1. M4B Format Support ✅

**What it does:**
- Converts final audiobook to M4B format (AAC compression in MP4 container)
- Embeds metadata from EPUB (title, author, album, genre, publisher, etc.)
- Produces files 10-15x smaller than WAV
- Standard audiobook format recognized by all major players

**How to use:**
```bash
python main.py book.epub speaker.wav -o audiobook.m4b --format m4b
```

**Requirements:**
- FFmpeg installed on system
- Automatically converts WAV to M4B after generation

**Metadata embedded:**
- Title, Author, Album (series if available)
- Genre (Audiobook)
- Publisher, Date
- Description, Comments

### 2. Character-Aware Processing ✅

**What it does:**
- Automatically detects characters in the story
- **Ollama integration** for context-aware character detection 🆕
- Analyzes gender (male/female/neutral/unknown) and demeanor
- **Pronoun resolution** - connects "he/she/I" to character names 🆕
- Identifies dialogue, thoughts, and narration
- Detects emotional state per segment (8 emotions supported)
- Generates audio with character-specific voices
- Applies dynamic emotions based on text analysis

**Character Detection Methods:**
1. **Heuristic** (default) - Fast, pattern-based, good for simple texts
2. **Ollama-enhanced** (recommended) - Context-aware, more accurate, filters false positives

**Workflow:**

#### Step 1: Detect Characters
```bash
# Recommended: Use Ollama for better accuracy
python main.py book.epub dummy.wav -o output.m4b \
  --detect-characters --ollama-character-detection

# Alternative: Heuristic detection (faster)
python main.py book.epub dummy.wav -o output.m4b --detect-characters
```

Creates:
- `work/detected_characters.json` - Character data
- `work/character_voices_template.json` - Voice mapping template
- `work/emotion_library_template.json` - Emotion reference template

**Ollama Benefits:**
- ✅ Context-aware: Understands story narrative
- ✅ Filters false positives: Distinguishes characters from places
- ✅ Better traits: More accurate gender and demeanor detection
- ✅ Pronoun resolution: Links "he/she/I" to actual characters
- ✅ Fewer errors: Smarter character identification

#### Step 2: Review Characters (Optional)
```bash
python character_review_tool.py work/detected_characters.json
```

Or integrate into generation:
```bash
python main.py book.epub dummy.wav -o output.m4b --review-characters
```

Interactive options:
- Display all detected characters
- Merge characters (nicknames, OCR errors)
- Edit character traits (gender, demeanor)
- Remove false positives

#### Step 3: Configure Voices

Edit `work/character_voices.json`:
```json
{
  "narrator_voice": {
    "speaker_audio": "voices/narrator.wav",
    "emotion_alpha": 0.7,
    "use_emo_text": true
  },
  "character_voices": {
    "John": {
      "speaker_audio": "voices/john.wav",
      "emotion_alpha": 1.0,
      "use_emo_text": true
    },
    "Mary": {
      "speaker_audio": "voices/mary.wav",
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

#### Step 4: Generate with Character Voices
```bash
python main.py book.epub dummy.wav -o audiobook.m4b \
  --format m4b \
  --character-mode \
  --character-config work/character_voices.json \
  --emotion-library work/emotion_library.json
```

## New Files Created

### Core Modules

1. **`character_analyzer.py`** - Character detection and analysis
   - `CharacterAnalyzer` class
   - Gender and demeanor detection
   - Dialogue/thought extraction
   - Emotion analysis
   - Character merging

2. **`character_segmenter.py`** - Character-aware text segmentation
   - `CharacterAwareSegmenter` class
   - Segments by character and emotion
   - Dynamic segment creation based on speaker changes

3. **`character_voice_config.py`** - Configuration system
   - `VoiceConfig` - Per-character voice settings
   - `CharacterVoiceMapping` - Complete voice mapping
   - `EmotionLibrary` - Emotion reference library
   - Template creation utilities

4. **`character_review_tool.py`** - Interactive character review
   - `CharacterReviewTool` class
   - Interactive merging, editing, removing
   - Configuration file generation

### Documentation

5. **`CHARACTER_MODE_GUIDE.md`** - Comprehensive guide
   - Complete workflow walkthrough
   - Configuration examples
   - Troubleshooting tips
   - Best practices

### Updated Files

6. **`main.py`** - Enhanced with character support
   - New command-line arguments for character mode
   - Character detection and review integration
   - Character-aware TTS processing
   - M4B format support

7. **`audio_merger.py`** - M4B conversion support
   - FFmpeg integration
   - Metadata embedding in M4B
   - Automatic compression

8. **`epub_extractor.py`** - Enhanced metadata
   - Album field extraction (series support)
   - Publisher, date, description extraction
   - Comprehensive metadata for M4B

9. **`README.md`** - Updated documentation
   - Character mode overview
   - M4B format documentation
   - New command-line options

10. **`config_template.py`** - Added format option

## New Command-Line Options

### M4B Format
```
--format {wav,m4b}        Output format (default: wav)
```

### Character Detection
```
--character-mode                Enable character-aware processing
--character-config FILE         Path to character voice configuration JSON
--emotion-library FILE          Path to emotion reference library JSON
--detect-characters             Detect characters and create config template
--review-characters             Interactive character review before processing
--ollama-character-detection    Use Ollama for advanced character detection (recommended)
```

## Data Structures

### CharacterTraits
```python
@dataclass
class CharacterTraits:
    name: str
    gender: str  # male, female, neutral, unknown
    demeanor: str  # calm, energetic, serious, etc.
    appearances: int
    dialogue_count: int
    thought_count: int
    first_appearance: int
```

### EmotionalState
```python
@dataclass
class EmotionalState:
    dominant_emotion: str  # happy, sad, angry, afraid, etc.
    intensity: float  # 0.0 to 1.0
    emotions: Dict[str, float]  # 8-emotion vector
```

### CharacterSegment
```python
@dataclass
class CharacterSegment:
    segment_id: int
    text: str
    character: Optional[str]
    is_dialogue: bool
    is_thought: bool
    is_narration: bool
    emotional_state: EmotionalState
```

## Emotion Detection

### Supported Emotions
1. **happy** - joy, delight, cheerful, smiled, laughed
2. **sad** - sorrow, grief, depressed, cried, tears
3. **angry** - rage, furious, irritated, shouted, yelled
4. **afraid** - fear, scared, terrified, anxious, trembled
5. **surprised** - shocked, amazed, astonished, gasped
6. **disgusted** - revolted, repulsed, grimaced
7. **calm** - peaceful, serene, tranquil, gentle
8. **melancholic** - wistful, nostalgic, pensive, somber

### Emotion Vector Format
IndexTTS2 format: `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`

## Character Detection

### How Characters are Detected
1. **Proper noun analysis** - Capitalized words appearing 3+ times
2. **Context filtering** - Removes common words, place names
3. **Gender detection** - Based on pronouns in character's context
4. **Demeanor inference** - Based on emotion keywords in character's sentences

### Dialogue/Thought Patterns
- **Dialogue**: `"text"`, `"text"`, `«text»`, `„text"`
- **Thoughts**: `*text*`, `_text_`, `(text)`
- **Speaker attribution**: "said John", "John said", "John: "text""

## Integration with TTS

### Standard Mode
- Single voice for entire audiobook
- Optional emotion reference
- Text-based emotion detection

### Character Mode
- Different voice per character
- Narrator voice for narration
- Per-segment emotion detection
- Character-specific emotion references
- Dynamic emotion vectors

## Example Workflows

### Simple Novel (Standard + M4B)
```bash
python main.py novel.epub narrator.wav -o novel.m4b --format m4b
```

### Multi-Voice Audiobook
```bash
# 1. Detect
python main.py book.epub dummy.wav -o book.m4b --detect-characters

# 2. Edit work/character_voices_template.json

# 3. Generate
python main.py book.epub dummy.wav -o book.m4b \
  --format m4b \
  --character-mode \
  --character-config work/character_voices.json
```

### Full Character Mode with Emotions
```bash
python main.py book.epub dummy.wav -o book.m4b \
  --format m4b \
  --character-mode \
  --character-config work/character_voices.json \
  --emotion-library work/emotion_library.json \
  --use-fp16
```

## Performance Considerations

### M4B Conversion
- Adds ~30-60 seconds for compression (depends on length)
- Reduces file size by 90-95%
- Requires FFmpeg installed

### Character Mode
- Processes segments individually (may be slower)
- More segments = more processing time
- GPU memory usage similar to standard mode
- Use `--use-fp16` for faster generation

## Testing Recommendations

1. **Test M4B conversion:**
   ```bash
   python main.py small_book.epub voice.wav -o test.m4b --format m4b
   ```

2. **Test character detection:**
   ```bash
   python main.py book.epub voice.wav -o test.m4b --detect-characters
   ```

3. **Test character review tool:**
   ```bash
   python character_review_tool.py work/detected_characters.json
   ```

4. **Test full character mode:**
   - Use a short book first (~10 pages)
   - Create voice files for 2-3 characters
   - Run with `--verbose` flag

## Known Limitations

1. **Character Detection:**
   - May miss characters with uncommon names
   - Can include place names or false positives
   - Gender detection is heuristic-based
   - Use review tool to refine

2. **Dialogue Attribution:**
   - Works best with standard quote formats
   - May miss complex dialogue structures
   - Thought detection depends on formatting

3. **Emotion Detection:**
   - Keyword-based (not deep semantic analysis)
   - Works best with explicit emotional language
   - Can be improved with Ollama integration (future)

4. **M4B Format:**
   - Requires FFmpeg installation
   - Chapter markers not yet supported (future feature)
   - Some metadata fields may not display on all players

## Future Enhancements

### Planned
- [ ] Ollama integration for better character/emotion detection
- [ ] Chapter markers in M4B format
- [ ] Vocal characteristic tuning (pitch, speed) per character
- [ ] Visual character configuration GUI
- [ ] Multi-language support
- [ ] Advanced dialogue attribution with ML

### Possible
- [ ] Character voice consistency validation
- [ ] Automatic voice selection based on traits
- [ ] Scene-based emotion analysis
- [ ] Custom emotion profiles per character
- [ ] Audio quality validation

## Migration from Previous Version

### For Existing Users

No breaking changes! Standard mode works exactly as before:
```bash
python main.py book.epub speaker.wav -o audiobook.wav
```

### To Use New Features

1. **Just want M4B:** Add `--format m4b`
2. **Want character mode:** Follow character detection workflow
3. **Both:** Combine options

## Dependencies

### New Requirements
- None! All new dependencies are Python standard library or already required

### Optional Requirements
- **FFmpeg** - For M4B format support (not required for WAV output)

## Support and Documentation

- **README.md** - Quick start and feature overview
- **CHARACTER_MODE_GUIDE.md** - Complete character mode walkthrough
- **Code comments** - Inline documentation in all modules
- **Example configurations** - Templates created by `--detect-characters`

## Summary

You now have a powerful audiobook converter with:
✅ M4B format support with metadata embedding
✅ Automatic character detection
✅ Multi-voice support with character-specific voices
✅ Dynamic emotion detection and application
✅ Interactive character review tool
✅ Comprehensive configuration system
✅ Flexible workflow (simple or advanced)

The system is designed to work at multiple levels:
- **Basic:** Standard single-voice with M4B output
- **Intermediate:** Character detection and review
- **Advanced:** Full character mode with emotions and custom voices

All features are optional and backwards compatible!
