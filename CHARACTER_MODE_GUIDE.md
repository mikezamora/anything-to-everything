# Character-Aware Audiobook Generation Guide

## Overview

Character-aware mode transforms your audiobook by:
- Detecting all characters in your story
- Analyzing their traits (gender, demeanor)
- Identifying dialogue, thoughts, and narration
- Analyzing emotional states throughout the text
- Generating audio with character-specific voices and emotions

## Complete Workflow

### 1. Character Detection

First, run character detection on your EPUB file:

```bash
# Recommended: Use Ollama for better accuracy
python main.py mybook.epub placeholder.wav -o output.m4b \
  --detect-characters --ollama-character-detection

# Alternative: Heuristic detection (faster, less accurate)
python main.py mybook.epub placeholder.wav -o output.m4b --detect-characters
```

**What happens:**

#### With Ollama (--ollama-character-detection):
- **Context-aware analysis**: LLM understands narrative context
- **Accurate character identification**: Distinguishes characters from places/objects
- **Better gender detection**: Based on character description and role
- **Nuanced demeanor**: More sophisticated personality analysis
- **Pronoun resolution**: Connects "he/she/I" to actual character names
- **Fewer false positives**: Filters out non-character entities
- Creates three configuration files

#### Without Ollama (heuristic):
- Analyzes text for character names (capitalized words appearing 3+ times)
- Detects gender based on pronouns (he/she/him/her) in context
- Infers demeanor from emotional keywords associated with character
- May include false positives (place names, etc.)
- Creates three configuration files

**Output files:**
```
work/
├── detected_characters.json         # Character data
├── character_voices_template.json   # Voice mapping template
└── emotion_library_template.json    # Emotion reference template
```

**Comparison Example:**

Given text: "John entered the room. London was beautiful this time of year. He greeted Mary warmly."

| Method | Characters Detected | Gender | Notes |
|--------|-------------------|--------|-------|
| Heuristic | John, London, Mary | John: male, London: unknown, Mary: female | Incorrectly includes "London" |
| Ollama | John, Mary | John: male, Mary: female | Correctly filters out "London", resolves "He" to John |

**When to use Ollama:**
- ✅ Complex narratives with many characters
- ✅ Stories with locations that look like names
- ✅ Books with pronoun-heavy dialogue
- ✅ First-person narratives
- ✅ When accuracy is critical

**When heuristic is enough:**
- ✅ Simple stories with few characters
- ✅ Clear character naming conventions
- ✅ Quick testing/prototyping
- ✅ Ollama not available

### 2. Review Characters (Recommended)

Use the interactive review tool to refine character detection:

```bash
python character_review_tool.py work/detected_characters.json
```

**Review options:**

#### Display Characters
See all detected characters with their traits:
```
1. John
   Gender: male
   Demeanor: calm
   Appearances: 45
   Dialogue: 23, Thoughts: 5

2. Mary
   Gender: female
   Demeanor: nervous
   Appearances: 38
   Dialogue: 31, Thoughts: 4
```

#### Merge Characters
Combine characters that refer to the same person:
- Nicknames: "Elizabeth" and "Liz"
- Titles: "Mr. Smith" and "Smith"
- OCR errors: "John" and "Jobn"

```
Enter primary character name: John
Enter character to merge: Johnny
✓ Merged 'Johnny' into 'John'
```

#### Edit Traits
Correct gender or demeanor if auto-detection is wrong:
```
Enter character name: Alex
New gender (male/female/neutral/unknown) [unknown]: neutral
New demeanor [calm]: energetic
✓ Updated traits for Alex
```

#### Remove False Positives
Remove detected "characters" that aren't actually characters:
- Place names: "London", "Paris"
- Common words that got capitalized: "Chapter", "Part"

```
Enter character name to remove: London
Remove 'London'? (y/n): y
✓ Removed 'London'
```

### 3. Prepare Voice Files

Create a directory structure for your voice files:

```
voices/
├── narrator.wav          # Narrator voice
├── john_male.wav         # John's voice
├── mary_female.wav       # Mary's voice
├── sarah_female.wav      # Sarah's voice
└── default.wav           # Fallback voice

emotions/                 # Optional: emotion references
├── happy.wav
├── sad.wav
├── angry.wav
├── afraid.wav
├── surprised.wav
├── disgusted.wav
├── calm.wav
└── melancholic.wav
```

**Voice file requirements:**
- Format: WAV (recommended) or any format supported by torchaudio
- Length: 3-10 seconds is ideal
- Quality: Clean, clear speech without background noise
- Content: Natural speech in the character's intended style

### 4. Configure Voice Mapping

Edit `work/character_voices.json` (or rename from template):

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
      "speaker_audio": "voices/john_male.wav",
      "emotion_audio": null,
      "emotion_alpha": 1.0,
      "use_emo_text": true
    },
    "Mary": {
      "speaker_audio": "voices/mary_female.wav",
      "emotion_audio": "emotions/calm_female.wav",
      "emotion_alpha": 0.8,
      "use_emo_text": true
    },
    "Sarah": {
      "speaker_audio": "voices/sarah_female.wav",
      "emotion_audio": null,
      "emotion_alpha": 1.0,
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

**Configuration fields:**
- `speaker_audio`: Voice reference for this character (required)
- `emotion_audio`: Optional emotion reference for this character
- `emotion_alpha`: Emotion blend strength (0.0-1.0)
  - 0.0 = no emotion
  - 0.5 = subtle emotion
  - 1.0 = full emotion
- `use_emo_text`: Auto-detect emotion from text (recommended: true)

### 5. Configure Emotion Library (Optional)

Edit `work/emotion_library.json`:

```json
{
  "happy": {
    "emotion_name": "happy",
    "audio_path": "emotions/happy.wav",
    "intensity": 1.0
  },
  "sad": {
    "emotion_name": "sad",
    "audio_path": "emotions/sad.wav",
    "intensity": 1.0
  },
  "angry": {
    "emotion_name": "angry",
    "audio_path": "emotions/angry.wav",
    "intensity": 1.0
  },
  "afraid": {
    "emotion_name": "afraid",
    "audio_path": "emotions/afraid.wav",
    "intensity": 1.0
  },
  "surprised": {
    "emotion_name": "surprised",
    "audio_path": "emotions/surprised.wav",
    "intensity": 1.0
  },
  "disgusted": {
    "emotion_name": "disgusted",
    "audio_path": "emotions/disgusted.wav",
    "intensity": 1.0
  },
  "calm": {
    "emotion_name": "calm",
    "audio_path": "emotions/calm.wav",
    "intensity": 0.8
  },
  "melancholic": {
    "emotion_name": "melancholic",
    "audio_path": "emotions/melancholic.wav",
    "intensity": 0.9
  }
}
```

**How emotion detection works:**
1. Text is analyzed for emotion keywords (happy, sad, angry, etc.)
2. Dominant emotion is identified
3. Emotion vector is created (8 values: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm)
4. If emotion library is provided, corresponding audio reference is used
5. If character has specific emotion_audio set, that overrides library

### 6. Generate Audiobook

Run the full conversion with character mode:

```bash
python main.py book.epub dummy.wav -o audiobook.m4b \
  --format m4b \
  --character-mode \
  --character-config work/character_voices.json \
  --emotion-library work/emotion_library.json \
  --max-words 400 \
  --use-fp16
```

**What happens:**
1. EPUB text is extracted
2. Characters are loaded from config
3. Text is segmented by:
   - Character (who is speaking/thinking)
   - Emotion (mood shifts)
   - Content type (dialogue/thought/narration)
4. Each segment is processed with:
   - Appropriate character voice
   - Detected emotion
   - Proper emotion intensity
5. Audio segments are merged into M4B with metadata

**Processing details:**
```
[1/6] Extracting text from EPUB...
  Title: My Book
  Author: Author Name
  
[2/6] Analyzing characters...
  Detected 3 characters:
    - John: male, calm (45 appearances)
    - Mary: female, nervous (38 appearances)
    - Sarah: female, energetic (22 appearances)
  ✓ Loaded voice configuration
  ✓ Loaded emotion library
  
[3/6] Creating character-aware segments...
  Created 156 character-aware segments
  Dialogue: 89, Thoughts: 12, Narration: 55
  Characters in text: John, Mary, Sarah
  
[5/6] Initializing IndexTTS2...
  
[6/6] Generating audio with IndexTTS2...
  Processing 156 character-aware segments...
  [1/156] John (Dialogue, calm)
  [2/156] NARRATOR (Narration, calm)
  [3/156] Mary (Dialogue, afraid)
  ...
```

## Advanced Usage

### One-Step Review and Generate

Review characters interactively, then immediately generate:

```bash
python main.py book.epub dummy.wav -o audiobook.m4b \
  --character-mode \
  --review-characters \
  --format m4b
```

After review, the tool will:
1. Save reviewed characters
2. Create voice config template
3. **Pause and ask you to configure voices**
4. Continue with generation

### Without Emotion Library

You can use character mode without an emotion library:

```bash
python main.py book.epub dummy.wav -o audiobook.m4b \
  --character-mode \
  --character-config work/character_voices.json \
  --format m4b
```

Emotions will still be detected and applied via emotion vectors, but without reference audio.

### Narrator-Only Mode

If you only want different voices for characters vs narrator:

```json
{
  "narrator_voice": {
    "speaker_audio": "voices/narrator.wav",
    "use_emo_text": true
  },
  "character_voices": {},
  "default_voice": {
    "speaker_audio": "voices/character_default.wav",
    "use_emo_text": true
  }
}
```

All characters will use `default_voice`, separate from narrator.

## Troubleshooting

### Character Not Detected

**Problem:** A character wasn't detected automatically.

**Solution:** Manually add to `detected_characters.json`:

```json
{
  "Alex": {
    "name": "Alex",
    "gender": "neutral",
    "demeanor": "calm",
    "appearances": 10,
    "dialogue_count": 5,
    "thought_count": 2,
    "first_appearance": -1
  }
}
```

Then add to voice config.

### Wrong Gender Detected

**Problem:** Character gender is incorrect.

**Solution:** Use review tool to edit, or manually edit `reviewed_characters.json`.

### Too Many Segments

**Problem:** Character mode creates too many short segments.

**Solution:** 
- Increase `--max-words` (e.g., 800)
- The segmenter will merge similar consecutive segments

### Character Voice Not Found

**Problem:** Error: "Speaker audio file not found"

**Solution:** Check paths in `character_voices.json` are relative to where you run the command.

### Emotion Not Applied

**Problem:** Emotion seems flat/not working.

**Solution:**
- Increase `emotion_alpha` in voice config (try 1.0)
- Ensure `use_emo_text` is true
- Verify emotion reference audio files exist and are good quality

## Best Practices

### Voice Selection

1. **Narrator:** Choose a neutral, clear voice that's easy to listen to for long periods
2. **Characters:** Select voices that match the character traits:
   - Age appropriate
   - Gender appropriate (or neutral if ambiguous)
   - Energy level matches demeanor
3. **Contrast:** Ensure character voices are distinct from each other
4. **Consistency:** Use the same voice actor for all recordings if possible

### Emotion References

1. **Quality over quantity:** Better to have 3-4 good emotion references than 8 poor ones
2. **Match voice:** Emotion references should ideally be from the same voice as the character
3. **Clear expression:** Emotion should be obvious in the reference audio
4. **No speech:** Pure emotional vocalizations work well (laughs, sighs, gasps)

### Segment Length

- **Standard dialogue:** 200-400 words works well
- **Action scenes:** Shorter segments (100-200 words) for dynamic pacing
- **Descriptive narration:** Longer segments (400-600 words) are fine

### Review Process

1. Always review detected characters
2. Merge obvious duplicates first
3. Remove false positives
4. Edit traits for important characters
5. Save and test with a short book first

## Examples

### Simple Novel (2-3 Main Characters)

```bash
# 1. Detect
python main.py novel.epub dummy.wav -o novel.m4b --detect-characters

# 2. Review
python character_review_tool.py work/detected_characters.json

# 3. Configure (edit character_voices.json)
# 4. Generate
python main.py novel.epub dummy.wav -o novel.m4b \
  --character-mode \
  --character-config work/character_voices.json \
  --format m4b
```

### Complex Story (Many Characters)

```bash
# 1. Detect and review together
python main.py story.epub dummy.wav -o story.m4b \
  --character-mode \
  --review-characters

# Interactive review...
# Configure voice files...

# 2. Generate with emotion library
python main.py story.epub dummy.wav -o story.m4b \
  --character-mode \
  --character-config work/character_voices.json \
  --emotion-library work/emotion_library.json \
  --format m4b \
  --use-fp16
```

### Narrator-Only with Emotion

```bash
# Standard mode with emotion detection
python main.py book.epub narrator.wav -o book.m4b \
  --format m4b \
  --emo-audio emotions/neutral.wav \
  --use-emo-text \
  --emo-alpha 0.7
```

## Configuration Reference

### Character Voice Config Schema

```json
{
  "narrator_voice": {
    "speaker_audio": "string (required)",
    "emotion_audio": "string (optional, null)",
    "emotion_alpha": "float 0.0-1.0 (default: 1.0)",
    "use_emo_text": "boolean (default: true)"
  },
  "character_voices": {
    "CharacterName": {
      "speaker_audio": "string (required)",
      "emotion_audio": "string (optional, null)",
      "emotion_alpha": "float 0.0-1.0 (default: 1.0)",
      "use_emo_text": "boolean (default: true)"
    }
  },
  "default_voice": {
    "speaker_audio": "string (required)",
    "emotion_audio": "string (optional, null)",
    "emotion_alpha": "float 0.0-1.0 (default: 0.8)",
    "use_emo_text": "boolean (default: true)"
  }
}
```

### Emotion Library Schema

```json
{
  "emotion_name": {
    "emotion_name": "string (matches key)",
    "audio_path": "string (required)",
    "intensity": "float 0.0-1.0 (default: 1.0)"
  }
}
```

### Character Data Schema

```json
{
  "CharacterName": {
    "name": "string",
    "gender": "male|female|neutral|unknown",
    "demeanor": "string",
    "appearances": "integer",
    "dialogue_count": "integer",
    "thought_count": "integer",
    "first_appearance": "integer"
  }
}
```

## Performance Notes

- Character mode processes segments individually, which may be slower
- Use `--use-fp16` for faster generation
- GPU memory usage is similar to standard mode
- M4B compression significantly reduces final file size

## Future Enhancements

Planned features:
- Ollama integration for better character/emotion detection
- Chapter markers in M4B based on characters
- Vocal characteristic tuning per character (pitch, speed)
- Multi-language character support
- Visual character configuration GUI

## Support

For issues or questions:
1. Check troubleshooting section above
2. Ensure all voice files exist and are accessible
3. Test with a small EPUB first
4. Review character detection output carefully
