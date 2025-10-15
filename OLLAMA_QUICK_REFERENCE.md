# Quick Reference: Ollama Segmented Character Detection

## What Changed?

The Ollama character detection now processes books in **overlapping segments** to maintain context while staying within LLM limits.

## Key Features

### 1. Automatic `<think>` Tag Removal
- Removes `<think>...</think>` blocks before processing
- Cleans Ollama responses as well
- No action needed - happens automatically

### 2. Segmented Processing
- Text split into ~3000 character segments
- Processed with overlapping windows:
  ```
  Window 1: Segments [1, 2]
  Window 2: Segments [1, 2, 3]
  Window 3: Segments [2, 3, 4]
  Window 4: Segments [3, 4, 5]
  ...
  ```
- Results intelligently merged

### 3. Complete Artifact Saving
All processing details saved to `work/character_detection/`:
- `prompts/` - What we asked Ollama
- `inputs/` - What text we sent (cleaned)
- `outputs/` - What Ollama returned
- `comparisons/` - Human-readable summaries
- `processing_summary.txt` - Overall results

## Commands

### Basic Usage (Unchanged)
```bash
python main.py book.epub voice.wav -o output.m4b \
  --detect-characters --ollama-character-detection
```

### Custom Configuration
```python
from character_analyzer import CharacterAnalyzer

analyzer = CharacterAnalyzer(
    use_ollama=True,
    work_dir="./work",      # Artifact location
    segment_words=500       # ~3000 chars per segment
)
```

## Testing

### Quick Test
```bash
python test_ollama_segmented.py
```

### What It Tests
1. ✅ `<think>` tag removal
2. ✅ Text segmentation
3. ✅ Full Ollama detection (if running)
4. ✅ Artifact creation

## Reviewing Results

### Check Processing Summary
```bash
cat work/character_detection/processing_summary.txt
```

Example output:
```
Character Detection Processing Summary
======================================

Total segments processed: 12
Successful analyses: 12
Final merged characters: 5

Detected Characters:
  - John (male)
    Demeanor: serious
    Appearances: 45
  - Mary (female)
    Demeanor: cheerful
    Appearances: 38
  ...
```

### Review Individual Segments
```bash
# See what was detected in segment 1
cat work/character_detection/comparisons/segment_0001_comparison.txt

# See the prompt used
cat work/character_detection/prompts/segment_0001_prompt.txt

# See the raw response
cat work/character_detection/outputs/segment_0001_output.json
```

## Workflow

```
1. Run detection
   └─> python main.py book.epub voice.wav --detect-characters --ollama-character-detection

2. Check summary
   └─> cat work/character_detection/processing_summary.txt

3. Review characters (if needed)
   └─> python character_review_tool.py

4. Generate audiobook
   └─> python main.py book.epub voice.wav --character-mode
```

## Troubleshooting

### Ollama Not Available
- **Symptom**: "Ollama not available at http://localhost:11434"
- **Solution**: 
  ```bash
  # Start Ollama
  ollama serve
  
  # Verify it's running
  curl http://localhost:11434/api/tags
  ```

### JSON Parsing Errors
- **Symptom**: "Failed to parse JSON for segments_X-Y"
- **Solution**: Check the comparison file for that segment:
  ```bash
  cat work/character_detection/comparisons/segment_000X_comparison.txt
  ```
- System automatically falls back to heuristic detection

### Missing Characters
- **Solution**: System combines Ollama + heuristic results
- Check `processing_summary.txt` to see if character was detected
- Heuristic catches characters with 5+ appearances

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_ollama` | `False` | Enable Ollama detection |
| `ollama_url` | `http://localhost:11434` | Ollama server URL |
| `ollama_model` | `llama2` | Model to use |
| `work_dir` | `./work` | Artifact storage location |
| `segment_words` | `500` | Segment size (~3000 chars) |

## Performance

- **Processing Time**: ~2-5 seconds per segment (depends on model)
- **Segment Count**: Varies by book length
  - Short story (5K words): ~3-5 segments
  - Novella (20K words): ~10-15 segments
  - Novel (100K words): ~40-60 segments
- **Timeout**: 120 seconds per segment

## Benefits vs. Previous Version

| Old Approach | New Segmented Approach |
|--------------|------------------------|
| ❌ Couldn't process large books | ✅ Handles any book size |
| ❌ No artifact saving | ✅ Complete debugging artifacts |
| ❌ `<think>` tags confused LLM | ✅ Automatically removed |
| ❌ Lost context at boundaries | ✅ Overlapping windows maintain context |
| ❌ Single point of failure | ✅ Segment-level error handling |

## Example Output

```
  Using Ollama for character detection...
  Processing 15 text segments with overlapping windows...
    ✓ Processed segments_1-2: 3 characters
    ✓ Processed segments_1-2-3: 4 characters
    ✓ Processed segments_2-3-4: 3 characters
    ✓ Processed segments_3-4-5: 4 characters
    ...
    ✓ Processed segments_14-15: 2 characters
  ✓ Ollama detected 8 characters
  + Added 'Officer Smith' from heuristic detection

  Final Character List:
  • John (male, serious) - 45 appearances
  • Mary (female, cheerful) - 38 appearances
  • Tom (male, nervous) - 22 appearances
  • Sarah (female, calm) - 31 appearances
  • Officer Smith (male, neutral) - 12 appearances
  • Dr. Jones (female, serious) - 8 appearances
  • Mrs. Brown (female, calm) - 15 appearances
  • The Captain (male, serious) - 19 appearances
```

## Files Modified

- ✅ `character_analyzer.py` - Core implementation
- ✅ `OLLAMA_CHARACTER_DETECTION.md` - Updated documentation
- ✅ `OLLAMA_SEGMENTED_IMPLEMENTATION.md` - Implementation details
- ✅ `test_ollama_segmented.py` - Test script
- ✅ `OLLAMA_QUICK_REFERENCE.md` - This file

## Next Steps

1. Test with a sample book
2. Review artifacts to verify quality
3. Adjust `segment_words` if needed
4. Run full audiobook generation with character mode

## Questions?

Check the detailed documentation:
- `OLLAMA_CHARACTER_DETECTION.md` - Feature overview
- `OLLAMA_SEGMENTED_IMPLEMENTATION.md` - Technical details
- `CHARACTER_MODE_GUIDE.md` - Complete workflow guide
