# Ollama Processing Artifacts Guide

When you use the `--use-ollama` flag, the EPUB to Audiobook converter saves detailed artifacts of the text processing workflow. This allows you to review, debug, and understand how Ollama modified your text.

## Directory Structure

After running with `--use-ollama`, you'll find these directories in your work folder:

```
work/
└── ollama/
    ├── prompts/                          # Prompts sent to Ollama
    │   ├── segment_0001_prompt.txt
    │   ├── segment_0002_prompt.txt
    │   └── ...
    ├── original_text/                    # Original text before processing
    │   ├── segment_0001.txt
    │   ├── segment_0002.txt
    │   └── ...
    ├── processed_text/                   # Text after Ollama processing
    │   ├── segment_0001.txt
    │   ├── segment_0002.txt
    │   └── ...
    ├── segment_0001_comparison.txt       # Side-by-side comparisons
    ├── segment_0002_comparison.txt
    ├── ...
    ├── session_metadata.txt              # Session information
    └── processing_summary.txt            # Processing summary
```

## File Descriptions

### Prompts Directory (`prompts/`)

Contains the exact prompts sent to Ollama for each segment. By default, the prompt instructs Ollama to:
- Remove formatting artifacts
- Fix obvious typos
- Ensure text flows naturally when read aloud
- Not add or remove content

**Example filename**: `segment_0001_prompt.txt`

**Use case**: Review to understand what instructions were given to Ollama, or to create custom prompts.

### Original Text Directory (`original_text/`)

Contains the original text segments exactly as extracted from the EPUB, before any Ollama processing.

**Example filename**: `segment_0001.txt`

**Use case**: Compare with processed text to see what changed, or to revert if processing was too aggressive.

### Processed Text Directory (`processed_text/`)

Contains the text after Ollama processing. This is what gets sent to IndexTTS2 for audio generation.

**Example filename**: `segment_0001.txt`

**Use case**: Review the cleaned text that will be used for TTS generation.

### Comparison Files

Side-by-side comparison files showing both original and processed text for easy review.

**Example filename**: `segment_0001_comparison.txt`

**Format**:
```
================================================================================
ORIGINAL TEXT
================================================================================
[original text here]

================================================================================
PROCESSED TEXT
================================================================================
[processed text here]
```

**Use case**: Quick visual inspection of changes made by Ollama.

### Session Metadata (`session_metadata.txt`)

Information about the Ollama processing session:
- Date and time of processing
- Model used (e.g., llama2, mistral, etc.)
- API endpoint
- Total number of segments processed

**Use case**: Track which model and settings were used for this conversion.

### Processing Summary (`processing_summary.txt`)

Summary of the entire Ollama processing run:
- Total segments processed
- Model used
- Directory locations for all artifacts

**Use case**: Quick reference for where all files are located.

## Common Use Cases

### 1. Review Changes Made by Ollama

```powershell
# Navigate to comparisons
cd work\ollama

# View a specific comparison
type segment_0001_comparison.txt

# Or view all comparisons
Get-ChildItem *_comparison.txt | ForEach-Object { type $_.Name; Write-Host "`n---`n" }
```

### 2. Find Aggressive Changes

```powershell
# Compare file sizes to find large changes
cd work\ollama
Get-ChildItem original_text\*.txt | ForEach-Object {
    $orig = (Get-Content $_.FullName -Raw).Length
    $proc = (Get-Content ("processed_text\" + $_.Name) -Raw).Length
    $diff = [Math]::Abs($orig - $proc)
    if ($diff -gt 100) {
        Write-Host "$($_.Name): $diff characters changed"
    }
}
```

### 3. Revert a Specific Segment

If Ollama over-processed a segment, you can manually edit the processed version:

```powershell
# Copy original to processed to revert
copy work\ollama\original_text\segment_0005.txt work\ollama\processed_text\segment_0005.txt
```

Note: This only works before TTS generation. After audio is generated, you'd need to re-run that segment.

### 4. Create Custom Prompts

Edit the default prompt in `ollama_processor.py` or pass a custom prompt template:

```python
custom_prompt = """Improve the following text for audiobook narration.
Fix grammar and make it more engaging. Text: {text}"""
```

### 5. Quality Assurance Workflow

1. Run conversion with `--use-ollama`
2. Check `processing_summary.txt` to verify settings
3. Review comparison files for random segments
4. If issues found, adjust prompt or model and re-run
5. Keep artifacts for future reference

## Tips

### Choosing the Right Ollama Model

Different models have different strengths:

- **llama2**: Good general-purpose model, balanced
- **llama3**: Better at following instructions
- **mistral**: Fast and efficient
- **gemma**: Good for creative text
- **qwen2.5**: Excellent for multi-language support

Example:
```powershell
python main.py book.epub speaker.wav -o out.wav --use-ollama --ollama-model llama3
```

### Batch Review Script

Create a PowerShell script to review all changes:

```powershell
# review_ollama.ps1
$comparisons = Get-ChildItem work\ollama\*_comparison.txt
Write-Host "Total segments: $($comparisons.Count)"
Write-Host "`nPress Enter to view each comparison..."

foreach ($file in $comparisons) {
    Clear-Host
    Write-Host "=== $($file.Name) ===" -ForegroundColor Cyan
    Get-Content $file.FullName
    Write-Host "`n[Press Enter for next, Ctrl+C to exit]" -ForegroundColor Yellow
    Read-Host
}
```

### Analyzing Processing Quality

```powershell
# Count total characters changed
$totalOrig = 0
$totalProc = 0

Get-ChildItem work\ollama\original_text\*.txt | ForEach-Object {
    $totalOrig += (Get-Content $_.FullName -Raw).Length
}

Get-ChildItem work\ollama\processed_text\*.txt | ForEach-Object {
    $totalProc += (Get-Content $_.FullName -Raw).Length
}

$percentChange = [Math]::Round((($totalOrig - $totalProc) / $totalOrig) * 100, 2)
Write-Host "Total original: $totalOrig chars"
Write-Host "Total processed: $totalProc chars"
Write-Host "Change: $percentChange%"
```

## Troubleshooting

### No artifacts saved

- Ensure you're using `--use-ollama` flag
- Check that `--work-dir` is writable
- Verify Ollama is actually processing (not just skipping due to unavailability)

### Empty processed files

- Check Ollama is responding (test with `ollama list`)
- Review session_metadata.txt for errors
- Try a different model

### Large differences in text

- Some formatting artifacts may cause large changes
- Review comparison files to verify quality
- Consider adjusting the prompt to be less aggressive

## Best Practices

1. **Always review artifacts for first-time conversions** - Different books have different formatting quirks
2. **Keep artifacts with audiobooks** - Useful for troubleshooting or re-generation
3. **Use version control** - Track which prompts/models work best for different content
4. **Document custom prompts** - If you customize prompts, save them for future use
5. **Spot-check comparisons** - Review 3-5 random segments to ensure quality

## Advanced: Custom Processing Pipeline

You can modify `ollama_processor.py` to:
- Save additional metadata (token counts, processing time)
- Generate quality scores
- Apply multiple processing passes
- Compare multiple models side-by-side

See the source code for extension points.

## Example Workflow

```powershell
# 1. Run conversion with Ollama
python main.py book.epub speaker.wav -o audiobook.wav --use-ollama --keep-segments

# 2. Review summary
type work\ollama\processing_summary.txt

# 3. Spot-check a few comparisons
type work\ollama\segment_0001_comparison.txt
type work\ollama\segment_0010_comparison.txt
type work\ollama\segment_0050_comparison.txt

# 4. If quality is good, you're done!
# If not, adjust model/prompt and re-run specific segments

# 5. Archive artifacts for reference
Compress-Archive -Path work\ollama -DestinationPath audiobook_ollama_artifacts.zip
```

---

**Remember**: These artifacts are for debugging and quality assurance. They help you understand and control how Ollama modifies your text before TTS generation.
