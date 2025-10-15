# Example PowerShell script for processing an EPUB audiobook

# Configuration
$EPUB_FILE = "book.epub"
$SPEAKER_AUDIO = "speaker_voice.wav"
$OUTPUT_FILE = "audiobook.wav"
$WORK_DIR = "./work"

# Check if files exist
if (-not (Test-Path $EPUB_FILE)) {
    Write-Error "Error: EPUB file not found: $EPUB_FILE"
    exit 1
}

if (-not (Test-Path $SPEAKER_AUDIO)) {
    Write-Error "Error: Speaker audio file not found: $SPEAKER_AUDIO"
    exit 1
}

# Run the converter
python main.py $EPUB_FILE $SPEAKER_AUDIO `
    -o $OUTPUT_FILE `
    --work-dir $WORK_DIR `
    --segment-words 500 `
    --use-fp16 `
    --temperature 0.8 `
    --keep-segments `
    --verbose

Write-Host "Done! Audiobook saved to: $OUTPUT_FILE"
