#!/bin/bash
# Example script for processing an EPUB audiobook

# Configuration
EPUB_FILE="book.epub"
SPEAKER_AUDIO="speaker_voice.wav"
OUTPUT_FILE="audiobook.wav"
WORK_DIR="./work"

# Check if files exist
if [ ! -f "$EPUB_FILE" ]; then
    echo "Error: EPUB file not found: $EPUB_FILE"
    exit 1
fi

if [ ! -f "$SPEAKER_AUDIO" ]; then
    echo "Error: Speaker audio file not found: $SPEAKER_AUDIO"
    exit 1
fi

# Run the converter
python main.py "$EPUB_FILE" "$SPEAKER_AUDIO" \
    -o "$OUTPUT_FILE" \
    --work-dir "$WORK_DIR" \
    --segment-words 500 \
    --use-fp16 \
    --temperature 0.8 \
    --keep-segments \
    --verbose

echo "Done! Audiobook saved to: $OUTPUT_FILE"
