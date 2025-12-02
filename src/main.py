#!/usr/bin/env python
"""
EPUB to Audiobook Converter
Main entry point for converting EPUB files to audiobooks using IndexTTS2

This is now a thin CLI wrapper around the step-based execution system.
All actual processing happens in processing_steps.py with automatic resume support.
"""
import argparse
import os
import sys
import uuid
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from text_extractor import TextExtractor
from job_executor import JobExecutor
from job_processor import JobDefinition
import processing_steps  # Register all processing steps


def main():
    """
    CLI entry point - creates a temporary job and executes it using the step-based system.
    This ensures all executions (CLI and job queue) use the same resumable execution logic.
    """
    parser = argparse.ArgumentParser(
        description="Convert EPUB files to audiobooks using IndexTTS2 (with automatic resume support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (WAV output) - EPUB
  python main.py book.epub speaker.wav -o audiobook.wav
  
  # Basic usage with PDF
  python main.py document.pdf speaker.wav -o audiobook.wav
  
  # M4B audiobook format with metadata
  python main.py book.epub speaker.wav -o audiobook.m4b --format m4b
  
  # Detect characters (heuristic)
  python main.py book.epub speaker.wav -o output.m4b --detect-characters
  
  # Detect characters with Ollama (more accurate, recommended)
  python main.py book.epub speaker.wav -o output.m4b \\
    --detect-characters --ollama-character-detection
  
  # Character-aware mode with multiple voices
  python main.py book.epub speaker.wav -o audiobook.m4b \\
    --format m4b --character-mode \\
    --character-config work/character_voices.json \\
    --emotion-library work/emotion_library.json
  
  # Interactive character review
  python main.py book.epub speaker.wav -o audiobook.m4b \\
    --character-mode --review-characters
  
  # With emotion reference (standard mode)
  python main.py book.epub speaker.wav -o audiobook.m4b --format m4b --emo-audio emotion.wav
  
  # With Ollama text processing
  python main.py book.epub speaker.wav -o audiobook.wav --use-ollama
  
  # Custom segment size and model settings
  python main.py book.epub speaker.wav -o audiobook.m4b --format m4b --segment-words 400 --use-fp16
        """
    )
    
    # Required arguments
    parser.add_argument("source_text_file", help="Path to source text file to convert (EPUB or PDF)")
    parser.add_argument("speaker_audio", nargs='?', default=None, help="Path to speaker reference audio file (optional in character mode with --character-config)")
    
    # Output options
    parser.add_argument("-o", "--output", required=True, help="Path for output audiobook file")
    parser.add_argument("--format", choices=['wav', 'm4b'], default='wav', help="Output format: wav or m4b (default: wav)")
    parser.add_argument("--work-dir", default="./work", help="Working directory for temporary files (default: ./work)")
    parser.add_argument("--keep-segments", action="store_true", help="Keep individual segment audio files")
    
    # Text processing options
    parser.add_argument("--segment-words", type=int, default=500, help="Target words per segment (default: 500)")
    parser.add_argument("--max-words", type=int, default=600, help="Maximum words per segment (default: 600)")
    parser.add_argument("--min-words", type=int, default=100, help="Minimum words per segment (default: 100)")
    parser.add_argument("--disable-strip-unknown-tokens", action="store_true", default=True, help="Do not strip tokens that commonly cause TTS encoding issues (default: False)")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama for text cleanup")
    parser.add_argument("--ollama-model", default="aratan/DeepSeek-R1-32B-Uncensored:latest", help="Ollama model to use (default: aratan/DeepSeek-R1-32B-Uncensored:latest)")
    parser.add_argument("--ollama-url", default="http://host.docker.internal:11434", help="Ollama API URL")
    
    # Character processing options
    parser.add_argument("--character-mode", action="store_true", help="Enable character-aware processing")
    parser.add_argument("--character-config", help="Path to character voice configuration JSON")
    parser.add_argument("--emotion-library", help="Path to emotion reference library JSON")
    parser.add_argument("--detect-characters", action="store_true", help="Detect characters and create config template")
    parser.add_argument("--review-characters", action="store_true", help="Interactive character review before processing")
    parser.add_argument("--ollama-character-detection", action="store_true", help="Use Ollama for advanced character detection (more accurate)")
    
    # TTS options
    parser.add_argument("--emo-audio", help="Path to emotion reference audio file")
    parser.add_argument("--emo-alpha", type=float, default=1.0, help="Emotion alpha value (default: 1.0)")
    parser.add_argument("--emo-vector", type=float, nargs=8, help="Emotion vector (8 values: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm)")
    parser.add_argument("--use-emo-text", action="store_true", help="Detect emotion from text")
    parser.add_argument("--interval-silence", type=int, default=200, help="Silence between sentences in ms (default: 200)")
    parser.add_argument("--segment-silence", type=int, default=500, help="Silence between segments in ms (default: 500)")
    
    # Model options
    parser.add_argument("--config", default="./checkpoints/config.yaml", help="Path to config file")
    parser.add_argument("--model-dir", default="./checkpoints", help="Path to model directory")
    parser.add_argument("--use-fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--device", help="Device to use (e.g., cuda:0, cpu)")
    parser.add_argument("--no-cuda-kernel", action="store_true", help="Disable CUDA kernel for BigVGAN")
    parser.add_argument("--use-deepspeed", action="store_true", help="Use DeepSpeed")
    
    # Generation options
    parser.add_argument("--max-text-tokens", type=int, default=120, help="Max text tokens per TTS segment")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=10.0, help="Repetition penalty")
    parser.add_argument("--length-penalty", type=float, default=0.0, help="Length penalty")
    parser.add_argument("--num-beams", type=int, default=3, help="Number of beams for beam search")
    
    # Other options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.source_text_file):
        print(f"Error: Source text file not found: {args.source_text_file}")
        sys.exit(1)
    
    # Validate file format
    if not TextExtractor.is_supported_file(args.source_text_file):
        supported = ', '.join(TextExtractor.get_supported_extensions())
        print(f"Error: Unsupported file format. Supported formats: {supported}")
        sys.exit(1)
    
    # Speaker audio validation
    if not args.character_mode or not args.character_config:
        # Standard mode: speaker audio is required
        if not args.speaker_audio:
            print(f"Error: speaker_audio argument is required (or use --character-mode with --character-config)")
            sys.exit(1)
        if not os.path.exists(args.speaker_audio):
            print(f"Error: Speaker audio file not found: {args.speaker_audio}")
            sys.exit(1)
    else:
        # Character mode with config: speaker audio is optional
        if args.speaker_audio:
            if not os.path.exists(args.speaker_audio):
                print(f"Warning: Speaker audio file not found: {args.speaker_audio} (will use character-specific voices only)")
                args.speaker_audio = None
            else:
                print(f"Using speaker audio as default narrator voice: {args.speaker_audio}")
        else:
            print("Character mode: No default speaker audio (using character-specific voices only)")
    
    if args.emo_audio and not os.path.exists(args.emo_audio):
        print(f"Error: Emotion audio file not found: {args.emo_audio}")
        sys.exit(1)
    
    print("="*70)
    print("Text to Audiobook Converter (Step-Based Execution)")
    print("="*70)
    print("✅ Automatic resume support enabled")
    print("   If execution fails, intermediate results are saved")
    print("   Run again with same arguments to resume from last step")
    print("="*70)
    
    # Create a temporary job for CLI execution
    job_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    temp_jobs_dir = Path(tempfile.gettempdir()) / "audiobook_cli_jobs"
    temp_jobs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create job definition from CLI args
    job_def = JobDefinition(
        job_id=job_id,
        source_text_file=args.source_text_file,
        output_path=args.output,
        voice_ref_path=args.speaker_audio,
        format=args.format,
        detect_characters=args.detect_characters,
        ollama_character_detection=args.ollama_character_detection,
        character_mode=args.character_mode,
        keep_segments=args.keep_segments,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        segment_words=args.segment_words,
        strip_unknown_tokens=not args.disable_strip_unknown_tokens,
        character_config=args.character_config,
        emotion_library=args.emotion_library,
        emo_audio_prompt=args.emo_audio,
        created_at=datetime.now().isoformat()
    )
    
    # Set up temporary job directory structure
    job_dir = temp_jobs_dir / "pending" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save job definition
    with open(job_dir / "job_definition.json", 'w') as f:
        import json
        json.dump(job_def.to_dict(), f, indent=2)
    
    # Create work directory in job folder
    work_dir = job_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute using step-based executor
    executor = JobExecutor(temp_jobs_dir)
    
    try:
        success = executor.execute_job(
            job_id=job_id,
            job_data=job_def.to_dict(),
            resume=False
        )
        
        if success:
            print("\n" + "="*70)
            print("✅ Audiobook generation complete!")
            print("="*70)
            print(f"Output file: {args.output}")
            sys.exit(0)
        else:
            print("\n" + "="*70)
            print("❌ Audiobook generation failed")
            print("="*70)
            print(f"Job directory: {job_dir}")
            print("Check job_state.json for details on which step failed")
            sys.exit(1)
            
    finally:
        # Clean up temp job directory if successful
        if success:
            try:
                shutil.rmtree(temp_jobs_dir)
            except:
                pass


if __name__ == "__main__":
    main()
