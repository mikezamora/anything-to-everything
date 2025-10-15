#!/usr/bin/env python
"""
EPUB to Audiobook Converter
Main entry point for converting EPUB files to audiobooks using IndexTTS2
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from epub_extractor import EPUBExtractor
from text_segmenter import TextSegmenter
from ollama_processor import OllamaProcessor
from tts_processor import TTSProcessor
from audio_merger import AudioMerger


def main():
    parser = argparse.ArgumentParser(
        description="Convert EPUB files to audiobooks using IndexTTS2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py book.epub speaker.wav -o audiobook.wav
  
  # With emotion reference
  python main.py book.epub speaker.wav -o audiobook.wav --emo-audio emotion.wav
  
  # With Ollama text processing
  python main.py book.epub speaker.wav -o audiobook.wav --use-ollama
  
  # Custom segment size and model settings
  python main.py book.epub speaker.wav -o audiobook.wav --segment-words 400 --use-fp16
        """
    )
    
    # Required arguments
    parser.add_argument("epub_file", help="Path to EPUB file to convert")
    parser.add_argument("speaker_audio", help="Path to speaker reference audio file")
    
    # Output options
    parser.add_argument("-o", "--output", required=True, help="Path for output audiobook WAV file")
    parser.add_argument("--work-dir", default="./work", help="Working directory for temporary files (default: ./work)")
    parser.add_argument("--keep-segments", action="store_true", help="Keep individual segment audio files")
    
    # Text processing options
    parser.add_argument("--segment-words", type=int, default=500, help="Target words per segment (default: 500)")
    parser.add_argument("--max-words", type=int, default=600, help="Maximum words per segment (default: 600)")
    parser.add_argument("--min-words", type=int, default=100, help="Minimum words per segment (default: 100)")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama for text cleanup")
    parser.add_argument("--ollama-model", default="llama2", help="Ollama model to use (default: llama2)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    
    # TTS options
    parser.add_argument("--emo-audio", help="Path to emotion reference audio file")
    parser.add_argument("--emo-alpha", type=float, default=1.0, help="Emotion alpha value (default: 1.0)")
    parser.add_argument("--emo-vector", type=float, nargs=8, help="Emotion vector (8 values: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm)")
    parser.add_argument("--use-emo-text", action="store_true", help="Detect emotion from text")
    parser.add_argument("--interval-silence", type=int, default=200, help="Silence between sentences in ms (default: 200)")
    parser.add_argument("--segment-silence", type=int, default=500, help="Silence between segments in ms (default: 500)")
    
    # Model options
    parser.add_argument("--config", default="../../checkpoints/config.yaml", help="Path to config file")
    parser.add_argument("--model-dir", default="../../checkpoints", help="Path to model directory")
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
    if not os.path.exists(args.epub_file):
        print(f"Error: EPUB file not found: {args.epub_file}")
        sys.exit(1)
    
    if not os.path.exists(args.speaker_audio):
        print(f"Error: Speaker audio file not found: {args.speaker_audio}")
        sys.exit(1)
    
    if args.emo_audio and not os.path.exists(args.emo_audio):
        print(f"Error: Emotion audio file not found: {args.emo_audio}")
        sys.exit(1)
    
    # Create working directory
    os.makedirs(args.work_dir, exist_ok=True)
    segments_dir = os.path.join(args.work_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    print("="*70)
    print("EPUB to Audiobook Converter")
    print("="*70)
    
    # Step 1: Extract text from EPUB
    print("\n[1/6] Extracting text from EPUB...")
    extractor = EPUBExtractor(args.epub_file)
    metadata = extractor.get_metadata()
    text = extractor.extract_text()
    
    print(f"  Title: {metadata.get('title', 'Unknown')}")
    print(f"  Author: {metadata.get('author', 'Unknown')}")
    print(f"  Language: {metadata.get('language', 'Unknown')}")
    print(f"  Extracted: {len(text)} characters")
    
    if not text:
        print("Error: No text extracted from EPUB")
        sys.exit(1)
    
    # Step 2: Segment text
    print(f"\n[2/6] Segmenting text ({args.segment_words} words per segment)...")
    segmenter = TextSegmenter(
        target_words=args.segment_words,
        max_words=args.max_words,
        min_words=args.min_words
    )
    segments = segmenter.segment_text(text)
    stats = segmenter.get_segment_stats(segments)
    
    print(f"  Created {stats['total_segments']} segments")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Avg words per segment: {stats['avg_words_per_segment']:.1f}")
    print(f"  Min/Max words: {stats['min_words']}/{stats['max_words']}")
    
    # Step 3: Optional Ollama processing
    if args.use_ollama:
        print(f"\n[3/6] Processing text with Ollama ({args.ollama_model})...")
        ollama = OllamaProcessor(
            base_url=args.ollama_url, 
            model=args.ollama_model,
            work_dir=args.work_dir
        )
        
        if ollama.is_available():
            segments = ollama.process_segments(segments, show_progress=True)
            print(f"  ✓ Processed {len(segments)} segments")
        else:
            print(f"  ⚠ Ollama not available, skipping text processing")
    else:
        print("\n[3/6] Skipping Ollama text processing (not enabled)")
    
    # Step 4: Initialize TTS
    print("\n[4/6] Initializing IndexTTS2...")
    tts_processor = TTSProcessor(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_fp16=args.use_fp16,
        device=args.device,
        use_cuda_kernel=not args.no_cuda_kernel,
        use_deepspeed=args.use_deepspeed
    )
    
    # Step 5: Process segments with TTS
    print(f"\n[5/6] Generating audio with IndexTTS2...")
    start_time = time.time()
    
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_beams": args.num_beams,
    }
    
    audio_files = tts_processor.process_segments(
        segments=segments,
        output_dir=segments_dir,
        spk_audio_prompt=args.speaker_audio,
        emo_audio_prompt=args.emo_audio,
        emo_alpha=args.emo_alpha,
        emo_vector=args.emo_vector,
        use_emo_text=args.use_emo_text,
        interval_silence=args.interval_silence,
        verbose=args.verbose,
        max_text_tokens_per_segment=args.max_text_tokens,
        **generation_kwargs
    )
    
    tts_time = time.time() - start_time
    print(f"\n  ✓ Generated {len(audio_files)} audio segments")
    print(f"  Total TTS time: {tts_time:.2f} seconds ({tts_time/60:.2f} minutes)")
    
    if not audio_files:
        print("Error: No audio files generated")
        sys.exit(1)
    
    # Step 6: Merge audio files
    print(f"\n[6/6] Merging audio segments...")
    merger = AudioMerger(silence_duration_ms=args.segment_silence)
    
    merge_metadata = {
        "title": metadata.get('title', 'Unknown'),
        "author": metadata.get('author', 'Unknown'),
        "segments": len(audio_files),
        "total_words": stats['total_words'],
        "generation_time_seconds": tts_time
    }
    
    final_audio = merger.merge_with_metadata(
        audio_files=audio_files,
        output_path=args.output,
        metadata=merge_metadata
    )
    
    if final_audio:
        print(f"\n{'='*70}")
        print("✓ Audiobook generation complete!")
        print(f"{'='*70}")
        print(f"Output file: {final_audio}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        
        # Cleanup temporary files if requested
        if not args.keep_segments:
            print("\nCleaning up temporary segment files...")
            for audio_file in audio_files:
                try:
                    os.remove(audio_file)
                except:
                    pass
            print("✓ Cleanup complete")
        else:
            print(f"\nSegment files kept in: {segments_dir}")
    else:
        print("\nError: Failed to merge audio files")
        sys.exit(1)


if __name__ == "__main__":
    main()
