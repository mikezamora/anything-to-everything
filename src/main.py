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
# sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "index-tts"))

from text_extractor import TextExtractor
from text_segmenter import TextSegmenter
from character_analyzer import CharacterAnalyzer
from character_segmenter import CharacterAwareSegmenter
from character_voice_config import CharacterVoiceMapping, EmotionLibrary
from character_review_tool import CharacterReviewTool
from ollama_processor import OllamaProcessor
from tts_processor import TTSProcessor
from audio_merger import AudioMerger


def main():
    parser = argparse.ArgumentParser(
        description="Convert EPUB files to audiobooks using IndexTTS2",
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
    
    # Create working directory
    os.makedirs(args.work_dir, exist_ok=True)
    segments_dir = os.path.join(args.work_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    print("="*70)
    print("Text to Audiobook Converter")
    print("="*70)
    
    # Step 1: Extract text from source file
    print(f"\n[1/6] Extracting text from {Path(args.source_text_file).suffix.upper()} file...")
    extractor = TextExtractor.create_extractor(args.source_text_file)
    metadata = extractor.get_metadata()
    text = extractor.extract_text()
    
    print(f"  Title: {metadata.get('title', 'Unknown')}")
    print(f"  Author: {metadata.get('author', 'Unknown')}")
    print(f"  Language: {metadata.get('language', 'Unknown')}")
    print(f"  Extracted: {len(text)} characters")
    
    if not text:
        print("Error: No text extracted from source file")
        sys.exit(1)
    
    # Character detection and configuration (if enabled)
    character_segments = None
    voice_mapping = None
    emotion_library = None
    
    if args.detect_characters:
        print("\n[2/6] Analyzing characters...")
        
        # Use Ollama for character detection if requested
        use_ollama_for_chars = args.ollama_character_detection or (args.use_ollama and args.character_mode)
        
        analyzer = CharacterAnalyzer(
            use_ollama=use_ollama_for_chars,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model
        )
        
        # Detect characters
        characters = analyzer.detect_characters(text)
        print(f"  Detected {len(characters)} characters:")
        for name, traits in sorted(characters.items(), key=lambda x: x[1].appearances, reverse=True):
            print(f"    - {name}: {traits.gender}, {traits.demeanor} ({traits.appearances} appearances)")
        
        # Save detected characters
        char_file = os.path.join(args.work_dir, "detected_characters.json")
        analyzer.save_characters(char_file)
        print(f"  Saved character data to: {char_file}")
        
        # Interactive review if requested
        if args.review_characters:
            print("\n[Character Review]")
            review_tool = CharacterReviewTool(analyzer)
            review_tool.run_interactive_review(args.work_dir)
            # Reload potentially updated characters
            char_file = os.path.join(args.work_dir, "reviewed_characters.json")
            if os.path.exists(char_file):
                analyzer.load_characters(char_file)
        elif args.detect_characters:
            # Just create template and exit
            voice_config_template = os.path.join(args.work_dir, "character_voices_template.json")
            CharacterVoiceMapping.create_template(list(characters.keys()), voice_config_template)
            
            emotion_lib_template = os.path.join(args.work_dir, "emotion_library_template.json")
            EmotionLibrary.create_template(emotion_lib_template)
            
            print("\n" + "="*70)
            print("Character detection complete!")
            print("="*70)
            print(f"\nFiles created:")
            print(f"  1. {char_file}")
            print(f"  2. {voice_config_template}")
            print(f"  3. {emotion_lib_template}")
            print(f"\nNext steps:")
            print(f"  1. Edit {voice_config_template} to map characters to voice files")
            print(f"  2. Edit {emotion_lib_template} to add emotion reference files")
            print(f"  3. Run converter with --character-mode and --character-config options")
            sys.exit(0)
        
    # Load voice configuration if provided
    if args.character_config:
        # Check if file exists in work dir (copied by job processor) or use provided path
        work_dir_config = os.path.join(args.work_dir, os.path.basename(args.character_config))
        if os.path.exists(work_dir_config):
            config_path = work_dir_config
            print(f"  Using character config from work directory: {config_path}")
        elif os.path.exists(args.character_config):
            config_path = args.character_config
            print(f"  Using character config from original path: {config_path}")
        else:
            print(f"Error: Character config file not found: {args.character_config}")
            print(f"  Also checked: {work_dir_config}")
            sys.exit(1)
        
        voice_mapping = CharacterVoiceMapping.load(config_path)
        print(f"  Loaded voice configuration with {len(voice_mapping.character_voices)} characters")
    
    # Load emotion library if provided
    if args.emotion_library:
        # Check if file exists in work dir (copied by job processor) or use provided path
        work_dir_emotion = os.path.join(args.work_dir, os.path.basename(args.emotion_library))
        if os.path.exists(work_dir_emotion):
            emotion_path = work_dir_emotion
            print(f"  Using emotion library from work directory: {emotion_path}")
        elif os.path.exists(args.emotion_library):
            emotion_path = args.emotion_library
            print(f"  Using emotion library from original path: {emotion_path}")
        else:
            print(f"Error: Emotion library not found: {args.emotion_library}")
            print(f"  Also checked: {work_dir_emotion}")
            sys.exit(1)
        
        emotion_library = EmotionLibrary.load(emotion_path)
        print(f"  Loaded emotion library")
    
    # Step 3: Optional Ollama text processing (BEFORE segmentation for better quality)
    cleaned_segments = None  # Will be set if Ollama processing is used
    
    if args.use_ollama:
        print(f"\n[3/6] Processing text with Ollama ({args.ollama_model})...")
        ollama = OllamaProcessor(
            base_url=args.ollama_url, 
            model=args.ollama_model,
            work_dir=args.work_dir
        )
        
        if ollama.is_available():
            # Process the raw text before segmentation
            print(f"  Cleaning up text for better segmentation...")
            # Create temporary segments for processing
            temp_segmenter = TextSegmenter(
                target_words=args.segment_words,
                max_words=args.max_words,
                min_words=args.min_words
            )
            temp_segments = temp_segmenter.segment_text(text)
            
            # Clean the segments
            cleaned_segments = ollama.process_segments(temp_segments, show_progress=True)
            
            # For character mode, reconstruct cleaned text for character-aware segmentation
            if args.character_mode:
                text = " ".join(cleaned_segments)
                print(f"  Text cleaned and ready for character-aware segmentation")
            else:
                # For standard mode, use cleaned segments directly (no need to re-segment)
                print(f"  Processed {len(cleaned_segments)} cleaned segments")
        else:
            print(f"  Ollama not available, skipping text processing")
    else:
        print("\n[3/6] Skipping Ollama text processing (not enabled)")
    
    # Step 4: Segment text
    if args.character_mode and voice_mapping:
        print(f"\n[4/6] Creating character-aware segments...")
        
        # Initialize analyzer with known characters from config
        analyzer = CharacterAnalyzer(
            use_ollama=False,  # We're not detecting, just using known characters
            work_dir=args.work_dir
        )
        
        # Load characters from voice mapping config
        character_names = list(voice_mapping.character_voices.keys())
        print(f"  Loaded {len(character_names)} characters from config: {', '.join(character_names)}")
        
        # Try to load character traits from detected_characters.json
        detected_chars_file = os.path.join(args.work_dir, "detected_characters.json")
        if os.path.exists(detected_chars_file):
            print(f"  Loading character traits from: {detected_chars_file}")
            analyzer.load_characters(detected_chars_file)
        else:
            # Create basic character traits for the known characters
            from character_analyzer import CharacterTraits
            print(f"  No detected_characters.json found, using basic character traits")
            for name in character_names:
                if name not in analyzer.characters:
                    analyzer.characters[name] = CharacterTraits(
                        name=name,
                        gender="Unknown",
                        demeanor="Unknown",
                        appearances=0  # Will be counted during segmentation
                    )
        
        # Decide whether to use Ollama for character segmentation
        use_ollama_for_segmentation = args.ollama_character_detection or args.use_ollama
        
        char_segmenter = CharacterAwareSegmenter(
            max_words_per_segment=args.max_words,
            min_words_per_segment=args.min_words,
            use_ollama=use_ollama_for_segmentation,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            work_dir=args.work_dir
        )
        
        # Override the segmenter's analyzer with our configured one
        char_segmenter.analyzer = analyzer
        
        # Create base segments first
        base_segmenter = TextSegmenter(
            target_words=args.segment_words,
            max_words=args.max_words,
            min_words=args.min_words
        )
        base_segments = base_segmenter.segment_text(text)
        
        # Create character-aware segments
        character_segments = char_segmenter.segment_text(text, base_segments)
        stats = char_segmenter.get_segment_stats(character_segments)
        
        print(f"  Created {stats['total_segments']} character-aware segments")
        print(f"  Total words: {stats['total_words']}")
        print(f"  Dialogue: {stats['dialogue_segments']}, Thoughts: {stats['thought_segments']}, Narration: {stats['narration_segments']}")
        print(f"  Characters in text: {', '.join(stats['characters']) if stats['characters'] else 'None'}")
    else:
        # Standard mode segmentation
        if cleaned_segments is not None:
            # Use pre-cleaned segments from Ollama (Step 3)
            print(f"\n[4/6] Using {len(cleaned_segments)} Ollama-cleaned segments")
            segments = cleaned_segments
            
            # Calculate stats for cleaned segments
            segmenter = TextSegmenter(
                target_words=args.segment_words,
                max_words=args.max_words,
                min_words=args.min_words
            )
            stats = segmenter.get_segment_stats(segments)
            
            print(f"  Total words: {stats['total_words']}")
            print(f"  Avg words per segment: {stats['avg_words_per_segment']:.1f}")
            print(f"  Min/Max words: {stats['min_words']}/{stats['max_words']}")
        else:
            # Create new segments from raw text
            print(f"\n[4/6] Segmenting text ({args.segment_words} words per segment)...")
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
    
    # Step 5: Initialize TTS
    print("\n[5/6] Initializing IndexTTS2...")
    tts_processor = TTSProcessor(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_fp16=args.use_fp16,
        device=args.device,
        use_cuda_kernel=not args.no_cuda_kernel,
        use_deepspeed=args.use_deepspeed
    )
    
    # Step 6: Process segments with TTS
    print(f"\n[6/6] Generating audio with IndexTTS2.....")
    start_time = time.time()
    
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_beams": args.num_beams,
    }
    
    if args.character_mode and character_segments and voice_mapping:
        # Character-aware TTS processing
        print(f"  Processing {len(character_segments)} character-aware segments...")
        audio_files = []
        
        for i, char_seg in enumerate(character_segments):
            print(f"character: {char_seg.character}, is_narration: {char_seg.is_narration}, dominant_emotion: {char_seg.emotional_state.dominant_emotion}")

            # Get voice config for this character
            voice_config = voice_mapping.get_voice_for_character(
                char_seg.character,
                char_seg.is_narration
            )

            if not os.path.exists(voice_config.speaker_audio):
                print(f"Error: speaker_audio file not found: {voice_config.speaker_audio}")
                sys.exit(1)

            print(f"  Loaded speaker_audio: {voice_config.speaker_audio}")

            # Get emotion audio if available
            emo_audio = None
            if emotion_library and char_seg.emotional_state.dominant_emotion:
                print(f"  Using emotion audio for: {char_seg.emotional_state.dominant_emotion}")
                emo_audio = emotion_library.get_emotion_audio(
                    char_seg.emotional_state.dominant_emotion
                )
                print(f"  Loaded emotion_audio: {emo_audio}")
            
            # Override with voice config's emotion audio if specified
            if voice_config.emotion_audio:
                print(f"  Overriding emotion audio with character-specific file: {voice_config.emotion_audio}")
                emo_audio = voice_config.emotion_audio
            
            # Convert emotion vector to list if needed
            emo_vector = None
            # if char_seg.emotional_state.emotions:
            #     emo_vector = [
            #         char_seg.emotional_state.emotions.get('happy', 0.0),
            #         char_seg.emotional_state.emotions.get('angry', 0.0),
            #         char_seg.emotional_state.emotions.get('sad', 0.0),
            #         char_seg.emotional_state.emotions.get('afraid', 0.0),
            #         char_seg.emotional_state.emotions.get('disgusted', 0.0),
            #         char_seg.emotional_state.emotions.get('melancholic', 0.0),
            #         char_seg.emotional_state.emotions.get('surprised', 0.0),
            #         char_seg.emotional_state.emotions.get('calm', 0.0),
            #     ]
            
            # if args.verbose:
            seg_type = "Dialogue" if char_seg.is_dialogue else "Thought" if char_seg.is_thought else "Narration"
            print(f"  [{i+1}/{len(character_segments)}] {char_seg.character or 'NARRATOR'} ({seg_type}, {char_seg.emotional_state.dominant_emotion})")
            
            # Process single segment
            seg_audio = tts_processor.process_segments(
                segments=[char_seg.text],
                index=i,
                index_total=len(character_segments),
                output_dir=segments_dir,
                spk_audio_prompt=voice_config.speaker_audio,
                emo_audio_prompt=emo_audio,
                emo_alpha=1.0,
                emo_vector=None,
                use_emo_text=False,
                interval_silence=args.interval_silence,
                verbose=False,
                max_text_tokens_per_segment=args.max_text_tokens,
                **generation_kwargs
            )
            
            if seg_audio:
                audio_files.extend(seg_audio)
    else:
        # Standard TTS processing
        audio_files = tts_processor.process_segments(
            segments=segments if character_segments is None else [seg.text for seg in character_segments],
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
    print(f"\n  Generated {len(audio_files)} audio segments")
    print(f"  Total TTS time: {tts_time:.2f} seconds ({tts_time/60:.2f} minutes)")
    
    if not audio_files:
        print("Error: No audio files generated")
        sys.exit(1)
    
    # Step 7: Merge audio files
    print(f"\n[7/6] Merging audio segments...")
    merger = AudioMerger(silence_duration_ms=args.segment_silence)
    
    merge_metadata = {
        "title": metadata.get('title', 'Unknown'),
        "author": metadata.get('author', 'Unknown'),
        "album": metadata.get('album', metadata.get('title', 'Unknown')),  # For M4B: use series or title
        "segments": len(audio_files),
        "total_words": stats['total_words'],
        "generation_time_seconds": tts_time
    }
    
    # Add optional metadata if available
    if 'publisher' in metadata:
        merge_metadata['publisher'] = metadata['publisher']
    if 'date' in metadata:
        merge_metadata['date'] = metadata['date']
    if 'series' in metadata:
        merge_metadata['series'] = metadata['series']
    
    final_audio = merger.merge_with_metadata(
        audio_files=audio_files,
        output_path=args.output,
        metadata=merge_metadata,
        output_format=args.format
    )
    
    if final_audio:
        print(f"\n{'='*70}")
        print("Audiobook generation complete!")
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
            print("Cleanup complete")
        else:
            print(f"\nSegment files kept in: {segments_dir}")
    else:
        print("\nError: Failed to merge audio files")
        sys.exit(1)


if __name__ == "__main__":
    main()
