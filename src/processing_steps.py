"""
Processing Steps Definition
Defines the actual steps for audiobook generation that can be tracked and resumed
"""
import os
import time
from pathlib import Path
from typing import Dict, Any

from step_registry import register_step
from job_state import StepExecutionContext


@register_step(
    step_id="extract_text",
    step_name="Extract Text from Source",
    order=1,
    required=True,
    max_retries=2
)
def extract_text_step(context: StepExecutionContext) -> Dict[str, Any]:
    """Extract text from EPUB/PDF file"""
    from text_extractor import TextExtractor
    
    source_file = context.job_data.get('source_text_file')
    
    extractor = TextExtractor.create_extractor(source_file)
    metadata = extractor.get_metadata()
    text = extractor.extract_text()
    
    # Save extracted text to work directory for potential resume
    text_file = Path(context.work_dir) / "extracted_text.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return {
        'text': text,
        'text_file': str(text_file),
        'metadata': metadata,
        'char_count': len(text)
    }


@register_step(
    step_id="process_ollama",
    step_name="Process Text with Ollama",
    order=2,
    required=False,  # Optional step
    max_retries=2
)
def ollama_processing_step(context: StepExecutionContext) -> Dict[str, Any]:
    """Process text with Ollama for cleanup (if enabled)"""
    from ollama_processor import OllamaProcessor
    from text_segmenter import TextSegmenter
    
    # Check if Ollama is enabled
    use_ollama = context.job_data.get('use_ollama', False)
    if not use_ollama:
        return {'skipped': True, 'reason': 'Ollama not enabled'}
    
    # Get text from previous step
    extract_result = context.get_previous_step_result('extract_text')
    if not extract_result:
        # Try loading from file
        text_file = Path(context.work_dir) / "extracted_text.txt"
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError("Cannot find extracted text")
    else:
        text = extract_result.get('text')
    
    ollama_model = context.job_data.get('ollama_model', 'aratan/DeepSeek-R1-32B-Uncensored:latest')
    ollama_url = context.job_data.get('ollama_url', 'http://host.docker.internal:11434')
    
    ollama = OllamaProcessor(
        base_url=ollama_url,
        model=ollama_model,
        work_dir=context.work_dir
    )
    
    if not ollama.is_available():
        return {'skipped': True, 'reason': 'Ollama not available'}
    
    # Create temporary segments for processing
    segment_words = context.job_data.get('segment_words', 500)
    max_words = context.job_data.get('max_words', 600)
    min_words = context.job_data.get('min_words', 100)
    strip_unknown = context.job_data.get('strip_unknown_tokens', True)
    
    segmenter = TextSegmenter(
        target_words=segment_words,
        max_words=max_words,
        min_words=min_words,
        strip_unknown_tokens=strip_unknown
    )
    temp_segments = segmenter.segment_text(text)
    
    # Process segments
    cleaned_segments = ollama.process_segments(temp_segments, show_progress=True)
    
    # Save cleaned segments
    cleaned_file = Path(context.work_dir) / "ollama_cleaned_segments.txt"
    with open(cleaned_file, 'w', encoding='utf-8') as f:
        for seg in cleaned_segments:
            f.write(seg + "\n\n")
    
    return {
        'cleaned_segments': cleaned_segments,
        'cleaned_file': str(cleaned_file),
        'segment_count': len(cleaned_segments)
    }


@register_step(
    step_id="segment_text",
    step_name="Segment Text for TTS",
    order=3,
    required=True,
    max_retries=2
)
def segment_text_step(context: StepExecutionContext) -> Dict[str, Any]:
    """Segment text into TTS-processable chunks"""
    from text_segmenter import TextSegmenter
    from character_segmenter import CharacterAwareSegmenter
    from character_analyzer import CharacterAnalyzer
    from character_voice_config import CharacterVoiceMapping
    
    character_mode = context.job_data.get('character_mode', False)
    
    segment_words = context.job_data.get('segment_words', 500)
    max_words = context.job_data.get('max_words', 600)
    min_words = context.job_data.get('min_words', 100)
    strip_unknown = context.job_data.get('strip_unknown_tokens', True)
    
    # Check if we have Ollama-cleaned segments
    ollama_result = context.get_previous_step_result('process_ollama')
    if ollama_result and not ollama_result.get('skipped', False):
        # Use cleaned segments
        segments = ollama_result.get('cleaned_segments', [])
        if character_mode:
            # Need to reconstruct text for character segmentation
            text = " ".join(segments)
        else:
            # Save segments and return
            segments_file = Path(context.work_dir) / "segments.txt"
            with open(segments_file, 'w', encoding='utf-8') as f:
                for seg in segments:
                    f.write(seg + "\n\n")
            
            return {
                'segments': segments,
                'segments_file': str(segments_file),
                'segment_count': len(segments),
                'character_mode': False
            }
    else:
        # Get original text
        extract_result = context.get_previous_step_result('extract_text')
        if extract_result:
            text = extract_result.get('text')
        else:
            # Load from file
            text_file = Path(context.work_dir) / "extracted_text.txt"
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
    
    if character_mode:
        # Character-aware segmentation
        character_config = context.job_data.get('character_config')
        
        # Load character voice mapping
        voice_mapping = CharacterVoiceMapping.load(character_config)
        character_names = list(voice_mapping.character_voices.keys())
        
        # Create analyzer
        analyzer = CharacterAnalyzer(use_ollama=False, work_dir=context.work_dir)
        
        # Load character traits if available
        detected_chars_file = Path(context.work_dir) / "detected_characters.json"
        if detected_chars_file.exists():
            analyzer.load_characters(str(detected_chars_file))
        else:
            # Create basic traits
            from character_analyzer import CharacterTraits
            for name in character_names:
                if name not in analyzer.characters:
                    analyzer.characters[name] = CharacterTraits(
                        name=name,
                        gender="Unknown",
                        demeanor="Unknown",
                        appearances=0
                    )
        
        # Create character-aware segmenter
        char_segmenter = CharacterAwareSegmenter(
            max_words_per_segment=max_words,
            min_words_per_segment=min_words,
            use_ollama=False,
            work_dir=context.work_dir
        )
        char_segmenter.analyzer = analyzer
        
        # Create base segments
        base_segmenter = TextSegmenter(
            target_words=segment_words,
            max_words=max_words,
            min_words=min_words,
            strip_unknown_tokens=strip_unknown
        )
        base_segments = base_segmenter.segment_text(text)
        
        # Create character-aware segments
        character_segments = char_segmenter.segment_text(text, base_segments)
        
        # Save character segments
        import json
        segments_file = Path(context.work_dir) / "character_segments.json"
        with open(segments_file, 'w', encoding='utf-8') as f:
            json.dump([seg.__dict__ for seg in character_segments], f, indent=2)
        
        return {
            'character_segments': character_segments,
            'segments_file': str(segments_file),
            'segment_count': len(character_segments),
            'character_mode': True
        }
    else:
        # Standard segmentation
        segmenter = TextSegmenter(
            target_words=segment_words,
            max_words=max_words,
            min_words=min_words,
            strip_unknown_tokens=strip_unknown
        )
        segments = segmenter.segment_text(text)
        
        # Save segments
        segments_file = Path(context.work_dir) / "segments.txt"
        with open(segments_file, 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(seg + "\n\n")
        
        return {
            'segments': segments,
            'segments_file': str(segments_file),
            'segment_count': len(segments),
            'character_mode': False
        }


@register_step(
    step_id="generate_audio",
    step_name="Generate Audio with TTS",
    order=4,
    required=True,
    max_retries=3  # Allow more retries since we can resume
)
def generate_audio_step(context: StepExecutionContext) -> Dict[str, Any]:
    """Generate audio using TTS with resume support for individual segments"""
    import json
    from tts_processor import TTSProcessor
    from character_voice_config import CharacterVoiceMapping, EmotionLibrary
    
    # Get segmentation results
    segment_result = context.get_previous_step_result('segment_text')
    character_mode = segment_result.get('character_mode', False)
    
    # Prepare output directory
    segments_dir = Path(context.work_dir) / "audio_segments"
    segments_dir.mkdir(exist_ok=True)
    
    # Progress tracking file
    progress_file = Path(context.work_dir) / "audio_generation_progress.json"
    
    # Load existing progress if resuming
    completed_segments = set()
    audio_files = []
    
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            completed_segments = set(progress_data.get('completed_segments', []))
            audio_files = progress_data.get('audio_files', [])
            print(f"  📂 Resuming: {len(completed_segments)} segments already generated")
    
    # Initialize TTS (only once, even on resume)
    config_path = context.job_data.get('config', './checkpoints/config.yaml')
    model_dir = context.job_data.get('model_dir', './checkpoints')
    use_fp16 = context.job_data.get('use_fp16', False)
    device = context.job_data.get('device')
    use_cuda_kernel = not context.job_data.get('no_cuda_kernel', False)
    use_deepspeed = context.job_data.get('use_deepspeed', False)
    
    tts_processor = TTSProcessor(
        cfg_path=config_path,
        model_dir=model_dir,
        use_fp16=use_fp16,
        device=device,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed
    )
    
    # Generation parameters
    generation_kwargs = {
        "temperature": context.job_data.get('temperature', 0.8),
        "top_p": context.job_data.get('top_p', 0.8),
        "top_k": context.job_data.get('top_k', 30),
        "repetition_penalty": context.job_data.get('repetition_penalty', 10.0),
        "length_penalty": context.job_data.get('length_penalty', 0.0),
        "num_beams": context.job_data.get('num_beams', 3),
    }
    
    # Helper function to save progress
    def save_progress():
        with open(progress_file, 'w') as f:
            json.dump({
                'completed_segments': list(completed_segments),
                'audio_files': audio_files,
                'total_segments': total_segments,
                'progress_percentage': (len(completed_segments) / total_segments * 100) if total_segments > 0 else 0
            }, f, indent=2)
    
    total_segments = 0
    
    if character_mode:
        # Character-aware TTS with resume support
        segments_file = segment_result.get('segments_file')
        with open(segments_file, 'r') as f:
            char_seg_data = json.load(f)
        
        # Reconstruct character segments
        from character_segmenter import CharacterSegment, EmotionalState
        character_segments = []
        for data in char_seg_data:
            char_seg = CharacterSegment(
                text=data['text'],
                start_pos=data['start_pos'],
                end_pos=data['end_pos'],
                character=data.get('character'),
                is_dialogue=data.get('is_dialogue', False),
                is_thought=data.get('is_thought', False),
                is_narration=data.get('is_narration', True),
                confidence=data.get('confidence', 0.0)
            )
            if 'emotional_state' in data:
                char_seg.emotional_state = EmotionalState(**data['emotional_state'])
            character_segments.append(char_seg)
        
        total_segments = len(character_segments)
        
        # Load voice mapping and emotion library
        voice_mapping = CharacterVoiceMapping.load(context.job_data['character_config'])
        emotion_library = None
        if context.job_data.get('emotion_library'):
            emotion_library = EmotionLibrary.load(context.job_data['emotion_library'])
        
        # Process each segment (skip already completed ones)
        for i, char_seg in enumerate(character_segments):
            # Check if this segment was already processed
            if i in completed_segments:
                print(f"  ⏭️  Skipping segment {i+1}/{total_segments} (already completed)")
                continue
            
            try:
                voice_config = voice_mapping.get_voice_for_character(
                    char_seg.character,
                    char_seg.is_narration
                )
                
                emo_audio = None
                if emotion_library and char_seg.emotional_state.dominant_emotion:
                    emo_audio = emotion_library.get_emotion_audio(
                        char_seg.emotional_state.dominant_emotion
                    )
                
                if voice_config.emotion_audio:
                    emo_audio = voice_config.emotion_audio
                
                print(f"  🎤 Generating segment {i+1}/{total_segments} - {char_seg.character or 'NARRATOR'}")
                
                seg_audio = tts_processor.process_segments(
                    segments=[char_seg.text],
                    index=i,
                    index_total=total_segments,
                    output_dir=str(segments_dir),
                    spk_audio_prompt=voice_config.speaker_audio,
                    emo_audio_prompt=emo_audio,
                    emo_alpha=1.0,
                    interval_silence=context.job_data.get('interval_silence', 200),
                    verbose=False,
                    max_text_tokens_per_segment=context.job_data.get('max_text_tokens', 120),
                    **generation_kwargs
                )
                
                if seg_audio:
                    audio_files.extend(seg_audio)
                    completed_segments.add(i)
                    
                    # Save progress after each successful segment
                    save_progress()
                    print(f"  ✓ Segment {i+1}/{total_segments} completed ({len(completed_segments)}/{total_segments} total)")
                else:
                    print(f"  ⚠️  No audio generated for segment {i+1}")
                    
            except Exception as e:
                print(f"  ✗ Error generating segment {i+1}/{total_segments}: {e}")
                # Save progress before raising error
                save_progress()
                raise
    else:
        # Standard TTS with resume support
        segments = segment_result.get('segments', [])
        total_segments = len(segments)
        
        # Process each segment individually (skip already completed ones)
        for i, segment_text in enumerate(segments):
            # Check if this segment was already processed
            if i in completed_segments:
                print(f"  ⏭️  Skipping segment {i+1}/{total_segments} (already completed)")
                continue
            
            try:
                print(f"  🎤 Generating segment {i+1}/{total_segments}")
                
                seg_audio = tts_processor.process_segments(
                    segments=[segment_text],
                    index=i,
                    index_total=total_segments,
                    output_dir=str(segments_dir),
                    spk_audio_prompt=context.job_data.get('voice_ref_path'),
                    emo_audio_prompt=context.job_data.get('emo_audio_prompt'),
                    emo_alpha=context.job_data.get('emo_alpha', 1.0),
                    interval_silence=context.job_data.get('interval_silence', 200),
                    verbose=False,
                    max_text_tokens_per_segment=context.job_data.get('max_text_tokens', 120),
                    **generation_kwargs
                )
                
                if seg_audio:
                    audio_files.extend(seg_audio)
                    completed_segments.add(i)
                    
                    # Save progress after each successful segment
                    save_progress()
                    print(f"  ✓ Segment {i+1}/{total_segments} completed ({len(completed_segments)}/{total_segments} total)")
                else:
                    print(f"  ⚠️  No audio generated for segment {i+1}")
                    
            except Exception as e:
                print(f"  ✗ Error generating segment {i+1}/{total_segments}: {e}")
                # Save progress before raising error
                save_progress()
                raise
    
    # Final progress save
    save_progress()
    
    print(f"\n  ✓ Audio generation complete: {len(audio_files)} audio files generated")
    
    return {
        'audio_files': audio_files,
        'audio_count': len(audio_files),
        'segments_dir': str(segments_dir),
        'completed_segments': len(completed_segments),
        'total_segments': total_segments
    }


@register_step(
    step_id="merge_audio",
    step_name="Merge Audio Segments",
    order=5,
    required=True,
    max_retries=2
)
def merge_audio_step(context: StepExecutionContext) -> Dict[str, Any]:
    """Merge audio segments into final audiobook"""
    from audio_merger import AudioMerger
    
    # Get audio files from previous step
    audio_result = context.get_previous_step_result('generate_audio')
    audio_files = audio_result.get('audio_files', [])
    
    if not audio_files:
        raise ValueError("No audio files to merge")
    
    # Get metadata
    extract_result = context.get_previous_step_result('extract_text')
    metadata = extract_result.get('metadata', {}) if extract_result else {}
    
    segment_result = context.get_previous_step_result('segment_text')
    
    # Prepare merge metadata
    merge_metadata = {
        "title": metadata.get('title', 'Unknown'),
        "author": metadata.get('author', 'Unknown'),
        "album": metadata.get('album', metadata.get('title', 'Unknown')),
        "segments": len(audio_files),
        "generation_time_seconds": 0  # Will be calculated
    }
    
    if 'publisher' in metadata:
        merge_metadata['publisher'] = metadata['publisher']
    if 'date' in metadata:
        merge_metadata['date'] = metadata['date']
    if 'series' in metadata:
        merge_metadata['series'] = metadata['series']
    
    # Merge audio
    output_path = context.job_data.get('output_path')
    output_format = context.job_data.get('format', 'm4b')
    segment_silence = context.job_data.get('segment_silence', 500)
    
    merger = AudioMerger(silence_duration_ms=segment_silence)
    
    final_audio = merger.merge_with_metadata(
        audio_files=audio_files,
        output_path=output_path,
        metadata=merge_metadata,
        output_format=output_format
    )
    
    if not final_audio:
        raise ValueError("Failed to merge audio files")
    
    # Cleanup if requested
    keep_segments = context.job_data.get('keep_segments', False)
    if not keep_segments:
        for audio_file in audio_files:
            try:
                os.remove(audio_file)
            except:
                pass
    
    return {
        'output_path': final_audio,
        'format': output_format,
        'segments_cleaned': not keep_segments
    }
