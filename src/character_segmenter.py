"""
Enhanced text segmenter that segments by character and emotion
"""
import re
from typing import List, Tuple, Optional
from character_analyzer import CharacterAnalyzer, CharacterSegment


class CharacterAwareSegmenter:
    """Segment text based on characters, dialogue, and emotions"""
    
    def __init__(self, 
                 max_words_per_segment: int = 600,
                 min_words_per_segment: int = 50,
                 split_on_character_change: bool = True,
                 split_on_emotion_change: bool = True,
                 use_ollama: bool = False,
                 ollama_url: str = "http://host.docker.internal:11434",
                 ollama_model: str = "aratan/DeepSeek-R1-32B-Uncensored:latest",
                 work_dir: str = "./work"):
        """
        Initialize character-aware segmenter
        
        Args:
            max_words_per_segment (int): Maximum words per segment
            min_words_per_segment (int): Minimum words per segment
            split_on_character_change (bool): Create new segment when character changes
            split_on_emotion_change (bool): Create new segment when emotion changes significantly
            use_ollama (bool): Use Ollama for character attribution
            ollama_url (str): Ollama API URL
            ollama_model (str): Ollama model to use
            work_dir (str): Working directory for artifacts
        """
        self.max_words = max_words_per_segment
        self.min_words = min_words_per_segment
        self.split_on_character = split_on_character_change
        self.split_on_emotion = split_on_emotion_change
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.work_dir = work_dir
        self.analyzer = CharacterAnalyzer(
            use_ollama=use_ollama,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            work_dir=work_dir
        )
    
    def segment_text(self, text: str, base_segments: List[str] = None) -> List[CharacterSegment]:
        """
        Segment text into character-aware segments
        
        Args:
            text (str): Full text to segment
            base_segments (List[str]): Optional pre-segmented text
            
        Returns:
            List of CharacterSegment objects
        """
        # If no base segments provided, create them
        if base_segments is None:
            base_segments = self._create_base_segments(text)
        
        if self.use_ollama:
            print(f"  Using Ollama for character attribution...")
            # Use Ollama to intelligently attribute text to characters
            char_segments = self._segment_with_ollama(text, base_segments)
            self.unload_model()
        else:
            # Use heuristic method (original implementation)
            char_segments = self.analyzer.create_character_segments(text, base_segments)
        
        # Remove duplicate text spans (keep first occurrence)
        char_segments = self._deduplicate_segments(char_segments)
        
        # Further split segments if they're too long
        final_segments = []
        for seg in char_segments:
            if len(seg.text.split()) > self.max_words:
                # Split long segments
                sub_segments = self._split_long_segment(seg)
                final_segments.extend(sub_segments)
            else:
                final_segments.append(seg)
        
        # Renumber segments
        for i, seg in enumerate(final_segments):
            seg.segment_id = i
        
        return final_segments
    
    def _deduplicate_segments(self, segments: List[CharacterSegment]) -> List[CharacterSegment]:
        """
        Remove duplicate text spans from segments
        
        Args:
            segments: List of CharacterSegment objects
            
        Returns:
            Deduplicated list
        """
        seen_text = set()
        unique_segments = []
        
        for seg in segments:
            # Normalize text for comparison
            normalized = seg.text.strip().lower()
            
            if normalized not in seen_text:
                seen_text.add(normalized)
                unique_segments.append(seg)
            else:
                print(f"  Warning: Skipping duplicate segment for {seg.character or 'NARRATOR'}")
        
        return unique_segments
    
    def _segment_with_ollama(self, text: str, base_segments: List[str]) -> List[CharacterSegment]:
        """
        Use Ollama to intelligently segment text and attribute to characters
        
        Args:
            text (str): Full text
            base_segments (List[str]): Base text segments
            
        Returns:
            List of CharacterSegment objects
        """
        import requests
        import json
        import os
        from datetime import datetime
        
        # Create artifact directories matching ollama_processor.py pattern
        if self.work_dir:
            self.prompts_dir = os.path.join(self.work_dir, "ollama_segmentation", "prompts")
            self.original_text_dir = os.path.join(self.work_dir, "ollama_segmentation", "original_text")
            self.processed_text_dir = os.path.join(self.work_dir, "ollama_segmentation", "processed_text")
            
            os.makedirs(self.prompts_dir, exist_ok=True)
            os.makedirs(self.original_text_dir, exist_ok=True)
            os.makedirs(self.processed_text_dir, exist_ok=True)
        
        char_segments = []
        character_names = list(self.analyzer.characters.keys()) if self.analyzer.characters else []
        
        # Save session metadata
        if self.work_dir:
            metadata_file = os.path.join(self.work_dir, "ollama_segmentation", "session_metadata.txt")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Character Segmentation Session\n")
                f.write(f"{'='*80}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.ollama_model}\n")
                f.write(f"Ollama URL: {self.ollama_url}\n")
                f.write(f"Total Base Segments: {len(base_segments)}\n")
                f.write(f"Known Characters: {', '.join(character_names) if character_names else 'None'}\n")
                f.write(f"{'='*80}\n")
        
        print(f"  Processing {len(base_segments)} segments with Ollama...")
        
        for i, segment_text in enumerate(base_segments):
            # Build prompt for character attribution
            prompt = f"""Analyze this text segment and identify who is speaking or thinking in each part.

Known characters: {', '.join(character_names) if character_names else 'None detected yet'}

Text segment:
{segment_text}

For each distinct part of the text (dialogue, thought, or narration), provide:
1. The exact text (copy it verbatim)
2. Who is speaking/thinking (character name, or "NARRATOR" for narration)
3. Type: "dialogue", "thought", or "narration"
4. Dominant emotion: happy, sad, angry, afraid, surprised, disgusted, calm, or melancholic

Respond ONLY with a JSON array like this:
[
  {{
    "text": "exact text here",
    "character": "CharacterName or NARRATOR",
    "type": "dialogue",
    "emotion": "happy"
  }},
  ...
]

IMPORTANT: Do not duplicate any text. Each piece of text should appear in exactly ONE entry."""
            
            # Save prompt if work_dir is specified
            if self.work_dir:
                prompt_file = os.path.join(self.prompts_dir, f"segment_{i+1:04d}_prompt.txt")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(prompt)
            
            # Save original text if work_dir is specified
            if self.work_dir:
                original_file = os.path.join(self.original_text_dir, f"segment_{i+1:04d}.txt")
                with open(original_file, 'w', encoding='utf-8') as f:
                    f.write(segment_text)
            
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.3,
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '')
                    
                    # Save the raw processed text with <think> tags (if any) to artifacts
                    raw_processed_text = response_text if response_text else segment_text
                    
                    if self.work_dir:
                        processed_file = os.path.join(self.processed_text_dir, f"segment_{i+1:04d}.txt")
                        with open(processed_file, 'w', encoding='utf-8') as f:
                            f.write(raw_processed_text)
                    
                    # Remove <think> tags
                    response_text = self.analyzer._remove_think_tags(response_text)
                    
                    # Extract JSON
                    start = response_text.find('[')
                    end = response_text.rfind(']') + 1
                    
                    if start >= 0 and end > start:
                        json_str = response_text[start:end]
                        attributions = json.loads(json_str)
                        
                        for attr in attributions:
                            text_part = attr.get('text', '').strip()
                            if not text_part:
                                continue
                            
                            char_name = attr.get('character', 'NARRATOR')
                            if char_name == 'NARRATOR':
                                char_name = None
                            
                            seg_type = attr.get('type', 'narration').lower()
                            emotion_name = attr.get('emotion', 'calm').lower()
                            
                            # Create emotional state
                            emotion_state = self.analyzer.analyze_emotion(text_part)
                            # Override with Ollama's emotion
                            emotion_state.dominant_emotion = emotion_name
                            
                            # Create segment
                            char_seg = CharacterSegment(
                                segment_id=i,
                                text=text_part,
                                character=char_name,
                                is_dialogue=(seg_type == 'dialogue'),
                                is_thought=(seg_type == 'thought'),
                                is_narration=(seg_type == 'narration'),
                                emotional_state=emotion_state
                            )
                            char_segments.append(char_seg)
                        
                        # Save comparison file
                        if self.work_dir:
                            comparison_file = os.path.join(self.work_dir, "ollama_segmentation", f"segment_{i+1:04d}_comparison.txt")
                            with open(comparison_file, 'w', encoding='utf-8') as f:
                                f.write("="*80 + "\n")
                                f.write("ORIGINAL TEXT\n")
                                f.write("="*80 + "\n")
                                f.write(segment_text + "\n\n")
                                f.write("="*80 + "\n")
                                f.write("PROCESSED TEXT (with <think> tags if present)\n")
                                f.write("="*80 + "\n")
                                f.write(raw_processed_text + "\n\n")
                                f.write("="*80 + "\n")
                                f.write(f"CHARACTER ATTRIBUTION ({len(attributions)} parts)\n")
                                f.write("="*80 + "\n")
                                
                                for idx, attr in enumerate(attributions, 1):
                                    f.write(f"\nPART {idx}:\n")
                                    f.write(f"  Character: {attr.get('character', 'NARRATOR')}\n")
                                    f.write(f"  Type: {attr.get('type', 'narration')}\n")
                                    f.write(f"  Emotion: {attr.get('emotion', 'calm')}\n")
                                    f.write(f"  Text: {attr.get('text', '')}\n")
                        
                        print(f"    Segment {i+1}/{len(base_segments)}: {len(attributions)} parts attributed")
                    else:
                        # Fallback to heuristic
                        print(f"    Segment {i+1}/{len(base_segments)}: Ollama failed, using heuristic")
                        fallback_segs = self.analyzer.create_character_segments(segment_text, [segment_text])
                        char_segments.extend(fallback_segs)
                else:
                    # Fallback to heuristic
                    print(f"    Segment {i+1}/{len(base_segments)}: Ollama request failed (status {response.status_code}), using heuristic")
                    fallback_segs = self.analyzer.create_character_segments(segment_text, [segment_text])
                    char_segments.extend(fallback_segs)
                    
            except Exception as e:
                print(f"    Segment {i+1}/{len(base_segments)}: Error ({e}), using heuristic")
                fallback_segs = self.analyzer.create_character_segments(segment_text, [segment_text])
                char_segments.extend(fallback_segs)
        
        # Create processing summary
        if self.work_dir:
            summary_file = os.path.join(self.work_dir, "ollama_segmentation", "processing_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Character Segmentation Processing Summary\n")
                f.write(f"{'='*80}\n")
                f.write(f"Total base segments processed: {len(base_segments)}\n")
                f.write(f"Total character segments created: {len(char_segments)}\n")
                f.write(f"Model used: {self.ollama_model}\n")
                f.write(f"\nArtifacts saved in:\n")
                f.write(f"  - Prompts: {self.prompts_dir}\n")
                f.write(f"  - Original text: {self.original_text_dir}\n")
                f.write(f"  - Processed text: {self.processed_text_dir}\n")
                f.write(f"  - Comparisons: {os.path.join(self.work_dir, 'ollama_segmentation')}\n")
                f.write(f"{'='*80}\n")
            print(f"\nOllama segmentation artifacts saved to: {os.path.join(self.work_dir, 'ollama_segmentation')}")
        
        return char_segments

    def unload_model(self) -> bool:
        """
        Unload the model from VRAM to free up memory.
        This is useful before starting TTS processing.
        
        Returns:
            bool: True if model was successfully unloaded
        """
        try:
            # Send an empty prompt with keep_alive=0 to unload the model
            payload = {
                "model": self.model,
                "keep_alive": 0
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                print(f"Unloaded Ollama model '{self.model}' from VRAM")
                return True
            else:
                print(f"Failed to unload Ollama model: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error unloading Ollama model: {e}")
            return False
    
    def _create_base_segments(self, text: str) -> List[str]:
        """Create base text segments"""
        sentences = self._split_into_sentences(text)
        
        segments = []
        current_segment = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_word_count = len(sentence.split())
            
            if current_word_count > 0 and (current_word_count + sentence_word_count) > self.max_words:
                segments.append(' '.join(current_segment))
                current_segment = [sentence]
                current_word_count = sentence_word_count
            else:
                current_segment.append(sentence)
                current_word_count += sentence_word_count
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_segment(self, segment: CharacterSegment) -> List[CharacterSegment]:
        """Split a long segment into smaller ones"""
        words = segment.text.split()
        
        if len(words) <= self.max_words:
            return [segment]
        
        # Split into chunks
        sub_segments = []
        for i in range(0, len(words), self.max_words):
            chunk_words = words[i:i + self.max_words]
            chunk_text = ' '.join(chunk_words)
            
            # Create new segment with same metadata
            sub_seg = CharacterSegment(
                segment_id=segment.segment_id,
                text=chunk_text,
                character=segment.character,
                is_dialogue=segment.is_dialogue,
                is_thought=segment.is_thought,
                is_narration=segment.is_narration,
                emotional_state=segment.emotional_state
            )
            sub_segments.append(sub_seg)
        
        return sub_segments
    
    def merge_similar_segments(self, segments: List[CharacterSegment]) -> List[CharacterSegment]:
        """
        Merge consecutive segments with same character and emotion
        
        Args:
            segments (List[CharacterSegment]): Input segments
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            
            # Check if we should merge
            should_merge = (
                seg.character == last.character and
                seg.is_dialogue == last.is_dialogue and
                seg.is_thought == last.is_thought and
                seg.is_narration == last.is_narration and
                seg.emotional_state.dominant_emotion == last.emotional_state.dominant_emotion and
                len(last.text.split()) + len(seg.text.split()) <= self.max_words
            )
            
            if should_merge:
                # Merge texts
                last.text += ' ' + seg.text
            else:
                merged.append(seg)
        
        # Renumber
        for i, seg in enumerate(merged):
            seg.segment_id = i
        
        return merged
    
    def get_segment_stats(self, segments: List[CharacterSegment]) -> dict:
        """Get statistics about segments"""
        if not segments:
            return {
                'total_segments': 0,
                'total_words': 0,
                'avg_words_per_segment': 0,
                'min_words': 0,
                'max_words': 0,
                'dialogue_segments': 0,
                'thought_segments': 0,
                'narration_segments': 0,
                'characters': []
            }
        
        word_counts = [len(seg.text.split()) for seg in segments]
        dialogue_count = sum(1 for seg in segments if seg.is_dialogue)
        thought_count = sum(1 for seg in segments if seg.is_thought)
        narration_count = sum(1 for seg in segments if seg.is_narration)
        
        characters = set(seg.character for seg in segments if seg.character)
        
        return {
            'total_segments': len(segments),
            'total_words': sum(word_counts),
            'avg_words_per_segment': sum(word_counts) / len(word_counts),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'dialogue_segments': dialogue_count,
            'thought_segments': thought_count,
            'narration_segments': narration_count,
            'characters': sorted(list(characters))
        }


if __name__ == "__main__":
    # Test the segmenter
    sample_text = """
    "Hello, how are you?" said John with a smile.
    Mary looked at him nervously. "I'm fine," she replied softly.
    John felt worried about her. (Why is she acting strange?) he thought.
    "Are you sure?" he asked gently.
    Sarah walked into the room, laughing. "What's going on here?"
    The narrator describes the scene with great detail. Everything was perfect.
    "Nothing!" Mary exclaimed, her voice shaking with fear.
    """
    
    segmenter = CharacterAwareSegmenter(max_words_per_segment=50)
    segments = segmenter.segment_text(sample_text)
    
    print(f"Created {len(segments)} character-aware segments:\n")
    for i, seg in enumerate(segments, 1):
        print(f"{i}. [{seg.character or 'NARRATOR'}] {seg.emotional_state.dominant_emotion}")
        print(f"   Type: {'Dialogue' if seg.is_dialogue else 'Thought' if seg.is_thought else 'Narration'}")
        print(f"   Text: {seg.text[:60]}...")
        print()
    
    stats = segmenter.get_segment_stats(segments)
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
