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
                 split_on_emotion_change: bool = True):
        """
        Initialize character-aware segmenter
        
        Args:
            max_words_per_segment (int): Maximum words per segment
            min_words_per_segment (int): Minimum words per segment
            split_on_character_change (bool): Create new segment when character changes
            split_on_emotion_change (bool): Create new segment when emotion changes significantly
        """
        self.max_words = max_words_per_segment
        self.min_words = min_words_per_segment
        self.split_on_character = split_on_character_change
        self.split_on_emotion = split_on_emotion_change
        self.analyzer = CharacterAnalyzer()
    
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
        
        # Analyze and create character segments
        char_segments = self.analyzer.create_character_segments(text, base_segments)
        
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
