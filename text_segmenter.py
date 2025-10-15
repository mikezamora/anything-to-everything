"""
Module for segmenting text into manageable chunks for TTS processing
"""
import re
from typing import List


class TextSegmenter:
    """Split text into segments based on word count while respecting sentence boundaries"""
    
    def __init__(self, target_words=500, max_words=600, min_words=100):
        """
        Initialize the text segmenter
        
        Args:
            target_words (int): Target number of words per segment
            max_words (int): Maximum words per segment (hard limit)
            min_words (int): Minimum words for a segment to be considered valid
        """
        self.target_words = target_words
        self.max_words = max_words
        self.min_words = min_words
    
    def segment_text(self, text: str) -> List[str]:
        """
        Segment text into chunks
        
        Args:
            text (str): Input text to segment
            
        Returns:
            List[str]: List of text segments
        """
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        segments = []
        current_segment = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence would exceed max_words, start a new segment
            if current_word_count > 0 and (current_word_count + sentence_word_count) > self.max_words:
                # Save current segment
                segments.append(' '.join(current_segment))
                current_segment = [sentence]
                current_word_count = sentence_word_count
            
            # If we're at or above target and adding would go over, start new segment
            elif current_word_count >= self.target_words and (current_word_count + sentence_word_count) > self.target_words:
                # Save current segment
                segments.append(' '.join(current_segment))
                current_segment = [sentence]
                current_word_count = sentence_word_count
            
            else:
                # Add sentence to current segment
                current_segment.append(sentence)
                current_word_count += sentence_word_count
        
        # Add remaining segment
        if current_segment:
            segment_text = ' '.join(current_segment)
            # Only add if it meets minimum word count, otherwise merge with previous
            if len(segment_text.split()) >= self.min_words or not segments:
                segments.append(segment_text)
            elif segments:
                # Merge with previous segment if too small
                segments[-1] = segments[-1] + ' ' + segment_text
        
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'\bMr\.', 'Mr', text)
        text = re.sub(r'\bMrs\.', 'Mrs', text)
        text = re.sub(r'\bDr\.', 'Dr', text)
        text = re.sub(r'\bMs\.', 'Ms', text)
        text = re.sub(r'\b([A-Z])\.', r'\1', text)  # Single letter abbreviations
        
        # Split on sentence boundaries
        # Look for periods, question marks, or exclamation marks followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        return sentences
    
    def get_segment_stats(self, segments: List[str]) -> dict:
        """
        Get statistics about the segments
        
        Args:
            segments (List[str]): List of text segments
            
        Returns:
            dict: Statistics dictionary
        """
        word_counts = [len(seg.split()) for seg in segments]
        
        return {
            'total_segments': len(segments),
            'total_words': sum(word_counts),
            'avg_words_per_segment': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min_words': min(word_counts) if word_counts else 0,
            'max_words': max(word_counts) if word_counts else 0
        }


if __name__ == "__main__":
    # Test the segmenter
    sample_text = """
    This is a test sentence. This is another test sentence. And here's a third one.
    This paragraph has multiple sentences that should be grouped together into segments.
    """ * 100  # Repeat to get enough text
    
    segmenter = TextSegmenter(target_words=500, max_words=600)
    segments = segmenter.segment_text(sample_text)
    
    print(f"Created {len(segments)} segments")
    stats = segmenter.get_segment_stats(segments)
    print(f"Statistics: {stats}")
    
    # Print first few segments
    for i, seg in enumerate(segments[:3]):
        print(f"\nSegment {i+1} ({len(seg.split())} words):")
        print(seg[:200] + "..." if len(seg) > 200 else seg)
