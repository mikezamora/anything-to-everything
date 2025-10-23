"""
Module for segmenting text into manageable chunks for TTS processing
"""
import re
from typing import List


class TextSegmenter:
    """Split text into segments based on word count while respecting sentence boundaries"""
    
    def __init__(self, target_words=500, max_words=600, min_words=100, strip_unknown_tokens=True):
        """
        Initialize the text segmenter
        
        Args:
            target_words (int): Target number of words per segment
            max_words (int): Maximum words per segment (hard limit)
            min_words (int): Minimum words for a segment to be considered valid
            strip_unknown_tokens (bool): Whether to remove tokens that might cause TTS issues
        """
        self.target_words = target_words
        self.max_words = max_words
        self.min_words = min_words
        self.strip_unknown_tokens = strip_unknown_tokens
    
    def segment_text(self, text: str) -> List[str]:
        """
        Segment text into chunks
        
        Args:
            text (str): Input text to segment
            
        Returns:
            List[str]: List of text segments
        """
        # Clean unknown tokens first if enabled
        if self.strip_unknown_tokens:
            text = self.clean_unknown_tokens(text)
        
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
    
    def clean_unknown_tokens(self, text: str) -> str:
        """
        Remove tokens that commonly cause TTS encoding issues while preserving normal punctuation
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not self.strip_unknown_tokens:
            return text
        
        # Define characters to remove (common problematic tokens for TTS)
        # Keep normal punctuation: . , ! ? : ; " ' ( ) - 
        # Remove: = # * + ~ ^ @ $ % & | \ / < > [ ] { } _ `
        problematic_chars = ['=', '#', '*', '+', '~', '^', '@', '$', '%', '&', '|', '\\', '/', '<', '>', '[', ']', '{', '}', '_', '`']
        
        cleaned_text = text
        
        # Remove standalone problematic characters (surrounded by spaces or at boundaries)
        for char in problematic_chars:
            # Remove standalone characters (surrounded by whitespace)
            pattern = r'\s+' + re.escape(char) + r'+\s+'
            cleaned_text = re.sub(pattern, ' ', cleaned_text)
            
            # Remove at beginning/end of text
            pattern = r'^' + re.escape(char) + r'+\s*'
            cleaned_text = re.sub(pattern, '', cleaned_text)
            
            pattern = r'\s*' + re.escape(char) + r'+$'
            cleaned_text = re.sub(pattern, '', cleaned_text)
            
            # Remove sequences of the same character (like ===== or ***)
            pattern = re.escape(char) + r'{2,}'
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # Clean up multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Clean up leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def get_segment_info(self, segments: List[str]) -> dict:
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
