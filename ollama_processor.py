"""
Module for optional text processing using Ollama LLM
"""
import os
import requests
import json
from typing import Optional
from datetime import datetime


class OllamaProcessor:
    """Process text using Ollama for cleanup and enhancement"""
    
    def __init__(self, base_url="http://localhost:11434", model="llama2", work_dir=None):
        """
        Initialize Ollama processor
        
        Args:
            base_url (str): Base URL for Ollama API
            model (str): Model name to use
            work_dir (str): Working directory to save artifacts
        """
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"
        self.work_dir = work_dir
        
        # Create artifact directories if work_dir is specified
        if self.work_dir:
            self.prompts_dir = os.path.join(self.work_dir, "ollama", "prompts")
            self.original_text_dir = os.path.join(self.work_dir, "ollama", "original_text")
            self.processed_text_dir = os.path.join(self.work_dir, "ollama", "processed_text")
            
            os.makedirs(self.prompts_dir, exist_ok=True)
            os.makedirs(self.original_text_dir, exist_ok=True)
            os.makedirs(self.processed_text_dir, exist_ok=True)
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available
        
        Returns:
            bool: True if Ollama is accessible
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Ollama not available: {e}")
            return False
    
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
    
    def process_text(self, text: str, prompt_template: Optional[str] = None, segment_id: Optional[int] = None) -> str:
        """
        Process text using Ollama
        
        Args:
            text (str): Input text to process
            prompt_template (str, optional): Custom prompt template
            segment_id (int, optional): Segment ID for saving artifacts
            
        Returns:
            str: Processed text
        """
        if not self.is_available():
            print("Ollama is not available, returning original text")
            return text
        
        # Default prompt for cleaning up text for TTS
        if prompt_template is None:
            prompt_template = """Clean up the following text for text-to-speech conversion. 
Remove any formatting artifacts, fix obvious typos, and ensure the text flows naturally when read aloud.
Do not add or remove content, just clean it up. Return only the cleaned text without any explanation.
Cleaning up the text may include fixing spacing issues, punctuation, and minor grammar corrections.
Also focus on making the text sound natural when spoken. And do not change the meaning of the text.
While editing the text, add *'s around phases that are NOT dialogue (narration, description, character thoughts, etc.) so they can be spoken differently by the TTS.
For example here is how you should format character thoughts, *I thought to myself, I should have known better.*
Also when pronouns like he, she, him, her, etc are used in narration or description, or thoughts, replace them with the character's name if known.
When you encounter a section where a character is reasoning or thinking through something complex, and it is not part of the main dialogue, encapsulate that reasoning within *'s. This helps to differentiate between spoken dialogue and internal thought processes.
For example: *He pondered the implications of his decision, weighing the pros and cons carefully.* This indicates that the text is a thought or reasoning process, not spoken aloud.
When a character is speaking, do not add *'s around their dialogue and do not change the content of their speech or the quotes.
When a piece of text is ambiguous about who is speaking or thinking/reasoning, try to infer the speaker or thinker from context and replace pronouns with the character's name if known and/or add context (e.g., "John said," "Mary thought") to the text for clarity and try to stay in character.
ONLY use *'s for narration, description, and character thoughts/reasoning. Do NOT use *'s for dialogue or direct quotes and DO NOT use *'s for single words or character names to express emphasis that will break the TTS.
Do use *'s for wrapping non-dialogue text so it can be spoken differently by the TTS.

Text to clean:
{text}

Cleaned text:"""
        
        prompt = prompt_template.format(text=text)
        
        # Save original text if work_dir is specified
        if self.work_dir and segment_id is not None:
            original_file = os.path.join(self.original_text_dir, f"segment_{segment_id:04d}.txt")
            with open(original_file, 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Save prompt if work_dir is specified
        if self.work_dir and segment_id is not None:
            prompt_file = os.path.join(self.prompts_dir, f"segment_{segment_id:04d}_prompt.txt")
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300  # Longer timeout for processing
            )
            
            if response.status_code == 200:
                result = response.json()
                processed_text = result.get('response', '').strip()
                
                # Save the raw processed text with <think> tags (if any) to artifacts
                raw_processed_text = processed_text if processed_text else text
                
                # Remove <think> tags for TTS but keep the content in artifacts
                final_text = self._remove_think_tags(processed_text) if processed_text else text
                
                # Save processed text (with <think> tags) if work_dir is specified
                if self.work_dir and segment_id is not None:
                    processed_file = os.path.join(self.processed_text_dir, f"segment_{segment_id:04d}.txt")
                    with open(processed_file, 'w', encoding='utf-8') as f:
                        f.write(raw_processed_text)
                    
                    # Also save a comparison file
                    comparison_file = os.path.join(self.work_dir, "ollama", f"segment_{segment_id:04d}_comparison.txt")
                    with open(comparison_file, 'w', encoding='utf-8') as f:
                        f.write("="*80 + "\n")
                        f.write("ORIGINAL TEXT\n")
                        f.write("="*80 + "\n")
                        f.write(text + "\n\n")
                        f.write("="*80 + "\n")
                        f.write("PROCESSED TEXT (with <think> tags if present)\n")
                        f.write("="*80 + "\n")
                        f.write(raw_processed_text + "\n\n")
                        f.write("="*80 + "\n")
                        f.write("FINAL TEXT (for TTS, <think> tags removed)\n")
                        f.write("="*80 + "\n")
                        f.write(final_text + "\n")
                
                return final_text
            else:
                print(f"Ollama API error: {response.status_code}")
                return text
                
        except Exception as e:
            print(f"Error processing text with Ollama: {e}")
            return text
    
    def _remove_think_tags(self, text: str) -> str:
        """
        Remove <think>...</think> tags and their content from text.
        This is used for models like DeepSeek-R1 that include reasoning in <think> tags.
        
        Args:
            text (str): Text potentially containing <think> tags
            
        Returns:
            str: Text with <think> tags and their content removed
        """
        import re
        
        # Remove <think>...</think> blocks (including newlines within)
        # Use re.DOTALL to make . match newlines as well
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any standalone opening or closing think tags that might remain
        cleaned_text = re.sub(r'</?think>', '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up excessive whitespace that might be left after removing tags
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Multiple blank lines to double
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def process_segments(self, segments: list, show_progress: bool = True) -> list:
        """
        Process multiple text segments
        
        Args:
            segments (list): List of text segments
            show_progress (bool): Whether to show progress
            
        Returns:
            list: List of processed segments
        """
        if not self.is_available():
            print("Ollama is not available, skipping text processing")
            return segments
        
        processed_segments = []
        total = len(segments)
        
        # Save session metadata
        if self.work_dir:
            metadata_file = os.path.join(self.work_dir, "ollama", "session_metadata.txt")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Ollama Processing Session\n")
                f.write(f"{'='*80}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Base URL: {self.base_url}\n")
                f.write(f"Total Segments: {total}\n")
                f.write(f"{'='*80}\n")
        
        for i, segment in enumerate(segments):
            if show_progress:
                print(f"Processing segment {i+1}/{total} with Ollama...")
            
            processed = self.process_text(segment, segment_id=i+1)
            processed_segments.append(processed)
        
        # Create summary file
        if self.work_dir:
            summary_file = os.path.join(self.work_dir, "ollama", "processing_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Ollama Processing Summary\n")
                f.write(f"{'='*80}\n")
                f.write(f"Total segments processed: {len(processed_segments)}\n")
                f.write(f"Model used: {self.model}\n")
                f.write(f"\nArtifacts saved in:\n")
                f.write(f"  - Prompts: {self.prompts_dir}\n")
                f.write(f"  - Original text: {self.original_text_dir}\n")
                f.write(f"  - Processed text: {self.processed_text_dir}\n")
                f.write(f"  - Comparisons: {os.path.join(self.work_dir, 'ollama')}\n")
                f.write(f"{'='*80}\n")
            print(f"\nOllama artifacts saved to: {os.path.join(self.work_dir, 'ollama')}")
        
        # Unload the model to free up VRAM for TTS
        print(f"\nUnloading Ollama model to free VRAM for TTS processing...")
        # self.unload_model()
        
        return processed_segments


if __name__ == "__main__":
    # Test the Ollama processor
    processor = OllamaProcessor()
    
    if processor.is_available():
        print("Ollama is available!")
        
        test_text = """This  is   a test   text with   extra    spaces.
        And some     formatting    issues."""
        
        processed = processor.process_text(test_text)
        print(f"\nOriginal:\n{test_text}")
        print(f"\nProcessed:\n{processed}")
    else:
        print("Ollama is not available. Make sure it's running.")
