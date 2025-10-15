"""
Module for processing text segments with IndexTTS2
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path to import IndexTTS2
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from indextts.infer_v2 import IndexTTS2


class TTSProcessor:
    """Process text segments using IndexTTS2"""
    
    def __init__(self, 
                 cfg_path="../../checkpoints/config.yaml",
                 model_dir="../../checkpoints",
                 use_fp16=False,
                 device=None,
                 use_cuda_kernel=None,
                 use_deepspeed=False):
        """
        Initialize TTS processor
        
        Args:
            cfg_path (str): Path to config file
            model_dir (str): Path to model directory
            use_fp16 (bool): Whether to use FP16
            device (str): Device to use
            use_cuda_kernel (bool): Whether to use CUDA kernel
            use_deepspeed (bool): Whether to use DeepSpeed
        """
        print("Initializing IndexTTS2...")
        self.tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=use_fp16,
            device=device,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed
        )
        print("IndexTTS2 initialized successfully")
    
    def process_segment(self,
                       segment_text: str,
                       output_path: str,
                       spk_audio_prompt: str,
                       emo_audio_prompt: str = None,
                       emo_alpha: float = 1.0,
                       emo_vector: list = None,
                       use_emo_text: bool = False,
                       emo_text: str = None,
                       use_random: bool = False,
                       interval_silence: int = 200,
                       verbose: bool = False,
                       max_text_tokens_per_segment: int = 120,
                       **generation_kwargs):
        """
        Process a single text segment
        
        Args:
            segment_text (str): Text to convert to speech
            output_path (str): Path to save the audio file
            spk_audio_prompt (str): Path to speaker audio reference
            emo_audio_prompt (str): Path to emotion audio reference
            emo_alpha (float): Emotion alpha value
            emo_vector (list): Emotion vector
            use_emo_text (bool): Whether to use emotion from text
            emo_text (str): Emotion text
            use_random (bool): Whether to use random emotion
            interval_silence (int): Silence interval in ms
            verbose (bool): Verbose output
            max_text_tokens_per_segment (int): Max tokens per segment
            **generation_kwargs: Additional generation parameters
            
        Returns:
            str: Path to generated audio file
        """
        print(f"Processing segment: {len(segment_text)} characters")
        
        start_time = time.time()
        
        result = self.tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=segment_text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_prompt,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=use_random,
            interval_silence=interval_silence,
            verbose=verbose,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            stream_return=False,
            **generation_kwargs
        )
        
        elapsed = time.time() - start_time
        print(f"Segment processed in {elapsed:.2f} seconds")
        
        return output_path
    
    def process_segments(self,
                        segments: str,
                        output_dir: str,
                        spk_audio_prompt: str,
                        emo_audio_prompt: str = None,
                        emo_alpha: float = 1.0,
                        emo_vector: list = None,
                        use_emo_text: bool = False,
                        use_random: bool = False,
                        interval_silence: int = 200,
                        verbose: bool = False,
                        max_text_tokens_per_segment: int = 120,
                        index: Optional[int] = None,
                        index_total: Optional[int] = None,
                        **generation_kwargs):
        """
        Process multiple text segments
        
        Args:
            segments (list): List of text segments
            output_dir (str): Directory to save audio files
            spk_audio_prompt (str): Path to speaker audio reference
            emo_audio_prompt (str): Path to emotion audio reference
            emo_alpha (float): Emotion alpha value
            emo_vector (list): Emotion vector
            use_emo_text (bool): Whether to use emotion from text
            use_random (bool): Whether to use random emotion
            interval_silence (int): Silence interval in ms
            verbose (bool): Verbose output
            max_text_tokens_per_segment (int): Max tokens per segment
            **generation_kwargs: Additional generation parameters
            
        Returns:
            list: List of paths to generated audio files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = []
        total = len(segments)
        
        print(f"\nProcessing {total} segments with IndexTTS2...")
        
        for i, segment in enumerate(segments):
            if index is not None and index_total is not None:
                i = index
                print(f"\n{'='*60}")
                print(f"Segment {i+1}/{index_total}")
                print(f"{'='*60}")
            else: 
                print(f"\n{'='*60}")
                print(f"Segment {i+1}/{total}")
                print(f"{'='*60}")
            
            output_path = os.path.join(output_dir, f"segment_{i+1:04d}.wav")
            
            try:
                self.process_segment(
                    segment_text=segment,
                    output_path=output_path,
                    spk_audio_prompt=spk_audio_prompt,
                    emo_audio_prompt=emo_audio_prompt,
                    emo_alpha=emo_alpha,
                    emo_vector=emo_vector,
                    use_emo_text=use_emo_text,
                    emo_text=None,  # Use segment text for emotion detection
                    use_random=use_random,
                    interval_silence=interval_silence,
                    verbose=verbose,
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    **generation_kwargs
                )
                
                audio_files.append(output_path)
                print(f"Segment {i+1} completed: {output_path}")
                
            except Exception as e:
                print(f"Error processing segment {i+1}: {e}")
                # Continue with next segment
                continue
        
        print(f"\n{'='*60}")
        print(f"Completed {len(audio_files)}/{total} segments")
        print(f"{'='*60}\n")
        
        return audio_files


if __name__ == "__main__":
    # Test the TTS processor
    processor = TTSProcessor()
    
    test_text = "This is a test of the TTS processor module."
    test_output = "test_output.wav"
    test_speaker = "examples/voice_01.wav"
    
    processor.process_segment(
        segment_text=test_text,
        output_path=test_output,
        spk_audio_prompt=test_speaker,
        verbose=True
    )
    
    print(f"Test audio saved to {test_output}")
