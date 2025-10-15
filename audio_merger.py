"""
Module for merging multiple audio files into a single output
"""
import os
import torch
import torchaudio
from typing import List


class AudioMerger:
    """Merge multiple audio files into a single output file"""
    
    def __init__(self, silence_duration_ms: int = 500):
        """
        Initialize audio merger
        
        Args:
            silence_duration_ms (int): Duration of silence between segments in milliseconds
        """
        self.silence_duration_ms = silence_duration_ms
    
    def merge_audio_files(self, audio_files: List[str], output_path: str, sampling_rate: int = 22050):
        """
        Merge multiple audio files into a single file
        
        Args:
            audio_files (List[str]): List of audio file paths to merge
            output_path (str): Path to save the merged audio
            sampling_rate (int): Target sampling rate
            
        Returns:
            str: Path to merged audio file
        """
        if not audio_files:
            print("No audio files to merge")
            return None
        
        print(f"\nMerging {len(audio_files)} audio files...")
        
        # Load all audio files
        waveforms = []
        
        for i, audio_file in enumerate(audio_files):
            if not os.path.exists(audio_file):
                print(f"Warning: Audio file not found: {audio_file}")
                continue
            
            print(f"Loading {i+1}/{len(audio_files)}: {audio_file}")
            
            try:
                waveform, sr = torchaudio.load(audio_file)
                
                # Resample if necessary
                if sr != sampling_rate:
                    resampler = torchaudio.transforms.Resample(sr, sampling_rate)
                    waveform = resampler(waveform)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                waveforms.append(waveform)
                
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
                continue
        
        if not waveforms:
            print("No valid audio files to merge")
            return None
        
        # Create silence tensor
        silence_samples = int(sampling_rate * self.silence_duration_ms / 1000.0)
        silence = torch.zeros(1, silence_samples)
        
        # Merge waveforms with silence between them
        merged_waveforms = []
        for i, waveform in enumerate(waveforms):
            merged_waveforms.append(waveform)
            
            # Add silence between segments (but not after the last one)
            if i < len(waveforms) - 1:
                merged_waveforms.append(silence)
        
        # Concatenate all waveforms
        final_waveform = torch.cat(merged_waveforms, dim=1)
        
        # Save merged audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, final_waveform, sampling_rate)
        
        duration = final_waveform.shape[1] / sampling_rate
        print(f"\n✓ Merged audio saved to: {output_path}")
        print(f"  Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Total segments: {len(waveforms)}")
        
        return output_path
    
    def merge_with_metadata(self, 
                           audio_files: List[str], 
                           output_path: str, 
                           metadata: dict = None,
                           sampling_rate: int = 22050):
        """
        Merge audio files and save metadata
        
        Args:
            audio_files (List[str]): List of audio file paths
            output_path (str): Path to save merged audio
            metadata (dict): Metadata to save alongside audio
            sampling_rate (int): Target sampling rate
            
        Returns:
            str: Path to merged audio file
        """
        # Merge audio
        result = self.merge_audio_files(audio_files, output_path, sampling_rate)
        
        # Save metadata if provided
        if result and metadata:
            metadata_path = output_path.replace('.wav', '_metadata.txt')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            print(f"✓ Metadata saved to: {metadata_path}")
        
        return result


if __name__ == "__main__":
    # Test the audio merger
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python audio_merger.py <output_file> <input_file1> <input_file2> ...")
        sys.exit(1)
    
    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    
    merger = AudioMerger(silence_duration_ms=500)
    merger.merge_audio_files(input_files, output_file)
