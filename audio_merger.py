"""
Module for merging multiple audio files into a single output
"""
import os
import subprocess
import torch
import torchaudio
from typing import List, Optional
from pathlib import Path


class AudioMerger:
    """Merge multiple audio files into a single output file"""
    
    def __init__(self, silence_duration_ms: int = 500):
        """
        Initialize audio merger
        
        Args:
            silence_duration_ms (int): Duration of silence between segments in milliseconds
        """
        self.silence_duration_ms = silence_duration_ms
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available for M4B conversion"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, 
                         check=True,
                         timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _ffmpeg_available(self) -> bool:
        """Check if FFmpeg is available"""
        return self._check_ffmpeg()
    
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
    
    def _convert_to_m4b(self, wav_path: str, m4b_path: str, metadata: dict = None) -> bool:
        """
        Convert WAV file to M4B format using FFmpeg
        
        Args:
            wav_path (str): Path to input WAV file
            m4b_path (str): Path to output M4B file
            metadata (dict): Metadata to embed in M4B file
            
        Returns:
            bool: True if conversion successful
        """
        if not self._ffmpeg_available():
            print("Error: FFmpeg not found. Please install FFmpeg to create M4B files.")
            print("Download from: https://ffmpeg.org/download.html")
            return False
        
        print(f"\nConverting to M4B format...")
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', wav_path,
            '-c:a', 'aac',
            '-b:a', '64k',  # Good quality for audiobooks
            '-f', 'mp4',    # M4B is based on MP4
        ]
        
        # Add metadata if provided
        if metadata:
            if 'title' in metadata:
                cmd.extend(['-metadata', f'title={metadata["title"]}'])
            if 'author' in metadata:
                cmd.extend(['-metadata', f'artist={metadata["author"]}'])
                cmd.extend(['-metadata', f'album_artist={metadata["author"]}'])
            if 'album' in metadata:
                cmd.extend(['-metadata', f'album={metadata["album"]}'])
            elif 'title' in metadata:
                cmd.extend(['-metadata', f'album={metadata["title"]}'])
            
            # Add genre for audiobook
            cmd.extend(['-metadata', 'genre=Audiobook'])
            
            # Add publisher if available
            if 'publisher' in metadata:
                cmd.extend(['-metadata', f'publisher={metadata["publisher"]}'])
            
            # Add date if available
            if 'date' in metadata:
                cmd.extend(['-metadata', f'date={metadata["date"]}'])
            
            # Add description as comment if available, otherwise use generation info
            if 'description' in metadata:
                # Truncate description if too long
                desc = metadata['description'][:500]
                cmd.extend(['-metadata', f'description={desc}'])
            
            # Add comment with generation info
            comment_parts = []
            if 'segments' in metadata:
                comment_parts.append(f"{metadata['segments']} segments")
            if 'total_words' in metadata:
                comment_parts.append(f"{metadata['total_words']} words")
            if 'series' in metadata:
                comment_parts.append(f"Series: {metadata['series']}")
            
            if comment_parts:
                comment = f"Generated audiobook: {', '.join(comment_parts)}"
                cmd.extend(['-metadata', f'comment={comment}'])
        
        cmd.extend(['-y', m4b_path])  # -y to overwrite without asking
        
        try:
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True,
                                  timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                print(f"✓ M4B file created: {m4b_path}")
                
                # Get file sizes for comparison
                wav_size = os.path.getsize(wav_path) / (1024 * 1024)  # MB
                m4b_size = os.path.getsize(m4b_path) / (1024 * 1024)  # MB
                print(f"  WAV size: {wav_size:.2f} MB")
                print(f"  M4B size: {m4b_size:.2f} MB (compression: {(1 - m4b_size/wav_size)*100:.1f}%)")
                return True
            else:
                print(f"Error converting to M4B: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Error: FFmpeg conversion timed out")
            return False
        except Exception as e:
            print(f"Error running FFmpeg: {e}")
            return False
    
    def merge_with_metadata(self, 
                           audio_files: List[str], 
                           output_path: str, 
                           metadata: dict = None,
                           sampling_rate: int = 22050,
                           output_format: str = 'wav'):
        """
        Merge audio files and save metadata
        
        Args:
            audio_files (List[str]): List of audio file paths
            output_path (str): Path to save merged audio
            metadata (dict): Metadata to save alongside audio
            sampling_rate (int): Target sampling rate
            output_format (str): Output format ('wav' or 'm4b')
            
        Returns:
            str: Path to merged audio file
        """
        output_format = output_format.lower()
        
        # Ensure output path has correct extension
        if output_format == 'm4b':
            # First create WAV, then convert to M4B
            wav_path = output_path.replace('.m4b', '.wav') if output_path.endswith('.m4b') else output_path + '.wav'
            final_output = output_path if output_path.endswith('.m4b') else output_path.replace('.wav', '.m4b')
        else:
            wav_path = output_path if output_path.endswith('.wav') else output_path + '.wav'
            final_output = wav_path
        
        # Merge audio to WAV first
        result = self.merge_audio_files(audio_files, wav_path, sampling_rate)
        
        if not result:
            return None
        
        # Convert to M4B if requested
        if output_format == 'm4b':
            if self._convert_to_m4b(wav_path, final_output, metadata):
                # Optionally remove intermediate WAV file
                try:
                    print(f"Removing intermediate WAV file: {wav_path}")
                    os.remove(wav_path)
                except Exception as e:
                    print(f"Warning: Could not remove intermediate WAV file: {e}")
                result = final_output
            else:
                print(f"Warning: M4B conversion failed, keeping WAV file: {wav_path}")
                result = wav_path
        
        # Save metadata text file if provided
        if result and metadata:
            if result.endswith('.m4b'):
                metadata_path = result.replace('.m4b', '_metadata.txt')
            else:
                metadata_path = result.replace('.wav', '_metadata.txt')
                
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
