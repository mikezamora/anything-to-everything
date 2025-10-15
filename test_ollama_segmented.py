"""
Test script for segmented Ollama character detection with artifact saving
"""

from character_analyzer import CharacterAnalyzer
import os
from pathlib import Path

def test_segmented_detection():
    """Test the new segmented Ollama character detection"""
    
    # Sample text with <think> tags
    test_text = """
    <think>This is internal reasoning that should be removed</think>
    
    John walked into the room with confidence. He had always been a serious person.
    
    "Hello," said Mary cheerfully. She was known for her energetic personality.
    
    <think>Another thought to remove</think>
    
    "Good to see you," John replied. His calm demeanor was reassuring.
    
    Later, Tom arrived. He seemed nervous about the meeting.
    
    Mary smiled at him warmly. She tried to make everyone feel comfortable.
    """
    
    # Create analyzer with Ollama
    analyzer = CharacterAnalyzer(
        use_ollama=True,
        ollama_url="http://localhost:11434",
        ollama_model="llama2",
        work_dir="./test_work",
        segment_words=500
    )
    
    print("Testing segmented Ollama character detection...")
    print("=" * 60)
    
    # Test <think> tag removal
    print("\n1. Testing <think> tag removal:")
    cleaned = analyzer._remove_think_tags(test_text)
    print(f"   Original length: {len(test_text)}")
    print(f"   Cleaned length: {len(cleaned)}")
    print(f"   <think> tags present: {'<think>' in cleaned}")
    
    # Test text segmentation
    print("\n2. Testing text segmentation:")
    segments = analyzer._split_text_into_segments(test_text, target_chars=100)
    print(f"   Number of segments: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"   Segment {i+1}: {len(seg)} chars")
    
    # Test full detection (only if Ollama is available)
    print("\n3. Testing full Ollama detection:")
    if analyzer._check_ollama_available():
        try:
            characters = analyzer.detect_characters(test_text)
            print(f"   Detected {len(characters)} characters:")
            for name, traits in characters.items():
                print(f"     - {name}: {traits.gender}, {traits.demeanor}, {traits.appearances} appearances")
            
            # Check artifacts
            print("\n4. Checking saved artifacts:")
            work_dir = Path("./test_work/character_detection")
            if work_dir.exists():
                for subdir in ['prompts', 'inputs', 'outputs', 'comparisons']:
                    subdir_path = work_dir / subdir
                    if subdir_path.exists():
                        files = list(subdir_path.glob("*"))
                        print(f"   {subdir}/: {len(files)} files")
                
                summary_file = work_dir / "processing_summary.txt"
                if summary_file.exists():
                    print(f"   ✓ processing_summary.txt exists")
                    print("\n   Summary content:")
                    with open(summary_file, 'r') as f:
                        print("   " + "\n   ".join(f.read().split('\n')))
        except Exception as e:
            print(f"   ⚠ Error during detection: {e}")
    else:
        print("   ⚠ Ollama not available - skipping full detection test")
        print("   You can still test <think> removal and segmentation above")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print(f"Check ./test_work/character_detection/ for artifacts")

if __name__ == "__main__":
    test_segmented_detection()
