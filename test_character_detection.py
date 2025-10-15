"""
Quick test script for character detection
Run this to verify the character analysis system works
"""
from character_analyzer import CharacterAnalyzer

# Sample text with multiple characters and emotions
sample_text = """
"Good morning!" said Sarah with a cheerful smile. She walked into the kitchen, humming a happy tune.

John looked up from his newspaper, annoyed. "Could you be any louder?" he snapped.

Sarah's smile faded. "I'm sorry," she replied softly, feeling hurt. (Why is he always so angry?) she thought.

"It's fine," John muttered, immediately regretting his harsh tone. He felt guilty for being so irritable.

Mary entered the room, laughing. "What's going on in here? You two look so serious!"

"Nothing," Sarah said quietly, her voice trembling with sadness.

John stood up abruptly. "I need some air," he said, walking out. His heart was pounding with anxiety.

Mary sat down next to Sarah. "Don't worry about him," she said gently, putting a comforting hand on Sarah's shoulder. "He's just stressed about work."

Sarah nodded, tears forming in her eyes. "I know," she whispered. "I just want everyone to be happy."

"We will be," Mary assured her with a warm smile. "Everything will work out."

The narrator observes that the morning had started with such promise, but quickly turned difficult. 
The three friends would need to talk things through later, when emotions had calmed down.
"""

print("="*70)
print("CHARACTER DETECTION TEST")
print("="*70)

# Initialize analyzer
analyzer = CharacterAnalyzer()

# Detect characters
print("\n1. DETECTING CHARACTERS...")
characters = analyzer.detect_characters(sample_text)

print(f"\nDetected {len(characters)} characters:")
for name, traits in sorted(characters.items(), key=lambda x: x[1].appearances, reverse=True):
    print(f"\n  {name}:")
    print(f"    Gender: {traits.gender}")
    print(f"    Demeanor: {traits.demeanor}")
    print(f"    Appearances: {traits.appearances}")

# Extract dialogue and thoughts
print("\n" + "="*70)
print("2. EXTRACTING DIALOGUE AND THOUGHTS...")
print("="*70)

dialogue_thoughts = analyzer.extract_dialogue_and_thoughts(sample_text)
print(f"\nFound {len(dialogue_thoughts)} dialogue/thought segments:")

for text, speaker, is_dialogue, is_thought in dialogue_thoughts:
    type_str = "DIALOGUE" if is_dialogue else "THOUGHT"
    speaker_str = speaker if speaker else "Unknown"
    print(f"\n[{type_str}] {speaker_str}:")
    print(f"  \"{text}\"")

# Analyze emotions
print("\n" + "="*70)
print("3. ANALYZING EMOTIONS...")
print("="*70)

# Split into sentences for emotion analysis
sentences = [
    "Sarah walked into the kitchen, humming a happy tune.",
    "John looked up from his newspaper, annoyed.",
    "Sarah's smile faded. She replied softly, feeling hurt.",
    "John felt guilty for being so irritable.",
    "Mary entered the room, laughing.",
    "Sarah said quietly, her voice trembling with sadness.",
    "His heart was pounding with anxiety.",
    "Mary said gently, putting a comforting hand on Sarah's shoulder.",
]

for sentence in sentences:
    emotion = analyzer.analyze_emotion(sentence)
    print(f"\nText: {sentence[:60]}...")
    print(f"  Emotion: {emotion.dominant_emotion} (intensity: {emotion.intensity:.2f})")
    # Show top 3 emotions
    top_emotions = sorted(emotion.emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top emotions: {', '.join([f'{e}:{v:.2f}' for e, v in top_emotions])}")

# Create character segments
print("\n" + "="*70)
print("4. CREATING CHARACTER SEGMENTS...")
print("="*70)

# Split text into base segments (simple split by paragraphs)
paragraphs = [p.strip() for p in sample_text.split('\n\n') if p.strip()]

character_segments = analyzer.create_character_segments(sample_text, paragraphs)

print(f"\nCreated {len(character_segments)} character-aware segments:")

for i, seg in enumerate(character_segments, 1):
    seg_type = "Dialogue" if seg.is_dialogue else "Thought" if seg.is_thought else "Narration"
    char_name = seg.character if seg.character else "NARRATOR"
    
    print(f"\n{i}. [{char_name}] - {seg_type} - {seg.emotional_state.dominant_emotion}")
    print(f"   Text: {seg.text[:70]}...")

print("\n" + "="*70)
print("5. TESTING PRONOUN RESOLUTION...")
print("="*70)

# Test pronoun resolution
test_cases = [
    ("John walked in. He smiled.", "John was nervous before."),
    ("Mary spoke softly. She was afraid.", "Mary entered the room."),
    ("I don't know what to do.", "John thought to himself."),
]

print("\nPronoun resolution tests:")
for text, context in test_cases:
    resolved = analyzer.resolve_pronoun_to_character(text, context)
    print(f"\nText: \"{text}\"")
    print(f"Context: \"{context}\"")
    print(f"Resolved to: {resolved if resolved else 'None'}")

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
print("\nIf you see character names, emotions, segments, and pronoun resolution,")
print("the character analysis system is working correctly!")
print("\nTo test with Ollama (more accurate), run with your Ollama server:")
print("  Set use_ollama=True in CharacterAnalyzer initialization")
