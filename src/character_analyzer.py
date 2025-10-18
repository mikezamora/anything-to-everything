"""
Module for analyzing and extracting characters from text
Detects characters, their traits (gender, demeanor), and emotional states
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class CharacterTraits:
    """Character traits detected from text"""
    name: str
    gender: str = "unknown"  # male, female, neutral, unknown
    demeanor: str = "neutral"  # calm, energetic, serious, playful, etc.
    appearances: int = 0
    dialogue_count: int = 0
    thought_count: int = 0
    first_appearance: int = -1  # Segment number
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class EmotionalState:
    """Emotional state for a text segment"""
    dominant_emotion: str  # happy, sad, angry, afraid, surprised, disgusted, neutral
    intensity: float  # 0.0 to 1.0
    emotions: Dict[str, float]  # Emotion vector
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class CharacterSegment:
    """A text segment with character and emotion information"""
    segment_id: int
    text: str
    character: Optional[str]  # Character speaking/thinking
    is_dialogue: bool
    is_thought: bool
    is_narration: bool
    emotional_state: EmotionalState
    
    def to_dict(self):
        data = asdict(self)
        data['emotional_state'] = self.emotional_state.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data):
        data['emotional_state'] = EmotionalState.from_dict(data['emotional_state'])
        return cls(**data)


class CharacterAnalyzer:
    """Analyze text for characters and their emotional states"""
    
    # Common patterns for dialogue and thoughts
    DIALOGUE_PATTERNS = [
        r'"([^"]+)"',  # Standard quotes
        r'"([^"]+)"',  # Curly quotes
        r'«([^»]+)»',  # French quotes
        r'„([^"]+)"',  # German quotes
    ]
    
    THOUGHT_PATTERNS = [
        r'\*([^*]+)\*',  # *thought*
        r'_([^_]+)_',    # _thought_
        r'\(([^)]+)\)',  # (thought)
    ]
    
    # Character attribution patterns
    SPEAKER_PATTERNS = [
        r'(?:said|says|asks|asked|replies|replied|whispers|whispered|shouts|shouted|yells|yelled|screams|screamed|mutters|muttered|exclaims|exclaimed)\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+(?:said|says|asks|asked|replies|replied|whispers|whispered|shouts|shouted|yells|yelled|screams|screamed|mutters|muttered|exclaims|exclaimed)',
        r'"[^"]+,"\s+([A-Z][a-z]+)\s+(?:said|says)',
        r'([A-Z][a-z]+):\s+"',  # Name: "dialogue"
    ]
    
    # Emotion keywords
    EMOTION_KEYWORDS = {
        'happy': ['happy', 'joy', 'delighted', 'pleased', 'cheerful', 'glad', 'smiled', 'laughed', 'grinned', 'chuckled'],
        'sad': ['sad', 'sorrow', 'grief', 'depressed', 'unhappy', 'miserable', 'cried', 'wept', 'tears', 'sobbed'],
        'angry': ['angry', 'rage', 'furious', 'mad', 'irritated', 'annoyed', 'shouted', 'yelled', 'screamed', 'snapped'],
        'afraid': ['afraid', 'fear', 'scared', 'terrified', 'frightened', 'anxious', 'worried', 'nervous', 'trembled', 'shook'],
        'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'startled', 'stunned', 'gasped', 'wondered'],
        'disgusted': ['disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'grimaced', 'scowled'],
        'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'composed', 'gentle', 'soft'],
        'melancholic': ['melancholy', 'wistful', 'nostalgic', 'pensive', 'reflective', 'somber', 'gloomy'],
    }
    
    # Gender indicators (simple heuristics)
    MALE_INDICATORS = ['he', 'him', 'his', 'mr', 'sir', 'lord', 'king', 'prince', 'brother', 'father', 'son', 'man', 'boy', 'gentleman']
    FEMALE_INDICATORS = ['she', 'her', 'hers', 'mrs', 'ms', 'miss', 'lady', 'queen', 'princess', 'sister', 'mother', 'daughter', 'woman', 'girl']
    
    def __init__(self, use_ollama: bool = False, ollama_url: str = "http://host.docker.internal:11434", ollama_model: str = "aratan/DeepSeek-R1-32B-Uncensored:latest", work_dir: str = "./work"):
        """
        Initialize character analyzer
        
        Args:
            use_ollama (bool): Use Ollama for advanced character/emotion analysis
            ollama_url (str): Ollama API URL
            ollama_model (str): Ollama model to use
            work_dir (str): Working directory to save artifacts
        """
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.work_dir = work_dir
        self.characters: Dict[str, CharacterTraits] = {}
        self.pronoun_map: Dict[str, str] = {}  # Maps pronouns to character names in context
        
        # Create artifact directories if using Ollama
        if self.use_ollama and self.work_dir:
            self.char_prompts_dir = os.path.join(self.work_dir, "ollama_characters", "prompts")
            self.char_inputs_dir = os.path.join(self.work_dir, "ollama_characters", "inputs")
            self.char_outputs_dir = os.path.join(self.work_dir, "ollama_characters", "outputs")
            
            os.makedirs(self.char_prompts_dir, exist_ok=True)
            os.makedirs(self.char_inputs_dir, exist_ok=True)
            os.makedirs(self.char_outputs_dir, exist_ok=True)
        
        # Check Ollama availability
        if self.use_ollama:
            self._check_ollama_available()
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            available = response.status_code == 200
            if available:
                print(f"Ollama available at {self.ollama_url}")
            return available
        except:
            print(f"Ollama not available at {self.ollama_url}")
            return False
    
    def _remove_think_tags(self, text: str) -> str:
        """
        Remove <think>...</think> tags and their content from text.
        
        Args:
            text (str): Text potentially containing <think> tags
            
        Returns:
            str: Text with <think> tags and their content removed
        """
        import re
        
        # Remove <think>...</think> blocks (including newlines within)
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any standalone opening or closing think tags
        cleaned_text = re.sub(r'</?think>', '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up excessive whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _split_text_into_segments(self, text: str, target_chars: int = 3000) -> List[str]:
        """
        Split text into segments for Ollama processing
        
        Args:
            text (str): Full text
            target_chars (int): Target characters per segment
            
        Returns:
            List of text segments
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        segments = []
        current_segment = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if current_length + para_length > target_chars and current_segment:
                # Save current segment
                segments.append('\n\n'.join(current_segment))
                current_segment = [para]
                current_length = para_length
            else:
                current_segment.append(para)
                current_length += para_length + 2  # +2 for \n\n
        
        if current_segment:
            segments.append('\n\n'.join(current_segment))
        
        return segments
        
    def extract_dialogue_and_thoughts(self, text: str) -> List[Tuple[str, str, bool, bool]]:
        """
        Extract dialogue and thoughts from text
        
        Args:
            text (str): Input text
            
        Returns:
            List of (text, speaker, is_dialogue, is_thought) tuples
        """
        segments = []
        
        # Find all dialogue
        for pattern in self.DIALOGUE_PATTERNS:
            for match in re.finditer(pattern, text):
                dialogue_text = match.group(1)
                speaker = self._find_speaker_near_match(text, match.start(), match.end())
                segments.append((dialogue_text, speaker, True, False))
        
        # Find all thoughts
        for pattern in self.THOUGHT_PATTERNS:
            for match in re.finditer(pattern, text):
                thought_text = match.group(1)
                # Thoughts are usually attributed to the viewpoint character
                thinker = self._find_speaker_near_match(text, match.start(), match.end())
                segments.append((thought_text, thinker, False, True))
        
        return segments
    
    def _find_speaker_near_match(self, text: str, start: int, end: int, window: int = 200) -> Optional[str]:
        """
        Find the speaker near a dialogue/thought match
        
        Args:
            text (str): Full text
            start (int): Match start position
            end (int): Match end position
            window (int): Characters to search before/after
            
        Returns:
            Speaker name or None
        """
        # Search in window around the match
        search_start = max(0, start - window)
        search_end = min(len(text), end + window)
        context = text[search_start:search_end]
        dialogue_text = text[start:end]
        
        # Try to find speaker by name
        for pattern in self.SPEAKER_PATTERNS:
            match = re.search(pattern, context)
            if match:
                speaker_name = match.group(1)
                # Verify this is a known character
                if speaker_name in self.characters:
                    return speaker_name
                # If not in characters yet, still return it
                return speaker_name
        
        # If no explicit speaker found, try pronoun resolution
        # Get more context (previous sentences)
        prev_context_start = max(0, start - 500)
        prev_context = text[prev_context_start:start]
        
        # Try to resolve pronouns in the dialogue itself
        resolved = self.resolve_pronoun_to_character(dialogue_text, prev_context)
        if resolved:
            return resolved
        
        # Try to resolve pronouns in the surrounding context
        resolved = self.resolve_pronoun_to_character(context, prev_context)
        if resolved:
            return resolved
        
        return None
    
    def detect_characters(self, text: str) -> Dict[str, CharacterTraits]:
        """
        Detect characters in text
        
        Args:
            text (str): Input text
            
        Returns:
            Dictionary of character names to traits
        """
        if self.use_ollama:
            return self._detect_characters_with_ollama(text)
        else:
            return self._detect_characters_heuristic(text)
    
    def _detect_characters_with_ollama(self, text: str) -> Dict[str, CharacterTraits]:
        """
        Detect characters using Ollama with segmented processing and artifact saving
        
        Args:
            text (str): Input text
            
        Returns:
            Dictionary of character names to traits
        """
        try:
            import requests
            import json
            from pathlib import Path
            
            print("  Using Ollama for character detection...")
            
            # Create work directories
            char_work_dir = Path(self.work_dir) / "character_detection"
            prompts_dir = char_work_dir / "prompts"
            inputs_dir = char_work_dir / "inputs"
            outputs_dir = char_work_dir / "outputs"
            comparisons_dir = char_work_dir / "comparisons"
            
            for dir_path in [prompts_dir, inputs_dir, outputs_dir, comparisons_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Remove <think> tags
            cleaned_text = self._remove_think_tags(text)
            
            # Split text into segments
            segments = self._split_text_into_segments(cleaned_text, target_chars=3000)
            
            print(f"  Processing {len(segments)} text segments with overlapping windows...")
            
            all_character_results = []
            
            # Process with overlapping windows
            for i in range(len(segments)):
                # Create overlapping window
                if i == 0:
                    # First segment: use segments 0-1
                    window_segments = segments[0:min(2, len(segments))]
                    window_label = f"segments_{i+1}-{min(i+2, len(segments))}"
                else:
                    # Subsequent segments: use previous, current, and next
                    start_idx = max(0, i-1)
                    end_idx = min(len(segments), i+2)
                    window_segments = segments[start_idx:end_idx]
                    window_label = f"segments_{start_idx+1}-{end_idx}"
                
                window_text = '\n\n'.join(window_segments)
                
                prompt = f"""Analyze this text and identify all characters (people). For each character, determine:
1. Character name
2. Gender (male, female, neutral, or unknown)
3. Personality/demeanor (one word: calm, energetic, nervous, serious, playful, etc.)

Text to analyze:
{window_text}

Respond ONLY with a JSON array like this:
[
  {{"name": "CharacterName", "gender": "male", "demeanor": "calm"}},
  {{"name": "AnotherName", "gender": "female", "demeanor": "nervous"}}
]

Do not include narrators, places, or non-character entities. Only list actual characters (people)."""
                
                # Save prompt
                prompt_file = prompts_dir / f"segment_{i+1:04d}_prompt.txt"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                # Save input text
                input_file = inputs_dir / f"segment_{i+1:04d}_input.txt"
                with open(input_file, 'w', encoding='utf-8') as f:
                    f.write(window_text)
                
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
                        
                        # Remove <think> tags from response
                        response_text = self._remove_think_tags(response_text)
                        
                        # Save output
                        output_file = outputs_dir / f"segment_{i+1:04d}_output.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(response_text)
                        
                        # Try to extract JSON from response
                        try:
                            # Find JSON array in response
                            start = response_text.find('[')
                            end = response_text.rfind(']') + 1
                            if start >= 0 and end > start:
                                json_str = response_text[start:end]
                                detected = json.loads(json_str)
                                all_character_results.append(detected)
                                
                                # Save comparison file
                                comparison_file = comparisons_dir / f"segment_{i+1:04d}_comparison.txt"
                                with open(comparison_file, 'w', encoding='utf-8') as f:
                                    f.write(f"=== Window: {window_label} ===\n\n")
                                    f.write(f"Characters Detected:\n")
                                    for char in detected:
                                        f.write(f"  - {char.get('name', 'Unknown')} ({char.get('gender', 'unknown')})\n")
                                        f.write(f"    Demeanor: {char.get('demeanor', 'N/A')}\n")
                                
                                print(f"Processed {window_label}: {len(detected)} characters")
                            else:
                                print(f"No JSON array found in {window_label}")
                                
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON for {window_label}: {e}")
                            # Save error info
                            comparison_file = comparisons_dir / f"segment_{i+1:04d}_comparison.txt"
                            with open(comparison_file, 'w', encoding='utf-8') as f:
                                f.write(f"=== Window: {window_label} ===\n\n")
                                f.write(f"ERROR: Failed to parse JSON\n")
                                f.write(f"Response: {response_text[:500]}\n")
                    else:
                        print(f"Request failed for {window_label}: {response.status_code}")
                        
                except requests.RequestException as e:
                    print(f"Request error for {window_label}: {e}")
            
            # Merge results from all segments
            characters = self._merge_character_results(all_character_results, text)
            
            # Save final summary
            summary_file = char_work_dir / "processing_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Character Detection Processing Summary\n")
                f.write(f"======================================\n\n")
                f.write(f"Total segments processed: {len(segments)}\n")
                f.write(f"Successful analyses: {len(all_character_results)}\n")
                f.write(f"Final merged characters: {len(characters)}\n\n")
                f.write(f"Detected Characters:\n")
                for name, traits in characters.items():
                    f.write(f"  - {name} ({traits.gender})\n")
                    f.write(f"    Demeanor: {traits.demeanor}\n")
                    f.write(f"    Appearances: {traits.appearances}\n")
            
            print(f"Ollama detected {len(characters)} characters")
            
            # Still run heuristic detection and merge results
            # heuristic_chars = self._detect_characters_heuristic(text)
            
            # Add any characters from heuristic that Ollama missed
            # for name, traits in heuristic_chars.items():
            #     if name not in characters and traits.appearances >= 5:
            #         characters[name] = traits
            #         print(f"  + Added '{name}' from heuristic detection")
            
            self.characters = characters
            
            # Build pronoun map
            self._build_pronoun_map(text)
            
            return characters
                    
        except Exception as e:
            print(f"Ollama error: {e}, falling back to heuristic detection")
        
        # Fallback to heuristic
        return self._detect_characters_heuristic(text)
    
    def _merge_character_results(self, results: List[List[dict]], full_text: str) -> Dict[str, CharacterTraits]:
        """
        Merge character results from multiple segments
        
        Args:
            results: List of character detection results from segments
            full_text: Full text for counting appearances
            
        Returns:
            Merged character dictionary
        """
        character_map = {}
        
        for result in results:
            for char_data in result:
                name = char_data.get('name', '')
                if not name:
                    continue
                
                # Normalize name
                name_key = name.lower()
                
                if name_key in character_map:
                    # Update gender if unknown
                    if character_map[name_key].gender == 'unknown' and char_data.get('gender') != 'unknown':
                        character_map[name_key] = CharacterTraits(
                            name=character_map[name_key].name,
                            gender=char_data.get('gender', 'unknown').lower(),
                            demeanor=character_map[name_key].demeanor,
                            appearances=character_map[name_key].appearances
                        )
                    
                    # Update demeanor if more specific
                    if character_map[name_key].demeanor == 'neutral' and char_data.get('demeanor') != 'neutral':
                        character_map[name_key] = CharacterTraits(
                            name=character_map[name_key].name,
                            gender=character_map[name_key].gender,
                            demeanor=char_data.get('demeanor', 'neutral').lower(),
                            appearances=character_map[name_key].appearances
                        )
                else:
                    # Count actual appearances in full text
                    appearances = len(re.findall(r'\b' + re.escape(name) + r'\b', full_text))
                    
                    character_map[name_key] = CharacterTraits(
                        name=name,
                        gender=char_data.get('gender', 'unknown').lower(),
                        demeanor=char_data.get('demeanor', 'neutral').lower(),
                        appearances=appearances
                    )
        
        # Convert back to name-keyed dict using original names
        return {traits.name: traits for traits in character_map.values()}
    
    def _detect_characters_heuristic(self, text: str) -> Dict[str, CharacterTraits]:
        """
        Detect characters using heuristic methods (original implementation)
        
        Args:
            text (str): Input text
            
        Returns:
            Dictionary of character names to traits
        """
        # Find proper nouns (potential character names)
        # Look for capitalized words that appear multiple times
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_counts = defaultdict(int)
        
        for word in words:
            # Skip common words
            if word.lower() in ['the', 'a', 'an', 'i', 'chapter', 'part']:
                continue
            word_counts[word] += 1
        
        # Characters are names that appear multiple times
        characters = {}
        for name, count in word_counts.items():
            if count >= 3:  # Appears at least 3 times
                character = CharacterTraits(
                    name=name,
                    appearances=count
                )
                
                # Detect gender
                character.gender = self._detect_gender(text, name)
                
                # Detect demeanor (basic heuristic)
                character.demeanor = self._detect_demeanor(text, name)
                
                characters[name] = character
        
        self.characters = characters
        
        # Build pronoun map
        self._build_pronoun_map(text)
        
        return characters
    
    def _detect_gender(self, text: str, name: str) -> str:
        """
        Detect gender of a character from context
        
        Args:
            text (str): Full text
            name (str): Character name
            
        Returns:
            Gender: male, female, neutral, or unknown
        """
        # Find sentences containing the name
        sentences = re.split(r'[.!?]', text)
        relevant_sentences = [s.lower() for s in sentences if name.lower() in s.lower()]
        
        male_score = 0
        female_score = 0
        
        for sentence in relevant_sentences:
            for indicator in self.MALE_INDICATORS:
                if indicator in sentence:
                    male_score += 1
            for indicator in self.FEMALE_INDICATORS:
                if indicator in sentence:
                    female_score += 1
        
        if male_score > female_score * 1.5:
            return "male"
        elif female_score > male_score * 1.5:
            return "female"
        elif male_score > 0 or female_score > 0:
            return "neutral"
        else:
            return "unknown"
    
    def _detect_demeanor(self, text: str, name: str) -> str:
        """
        Detect general demeanor of a character
        
        Args:
            text (str): Full text
            name (str): Character name
            
        Returns:
            Demeanor description
        """
        # Find sentences with the character
        sentences = re.split(r'[.!?]', text)
        relevant_sentences = [s.lower() for s in sentences if name.lower() in s.lower()]
        
        # Count emotion keywords in character's context
        emotion_scores = defaultdict(int)
        for sentence in relevant_sentences:
            for emotion, keywords in self.EMOTION_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in sentence:
                        emotion_scores[emotion] += 1
        
        if not emotion_scores:
            return "neutral"
        
        # Return dominant emotion as demeanor
        dominant = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant[0]
    
    def _build_pronoun_map(self, text: str):
        """
        Build a map of pronouns to character names based on context
        
        Args:
            text (str): Full text
        """
        if not self.characters:
            return
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        # Track last mentioned character per gender
        last_male = None
        last_female = None
        last_neutral = None
        last_any = None
        
        for sentence in sentences:
            # Find characters mentioned in this sentence
            mentioned_chars = []
            for name in self.characters.keys():
                if re.search(r'\b' + re.escape(name) + r'\b', sentence, re.IGNORECASE):
                    mentioned_chars.append(name)
            
            # Update last mentioned tracking
            for name in mentioned_chars:
                gender = self.characters[name].gender
                last_any = name
                
                if gender == 'male':
                    last_male = name
                elif gender == 'female':
                    last_female = name
                elif gender == 'neutral':
                    last_neutral = name
            
            # Map pronouns in this sentence to last mentioned character of that gender
            sentence_lower = sentence.lower()
            
            # Male pronouns
            if any(pronoun in sentence_lower for pronoun in ['he ', ' him ', ' his ']):
                if last_male:
                    self.pronoun_map[sentence] = last_male
            
            # Female pronouns
            elif any(pronoun in sentence_lower for pronoun in ['she ', ' her ', ' hers ']):
                if last_female:
                    self.pronoun_map[sentence] = last_female
            
            # Neutral/first person (map to last mentioned if only one character)
            elif any(pronoun in sentence_lower for pronoun in [' i ', ' my ', ' me ']):
                if last_any and len(mentioned_chars) == 0:
                    self.pronoun_map[sentence] = last_any
    
    def resolve_pronoun_to_character(self, text: str, context: str = None) -> Optional[str]:
        """
        Resolve a pronoun in text to a character name
        
        Args:
            text (str): Text segment with pronoun
            context (str): Optional context (previous sentences)
            
        Returns:
            Character name or None
        """
        text_lower = text.lower()
        
        # Check direct pronoun map
        for sentence, char in self.pronoun_map.items():
            if text in sentence or sentence in text:
                return char
        
        # Analyze pronouns in text
        has_male = any(p in text_lower for p in ['he ', ' him ', ' his '])
        has_female = any(p in text_lower for p in ['she ', ' her ', ' hers '])
        has_first_person = any(p in text_lower for p in [' i ', ' my ', ' me ', ' myself '])
        
        # If context provided, look for last mentioned character
        if context:
            for name in self.characters.keys():
                if re.search(r'\b' + re.escape(name) + r'\b', context, re.IGNORECASE):
                    char_gender = self.characters[name].gender
                    
                    if has_male and char_gender == 'male':
                        return name
                    elif has_female and char_gender == 'female':
                        return name
                    elif has_first_person:
                        return name
        
        return None
    
    def analyze_emotion(self, text: str) -> EmotionalState:
        """
        Analyze emotional state of a text segment
        
        Args:
            text (str): Text segment
            
        Returns:
            EmotionalState object
        """
        text_lower = text.lower()
        
        # Count emotion keywords
        emotion_scores = defaultdict(float)
        total_keywords = 0
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                count = text_lower.count(keyword)
                emotion_scores[emotion] += count
                total_keywords += count
        
        # Normalize scores
        if total_keywords > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_keywords
        else:
            # No emotion keywords found, default to neutral
            emotion_scores['calm'] = 1.0
        
        # Find dominant emotion
        if emotion_scores:
            dominant = max(emotion_scores.items(), key=lambda x: x[1])
            dominant_emotion = dominant[0]
            intensity = min(1.0, dominant[1] * 2)  # Scale intensity
        else:
            dominant_emotion = 'calm'
            intensity = 0.5
        
        # Create emotion vector (8 emotions for IndexTTS2)
        # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        emotion_vector = {
            'happy': emotion_scores.get('happy', 0.0),
            'angry': emotion_scores.get('angry', 0.0),
            'sad': emotion_scores.get('sad', 0.0),
            'afraid': emotion_scores.get('afraid', 0.0),
            'disgusted': emotion_scores.get('disgusted', 0.0),
            'melancholic': emotion_scores.get('melancholic', 0.0),
            'surprised': emotion_scores.get('surprised', 0.0),
            'calm': emotion_scores.get('calm', 0.0),
        }
        
        return EmotionalState(
            dominant_emotion=dominant_emotion,
            intensity=intensity,
            emotions=emotion_vector
        )
    
    def create_character_segments(self, text: str, base_segments: List[str]) -> List[CharacterSegment]:
        """
        Create character-aware segments from base text segments
        
        Args:
            text (str): Full text
            base_segments (List[str]): Base text segments
            
        Returns:
            List of CharacterSegment objects
        """
        # First detect all characters
        if not self.characters:
            self.detect_characters(text)
        
        character_segments = []
        
        for i, segment_text in enumerate(base_segments):
            # Extract dialogue/thoughts
            dialogue_thoughts = self.extract_dialogue_and_thoughts(segment_text)
            
            if dialogue_thoughts:
                # Segment has dialogue/thoughts - create sub-segments
                for dt_text, speaker, is_dialogue, is_thought in dialogue_thoughts:
                    emotion = self.analyze_emotion(dt_text)
                    
                    char_seg = CharacterSegment(
                        segment_id=i,
                        text=dt_text,
                        character=speaker,
                        is_dialogue=is_dialogue,
                        is_thought=is_thought,
                        is_narration=False,
                        emotional_state=emotion
                    )
                    character_segments.append(char_seg)
            else:
                # Pure narration
                emotion = self.analyze_emotion(segment_text)
                
                char_seg = CharacterSegment(
                    segment_id=i,
                    text=segment_text,
                    character=None,
                    is_dialogue=False,
                    is_thought=False,
                    is_narration=True,
                    emotional_state=emotion
                )
                character_segments.append(char_seg)
        
        # Save debug file with annotated segments
        self._save_character_segments_debug(character_segments)
        
        return character_segments
    
    def _save_character_segments_debug(self, segments: List[CharacterSegment]):
        """
        Save annotated character segments to debug file
        
        Args:
            segments: List of CharacterSegment objects
        """
        if not self.work_dir:
            return
        
        debug_file = os.path.join(self.work_dir, "character_segments_debug.txt")
        
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("CHARACTER SEGMENTS DEBUG OUTPUT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Total Segments: {len(segments)}\n\n")
                
                # Count statistics
                dialogue_count = sum(1 for s in segments if s.is_dialogue)
                thought_count = sum(1 for s in segments if s.is_thought)
                narration_count = sum(1 for s in segments if s.is_narration)
                
                f.write(f"Dialogue Segments: {dialogue_count}\n")
                f.write(f"Thought Segments: {thought_count}\n")
                f.write(f"Narration Segments: {narration_count}\n\n")
                
                # Character statistics
                character_counts = defaultdict(int)
                for seg in segments:
                    if seg.character:
                        character_counts[seg.character] += 1
                
                if character_counts:
                    f.write("Character Appearances:\n")
                    for char, count in sorted(character_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  - {char}: {count} segments\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n\n")
                
                # Write each segment with annotations
                for i, seg in enumerate(segments):
                    f.write(f"SEGMENT {i+1:04d}\n")
                    f.write("-" * 80 + "\n")
                    
                    # Type annotation
                    if seg.is_dialogue:
                        seg_type = "DIALOGUE"
                    elif seg.is_thought:
                        seg_type = "THOUGHT"
                    elif seg.is_narration:
                        seg_type = "NARRATION"
                    else:
                        seg_type = "UNKNOWN"
                    
                    f.write(f"Type: {seg_type}\n")
                    
                    # Character annotation
                    if seg.character:
                        f.write(f"Character: {seg.character}\n")
                    else:
                        f.write(f"Character: [NARRATOR]\n")
                    
                    # Emotion annotation
                    f.write(f"Emotion: {seg.emotional_state.dominant_emotion} ")
                    f.write(f"(intensity: {seg.emotional_state.intensity:.2f})\n")
                    
                    # Emotion vector
                    if seg.emotional_state.emotions:
                        top_emotions = sorted(
                            seg.emotional_state.emotions.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:3]
                        if top_emotions and top_emotions[0][1] > 0:
                            emotion_str = ", ".join([f"{e}:{v:.2f}" for e, v in top_emotions if v > 0])
                            f.write(f"Emotion Vector: {emotion_str}\n")
                    
                    f.write("\n")
                    
                    # Text content with visual separator
                    f.write("TEXT:\n")
                    f.write("┌" + "─" * 78 + "┐\n")
                    
                    # Wrap text at 76 characters
                    text_lines = seg.text.split('\n')
                    for line in text_lines:
                        # Word wrap
                        words = line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line) + len(word) + 1 <= 76:
                                current_line += (" " if current_line else "") + word
                            else:
                                f.write(f"│ {current_line:<76} │\n")
                                current_line = word
                        if current_line:
                            f.write(f"│ {current_line:<76} │\n")
                    
                    f.write("└" + "─" * 78 + "┘\n")
                    f.write("\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF CHARACTER SEGMENTS DEBUG OUTPUT\n")
                f.write("=" * 80 + "\n")
            
            print(f"  Saved character segments debug file: {debug_file}")
            
        except Exception as e:
            print(f"  Warning: Failed to save debug file: {e}")
    
    def save_characters(self, filepath: str):
        """Save detected characters to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            data = {name: char.to_dict() for name, char in self.characters.items()}
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_characters(self, filepath: str):
        """Load characters from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.characters = {name: CharacterTraits.from_dict(char_data) 
                             for name, char_data in data.items()}
    
    def merge_characters(self, char1: str, char2: str):
        """
        Merge two characters (e.g., nicknames or OCR errors)
        
        Args:
            char1 (str): Primary character name
            char2 (str): Character to merge into char1
        """
        if char1 in self.characters and char2 in self.characters:
            # Merge appearances
            self.characters[char1].appearances += self.characters[char2].appearances
            self.characters[char1].dialogue_count += self.characters[char2].dialogue_count
            self.characters[char1].thought_count += self.characters[char2].thought_count
            
            # Remove char2
            del self.characters[char2]
            
            print(f"Merged '{char2}' into '{char1}'")
        else:
            print(f"Error: One or both characters not found")


if __name__ == "__main__":
    # Test the analyzer
    sample_text = """
    "Hello, how are you?" said John with a smile.
    Mary looked at him nervously. "I'm fine," she replied softly.
    John felt worried about her. (Why is she acting strange?) he thought.
    "Are you sure?" he asked gently.
    Sarah walked into the room, laughing. "What's going on here?"
    """
    
    analyzer = CharacterAnalyzer()
    
    # Detect characters
    characters = analyzer.detect_characters(sample_text)
    print("Detected Characters:")
    for name, traits in characters.items():
        print(f"  {name}: gender={traits.gender}, demeanor={traits.demeanor}, appearances={traits.appearances}")
    
    # Extract dialogue
    print("\nDialogue and Thoughts:")
    dialogue_thoughts = analyzer.extract_dialogue_and_thoughts(sample_text)
    for text, speaker, is_dialogue, is_thought in dialogue_thoughts:
        type_str = "dialogue" if is_dialogue else "thought"
        print(f"  [{type_str}] {speaker or 'Unknown'}: {text}")
    
    # Analyze emotion
    print("\nEmotional Analysis:")
    emotion = analyzer.analyze_emotion(sample_text)
    print(f"  Dominant: {emotion.dominant_emotion} (intensity: {emotion.intensity:.2f})")
    print(f"  Vector: {emotion.emotions}")
