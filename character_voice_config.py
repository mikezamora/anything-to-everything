"""
Character voice configuration for audiobook generation
Maps characters to voice files and emotion references
"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
import json
from pathlib import Path


@dataclass
class VoiceConfig:
    """Voice configuration for a character"""
    speaker_audio: str  # Path to speaker reference audio
    emotion_audio: Optional[str] = None  # Path to emotion reference audio
    emotion_alpha: float = 1.0  # Emotion blend strength
    use_emo_text: bool = False  # Auto-detect emotion from text
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class CharacterVoiceMapping:
    """Complete character to voice mapping configuration"""
    narrator_voice: VoiceConfig
    character_voices: Dict[str, VoiceConfig]
    default_voice: Optional[VoiceConfig] = None  # Fallback for unknown characters
    
    def to_dict(self):
        return {
            'narrator_voice': self.narrator_voice.to_dict(),
            'character_voices': {name: voice.to_dict() for name, voice in self.character_voices.items()},
            'default_voice': self.default_voice.to_dict() if self.default_voice else None
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            narrator_voice=VoiceConfig.from_dict(data['narrator_voice']),
            character_voices={name: VoiceConfig.from_dict(voice_data) 
                            for name, voice_data in data['character_voices'].items()},
            default_voice=VoiceConfig.from_dict(data['default_voice']) if data.get('default_voice') else None
        )
    
    def get_voice_for_character(self, character_name: Optional[str], is_narration: bool = False) -> VoiceConfig:
        """
        Get voice configuration for a character
        
        Args:
            character_name (str): Character name
            is_narration (bool): Whether this is narration
            
        Returns:
            VoiceConfig for the character
        """
        if is_narration or character_name is None:
            return self.narrator_voice
        
        if character_name in self.character_voices:
            return self.character_voices[character_name]
        
        # Return default or narrator voice as fallback
        return self.default_voice if self.default_voice else self.narrator_voice
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Voice configuration saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return cls.from_dict(data)
    
    @classmethod
    def create_template(cls, characters: List[str], output_path: str):
        """
        Create a template configuration file for detected characters
        
        Args:
            characters (List[str]): List of character names
            output_path (str): Path to save template
        """
        # Create default narrator voice
        narrator = VoiceConfig(
            speaker_audio="path/to/narrator_voice.wav",
            emotion_audio=None,
            emotion_alpha=0.7,
            use_emo_text=False
        )
        
        # Create template entries for each character
        char_voices = {}
        for char in characters:
            char_voices[char] = VoiceConfig(
                speaker_audio=f"path/to/{char.lower()}_voice.wav",
                emotion_audio=None,
                emotion_alpha=1.0,
                use_emo_text=False
            )
        
        # Create default voice
        default = VoiceConfig(
            speaker_audio="path/to/default_voice.wav",
            emotion_alpha=0.8,
            use_emo_text=False
        )
        
        mapping = cls(
            narrator_voice=narrator,
            character_voices=char_voices,
            default_voice=default
        )
        
        mapping.save(output_path)
        return mapping


@dataclass
class EmotionReference:
    """Emotion reference audio configuration"""
    emotion_name: str
    audio_path: str
    intensity: float = 1.0  # 0.0 to 1.0
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class EmotionLibrary:
    """Library of emotion reference audio files"""
    emotions: Dict[str, EmotionReference]
    
    def to_dict(self):
        return {name: emotion.to_dict() for name, emotion in self.emotions.items()}
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            emotions={name: EmotionReference.from_dict(emotion_data) 
                     for name, emotion_data in data.items()}
        )
    
    def get_emotion_audio(self, emotion_name: str) -> Optional[str]:
        """Get audio path for an emotion"""
        if emotion_name in self.emotions:
            return self.emotions[emotion_name].audio_path
        return None
    
    def save(self, filepath: str):
        """Save emotion library to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Emotion library saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load emotion library from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return cls.from_dict(data)
    
    @classmethod
    def create_template(cls, output_path: str):
        """Create template emotion library"""
        emotions = {
            'happy': EmotionReference('happy', 'emotions/happy.wav', 1.0),
            'sad': EmotionReference('sad', 'emotions/sad.wav', 1.0),
            'angry': EmotionReference('angry', 'emotions/angry.wav', 1.0),
            'afraid': EmotionReference('afraid', 'emotions/afraid.wav', 1.0),
            'surprised': EmotionReference('surprised', 'emotions/surprised.wav', 1.0),
            'disgusted': EmotionReference('disgusted', 'emotions/disgusted.wav', 1.0),
            'calm': EmotionReference('calm', 'emotions/calm.wav', 0.8),
            'melancholic': EmotionReference('melancholic', 'emotions/melancholic.wav', 0.9),
        }
        
        library = cls(emotions=emotions)
        library.save(output_path)
        return library


if __name__ == "__main__":
    # Create example configurations
    import sys
    
    # Create character voice mapping template
    print("Creating character voice mapping template...")
    characters = ['John', 'Mary', 'Sarah']
    mapping = CharacterVoiceMapping.create_template(characters, 'character_voices_template.json')
    
    print("\nCreating emotion library template...")
    library = EmotionLibrary.create_template('emotion_library_template.json')
    
    print("\n" + "="*70)
    print("Template files created!")
    print("="*70)
    print("\nNext steps:")
    print("1. Edit 'character_voices_template.json' with your voice file paths")
    print("2. Edit 'emotion_library_template.json' with your emotion reference paths")
    print("3. Use these files with the audiobook converter")
