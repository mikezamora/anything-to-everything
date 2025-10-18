"""
Interactive tool for reviewing and managing detected characters
Allows merging characters, editing traits, and configuring voices
"""
import json
from pathlib import Path
from typing import Dict, Optional
from character_analyzer import CharacterAnalyzer, CharacterTraits
from character_voice_config import CharacterVoiceMapping, VoiceConfig


class CharacterReviewTool:
    """Interactive tool for reviewing detected characters"""
    
    def __init__(self, analyzer: CharacterAnalyzer):
        """
        Initialize review tool
        
        Args:
            analyzer (CharacterAnalyzer): Analyzer with detected characters
        """
        self.analyzer = analyzer
        self.characters = analyzer.characters.copy()
    
    def display_characters(self):
        """Display all detected characters"""
        print("\n" + "="*70)
        print("DETECTED CHARACTERS")
        print("="*70)
        
        if not self.characters:
            print("No characters detected.")
            return
        
        # Sort by appearances
        sorted_chars = sorted(self.characters.items(), 
                            key=lambda x: x[1].appearances, 
                            reverse=True)
        
        for i, (name, traits) in enumerate(sorted_chars, 1):
            print(f"\n{i}. {name}")
            print(f"   Gender: {traits.gender}")
            print(f"   Demeanor: {traits.demeanor}")
            print(f"   Appearances: {traits.appearances}")
            print(f"   Dialogue: {traits.dialogue_count}, Thoughts: {traits.thought_count}")
    
    def merge_characters_interactive(self):
        """Interactive character merging"""
        self.display_characters()
        
        if len(self.characters) < 2:
            print("\nNeed at least 2 characters to merge.")
            return
        
        print("\n" + "="*70)
        print("MERGE CHARACTERS")
        print("="*70)
        print("Merge characters that refer to the same person")
        print("(e.g., 'John' and 'Johnny', or OCR errors)")
        
        while True:
            primary = input("\nEnter primary character name (or 'done' to finish): ").strip()
            if primary.lower() == 'done':
                break
            
            if primary not in self.characters:
                print(f"Character '{primary}' not found.")
                continue
            
            secondary = input(f"Enter character to merge into '{primary}': ").strip()
            if secondary not in self.characters:
                print(f"Character '{secondary}' not found.")
                continue
            
            if primary == secondary:
                print("Cannot merge a character with itself.")
                continue
            
            # Confirm merge
            confirm = input(f"Merge '{secondary}' into '{primary}'? (y/n): ").strip().lower()
            if confirm == 'y':
                self.analyzer.merge_characters(primary, secondary)
                self.characters = self.analyzer.characters.copy()
                print(f"Merged '{secondary}' into '{primary}'")
                self.display_characters()
    
    def edit_character_traits(self):
        """Interactive character trait editing"""
        self.display_characters()
        
        print("\n" + "="*70)
        print("EDIT CHARACTER TRAITS")
        print("="*70)
        
        while True:
            name = input("\nEnter character name to edit (or 'done' to finish): ").strip()
            if name.lower() == 'done':
                break
            
            if name not in self.characters:
                print(f"Character '{name}' not found.")
                continue
            
            char = self.characters[name]
            print(f"\nCurrent traits for {name}:")
            print(f"  Gender: {char.gender}")
            print(f"  Demeanor: {char.demeanor}")
            
            # Edit gender
            gender = input(f"New gender (male/female/neutral/unknown) [{char.gender}]: ").strip().lower()
            if gender in ['male', 'female', 'neutral', 'unknown']:
                char.gender = gender
            elif gender:
                print(f"Invalid gender. Keeping '{char.gender}'")
            
            # Edit demeanor
            demeanor = input(f"New demeanor [{char.demeanor}]: ").strip()
            if demeanor:
                char.demeanor = demeanor
            
            print(f"Updated traits for {name}")
    
    def remove_characters(self):
        """Remove false positive characters"""
        self.display_characters()
        
        print("\n" + "="*70)
        print("REMOVE CHARACTERS")
        print("="*70)
        print("Remove false positives (non-character names)")
        
        while True:
            name = input("\nEnter character name to remove (or 'done' to finish): ").strip()
            if name.lower() == 'done':
                break
            
            if name not in self.characters:
                print(f"Character '{name}' not found.")
                continue
            
            confirm = input(f"Remove '{name}'? (y/n): ").strip().lower()
            if confirm == 'y':
                del self.characters[name]
                del self.analyzer.characters[name]
                print(f"Removed '{name}'")
                self.display_characters()
    
    def save_reviewed_characters(self, output_path: str):
        """Save reviewed characters"""
        self.analyzer.characters = self.characters.copy()
        self.analyzer.save_characters(output_path)
        print(f"\nSaved reviewed characters to: {output_path}")
    
    def create_voice_config_template(self, output_path: str):
        """Create voice configuration template from reviewed characters"""
        character_names = list(self.characters.keys())
        CharacterVoiceMapping.create_template(character_names, output_path)
        print(f"Created voice config template: {output_path}")
    
    def run_interactive_review(self, output_dir: str = "./work"):
        """Run full interactive review process"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("CHARACTER REVIEW AND CONFIGURATION TOOL")
        print("="*70)
        
        while True:
            print("\n" + "-"*70)
            print("OPTIONS:")
            print("  1. Display characters")
            print("  2. Merge characters")
            print("  3. Edit character traits")
            print("  4. Remove characters")
            print("  5. Save and create voice config")
            print("  6. Exit without saving")
            print("-"*70)
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                self.display_characters()
            elif choice == '2':
                self.merge_characters_interactive()
            elif choice == '3':
                self.edit_character_traits()
            elif choice == '4':
                self.remove_characters()
            elif choice == '5':
                # Save reviewed characters
                char_file = output_dir / "reviewed_characters.json"
                self.save_reviewed_characters(str(char_file))
                
                # Create voice config template
                voice_config_file = output_dir / "character_voices.json"
                self.create_voice_config_template(str(voice_config_file))
                
                print("\n" + "="*70)
                print("Review complete!")
                print("="*70)
                print(f"\nFiles created:")
                print(f"  1. {char_file}")
                print(f"  2. {voice_config_file}")
                print(f"\nNext steps:")
                print(f"  1. Edit {voice_config_file} to map characters to voice files")
                print(f"  2. Run the audiobook converter with --character-config option")
                break
            elif choice == '6':
                print("Exiting without saving.")
                break
            else:
                print("Invalid choice. Please enter 1-6.")


def review_characters_from_file(characters_file: str, output_dir: str = "./work"):
    """
    Review characters from a saved JSON file
    
    Args:
        characters_file (str): Path to characters JSON file
        output_dir (str): Output directory for reviewed files
    """
    analyzer = CharacterAnalyzer()
    analyzer.load_characters(characters_file)
    
    tool = CharacterReviewTool(analyzer)
    tool.run_interactive_review(output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python character_review_tool.py <characters.json> [output_dir]")
        print("\nThis tool helps you review and configure detected characters.")
        sys.exit(1)
    
    characters_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./work"
    
    review_characters_from_file(characters_file, output_dir)
