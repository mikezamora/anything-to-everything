"""
Gradio Web UI for EPUB to Audiobook Converter with Job Queue
Provides interface for job creation, monitoring, character management, and file uploads
"""
import os
import sys
import json
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from job_processor import JobQueue, JobDefinition, JobStatus
from character_analyzer import CharacterAnalyzer, CharacterTraits
from character_voice_config import CharacterVoiceMapping, EmotionLibrary
from text_extractor import TextExtractor

# Create necessary directories
os.makedirs("uploads/text", exist_ok=True)  # Combined directory for EPUB and PDF
os.makedirs("uploads/voice", exist_ok=True)
os.makedirs("uploads/emotion", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("jobs", exist_ok=True)

# Global job queue
job_queue = JobQueue(jobs_dir="./jobs")

# Global log queue for terminal output
log_queue = queue.Queue()
terminal_logs = []

# Global state for character detection
current_characters = {}
current_epub_text = ""
current_analyzer = None


def get_uploaded_files(file_type: str) -> List[str]:
    """Get list of uploaded files of given type"""
    upload_dir = f"uploads/{file_type}"
    if not os.path.exists(upload_dir):
        return []
    
    files = []
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            files.append(file_path)
    
    return sorted(files)


def get_uploaded_text_files() -> List[str]:
    """Get list of uploaded text files (EPUB and PDF)"""
    return get_uploaded_files("text")


def get_uploaded_voice_files() -> List[str]:
    """Get list of uploaded voice files"""  
    return get_uploaded_files("voice")


def get_uploaded_emotion_files() -> List[str]:
    """Get list of uploaded emotion files"""
    return get_uploaded_files("emotion")


def log_message(message: str):
    """Add message to terminal log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    terminal_logs.append(log_entry)
    log_queue.put(log_entry)
    # Keep only last 1000 log entries
    if len(terminal_logs) > 1000:
        terminal_logs.pop(0)


def get_terminal_output():
    """Get current terminal output"""
    return "\n".join(terminal_logs)


def upload_text_file(file) -> Tuple[str, str]:
    """Handle text file upload (EPUB or PDF)"""
    if file is None:
        return "", "No file uploaded"
    
    try:
        # Validate file format
        filename = os.path.basename(file.name)
        if not TextExtractor.is_supported_file(filename):
            supported = ', '.join(TextExtractor.get_supported_extensions())
            return "", f"Unsupported file format. Supported: {supported}"
        
        # Copy to uploads directory
        dest_path = os.path.join("uploads/text", filename)
        
        # Read and write file
        with open(file.name, 'rb') as src:
            with open(dest_path, 'wb') as dst:
                dst.write(src.read())
        
        log_message(f"✓ Uploaded text file: {filename}")
        
        # Extract metadata
        try:
            extractor = TextExtractor.create_extractor(dest_path)
            metadata = extractor.get_metadata()
            info = f"Title: {metadata.get('title', 'Unknown')}\n"
            info += f"Author: {metadata.get('author', 'Unknown')}\n"
            info += f"Language: {metadata.get('language', 'Unknown')}"
            return dest_path, info
        except Exception as e:
            log_message(f"⚠ Could not extract metadata: {e}")
            return dest_path, "Text file uploaded successfully"
            
    except Exception as e:
        log_message(f"✗ Failed to upload text file: {e}")
        return "", f"Error: {str(e)}"


def upload_voice_file(file) -> Tuple[str, str]:
    """Handle voice reference file upload"""
    if file is None:
        return "", "No file uploaded"
    
    try:
        filename = os.path.basename(file.name)
        dest_path = os.path.join("uploads/voice", filename)
        
        with open(file.name, 'rb') as src:
            with open(dest_path, 'wb') as dst:
                dst.write(src.read())
        
        log_message(f"✓ Uploaded voice reference: {filename}")
        return dest_path, f"Voice file uploaded: {filename}"
    except Exception as e:
        log_message(f"✗ Failed to upload voice file: {e}")
        return "", f"Error: {str(e)}"


def upload_emotion_files(files) -> Tuple[str, str]:
    """Handle emotion reference files upload"""
    if not files:
        return "", "No files uploaded"
    
    try:
        uploaded = []
        for file in files:
            filename = os.path.basename(file.name)
            dest_path = os.path.join("uploads/emotion", filename)
            
            with open(file.name, 'rb') as src:
                with open(dest_path, 'wb') as dst:
                    dst.write(src.read())
            
            uploaded.append(filename)
        
        log_message(f"✓ Uploaded {len(uploaded)} emotion files")
        emotion_dir = os.path.abspath("uploads/emotion")
        return emotion_dir, f"Uploaded {len(uploaded)} files:\n" + "\n".join(uploaded)
    except Exception as e:
        log_message(f"✗ Failed to upload emotion files: {e}")
        return "", f"Error: {str(e)}"


def select_existing_text(selected_text: str) -> Tuple[str, str]:
    """Handle selection of existing text file"""
    if not selected_text:
        return "", ""
    
    try:
        # Get text file metadata
        extractor = TextExtractor.create_extractor(selected_text)
        metadata = extractor.get_metadata()
        
        info_lines = []
        for key, value in metadata.items():
            if value:
                info_lines.append(f"{key.title()}: {value}")
        
        text_info = "\n".join(info_lines)
        log_message(f"✓ Selected existing text file: {os.path.basename(selected_text)}")
        
        return selected_text, text_info
    except Exception as e:
        log_message(f"✗ Failed to read text file metadata: {e}")
        return selected_text, f"Selected: {os.path.basename(selected_text)} (metadata unavailable)"


def select_existing_voice(selected_voice: str) -> Tuple[str, str]:
    """Handle selection of existing voice file"""
    if not selected_voice:
        return "", ""
    
    log_message(f"✓ Selected existing voice: {os.path.basename(selected_voice)}")
    return selected_voice, f"Selected: {os.path.basename(selected_voice)}"


def select_existing_emotion(selected_emotion: str) -> str:
    """Handle selection of existing emotion audio file"""
    if not selected_emotion:
        return ""
    
    log_message(f"✓ Selected existing emotion audio: {os.path.basename(selected_emotion)}")
    return selected_emotion


def refresh_file_lists() -> Tuple[List[str], List[str]]:
    """Refresh the dropdown lists with current uploaded files"""
    return get_uploaded_epub_files(), get_uploaded_voice_files()


def create_single_job(
    source_text_file: str,
    voice_path: str,
    output_name: str,
    format_choice: str,
    priority: int,
    detect_chars: bool,
    ollama_char: bool,
    character_mode: bool,
    keep_segments: bool,
    use_ollama: bool,
    ollama_model: str,
    ollama_url: str,
    segment_words: int,
    max_words: int,
    min_words: int,
    strip_unknown_tokens: bool,
    emo_audio: str,
    character_config: str,
    emotion_library: str,
) -> Tuple[str, str]:
    """Create a single job"""
    try:
        # Validate inputs
        if not source_text_file or not os.path.exists(source_text_file):
            return "", "Error: Please upload a source text file first"
        
        # Voice reference is optional in character mode (can use character-specific voices)
        if not character_mode:
            if not voice_path or not os.path.exists(voice_path):
                return "", "Error: Please upload a voice reference file first (or enable Character Mode)"
        
        # If character mode but no voice path, check if character config exists
        if character_mode and (not voice_path or not os.path.exists(voice_path)):
            if not character_config or not os.path.exists(character_config):
                return "", "Error: Character Mode requires either a default voice file OR a character config with voice mappings"
            # In character mode with valid config, voice is optional
            voice_path = None
        
        if not output_name:
            output_name = f"audiobook_{int(time.time())}.{format_choice}"
        elif not output_name.endswith(f".{format_choice}"):
            output_name = f"{output_name}.{format_choice}"
        
        output_path = os.path.join("outputs", output_name)
        
        # Create job definition
        job_def = JobDefinition(
            job_id="",  # Auto-generated
            source_text_file=source_text_file,
            output_path=output_path,
            voice_ref_path=voice_path if voice_path else None,
            format=format_choice,
            detect_characters=detect_chars,
            ollama_character_detection=ollama_char,
            character_mode=character_mode,
            keep_segments=keep_segments,
            use_ollama=use_ollama,
            ollama_model=ollama_model if use_ollama or ollama_char else None,
            ollama_url=ollama_url,
            segment_words=segment_words,
            strip_unknown_tokens=strip_unknown_tokens,
            character_config=character_config if character_config and os.path.exists(character_config) else None,
            emotion_library=emotion_library if emotion_library and os.path.exists(emotion_library) else None,
            emo_audio_prompt=emo_audio if emo_audio and os.path.exists(emo_audio) else None,
            priority=priority
        )
        
        # Create job
        job_id = job_queue.create_job(job_def)
        log_message(f"✓ Created job {job_id} (priority {priority})")
        
        return job_id, f"Job created successfully!\nJob ID: {job_id}\nPriority: {priority}"
        
    except Exception as e:
        log_message(f"✗ Failed to create job: {e}")
        return "", f"Error creating job: {str(e)}"


def get_job_list(status_filter: str = "all") -> pd.DataFrame:
    """Get list of jobs as DataFrame"""
    try:
        if status_filter == "all":
            status = None
        else:
            status = status_filter
        
        job_ids = job_queue.list_jobs(status=status)
        
        if not job_ids:
            return pd.DataFrame(columns=["Job ID", "Status", "Priority", "Source", "Output"])
        
        data = []
        for job_id in job_ids:
            # Try to load job definition
            for stat in ["pending", "running", "completed", "failed"]:
                job_file = Path(f"jobs/{stat}/{job_id}/job_definition.json")
                if job_file.exists():
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                    
                    data.append({
                        "Job ID": job_id,
                        "Status": stat,
                        "Priority": job_data.get("priority", 0),
                        "Source": os.path.basename(job_data.get("source_text_file", "")),
                        "Output": os.path.basename(job_data.get("output_path", "")),
                    })
                    break
        
        return pd.DataFrame(data)
        
    except Exception as e:
        log_message(f"⚠ Error getting job list: {e}")
        return pd.DataFrame(columns=["Job ID", "Status", "Priority", "Source", "Output"])


def get_job_details(job_id: str) -> str:
    """Get detailed information about a job"""
    try:
        if not job_id or len(job_id) < 8:
            return "Enter a Job ID to view details"
        
        # Find full job ID
        all_jobs = []
        for status in ["pending", "running", "completed", "failed"]:
            all_jobs.extend(job_queue.list_jobs(status))
        
        full_job_id = None
        for jid in all_jobs:
            if jid.startswith(job_id) or job_id in jid:
                full_job_id = jid
                break
        
        if not full_job_id:
            return f"Job ID '{job_id}' not found"
        
        # Get job status
        result = job_queue.get_job_status(full_job_id)
        
        if not result:
            return f"Job '{job_id}' not found"
        
        # Load job definition
        for stat in ["pending", "running", "completed", "failed"]:
            job_file = Path(f"jobs/{stat}/{full_job_id}/job_definition.json")
            if job_file.exists():
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                break
        else:
            job_data = {}
        
        details = f"=== Job Details ===\n\n"
        details += f"Job ID: {full_job_id}\n"
        details += f"Status: {result.status.value}\n"
        details += f"Priority: {job_data.get('priority', 0)}\n\n"
        
        details += f"EPUB: {job_data.get('epub_path', 'N/A')}\n"
        details += f"Voice: {job_data.get('voice_ref_path', 'N/A')}\n"
        details += f"Output: {job_data.get('output_path', 'N/A')}\n"
        details += f"Format: {job_data.get('format', 'wav')}\n\n"
        
        details += f"Character Detection: {job_data.get('detect_characters', False)}\n"
        details += f"Ollama Character Detection: {job_data.get('ollama_character_detection', False)}\n"
        details += f"Character Mode: {job_data.get('character_mode', False)}\n"
        details += f"Keep Segments: {job_data.get('keep_segments', False)}\n"
        details += f"Use Ollama: {job_data.get('use_ollama', False)}\n\n"
        
        if job_data.get('character_config'):
            details += f"Character Config: {job_data.get('character_config')}\n"
        if job_data.get('emotion_library'):
            details += f"Emotion Library: {job_data.get('emotion_library')}\n"
        
        details += "\n"
        
        if result.start_time:
            details += f"Started: {result.start_time}\n"
        if result.end_time:
            details += f"Ended: {result.end_time}\n"
        if result.exit_code is not None:
            details += f"Exit Code: {result.exit_code}\n"
        if result.error_message:
            details += f"\nError: {result.error_message}\n"
        
        return details
        
    except Exception as e:
        return f"Error getting job details: {str(e)}"


def cancel_job(job_id: str) -> str:
    """Cancel a pending job"""
    try:
        if not job_id:
            return "Enter a Job ID to cancel"
        
        # Find full job ID
        pending_jobs = job_queue.list_jobs("pending")
        full_job_id = None
        for jid in pending_jobs:
            if jid.startswith(job_id) or job_id in jid:
                full_job_id = jid
                break
        
        if not full_job_id:
            return f"Pending job '{job_id}' not found"
        
        if job_queue.cancel_job(full_job_id):
            log_message(f"✓ Cancelled job {full_job_id}")
            return f"Job {full_job_id} cancelled successfully"
        else:
            return f"Failed to cancel job {full_job_id}"
            
    except Exception as e:
        return f"Error: {str(e)}"


def start_processing(max_jobs: int, stop_on_error: bool) -> str:
    """Start processing the job queue"""
    try:
        log_message(f"Starting job processor (max_jobs={max_jobs}, stop_on_error={stop_on_error})")
        
        # Run in background thread
        def process_queue():
            try:
                results = job_queue.process_queue(
                    max_jobs=max_jobs if max_jobs > 0 else None,
                    stop_on_error=stop_on_error
                )
                
                completed = sum(1 for r in results if r.status == JobStatus.COMPLETED)
                failed = sum(1 for r in results if r.status == JobStatus.FAILED)
                
                log_message(f"✓ Queue processing complete: {completed} completed, {failed} failed")
            except Exception as e:
                log_message(f"✗ Queue processing error: {e}")
        
        thread = threading.Thread(target=process_queue, daemon=True)
        thread.start()
        
        return "Job processor started. Check terminal for progress."
        
    except Exception as e:
        log_message(f"✗ Failed to start processor: {e}")
        return f"Error: {str(e)}"


def start_single_job(job_id: str) -> str:
    """Start processing a specific job"""
    try:
        if not job_id:
            return "Enter a Job ID to start"
        
        # Find full job ID
        pending_jobs = job_queue.list_jobs("pending")
        full_job_id = None
        for jid in pending_jobs:
            if jid.startswith(job_id) or job_id in jid:
                full_job_id = jid
                break
        
        if not full_job_id:
            return f"Pending job '{job_id}' not found"
        
        log_message(f"Starting job {full_job_id}")
        
        # Run in background thread
        def process_job():
            try:
                result = job_queue.process_single_job(full_job_id)
                if result:
                    if result.status == JobStatus.COMPLETED:
                        log_message(f"✓ Job {full_job_id} completed successfully")
                    else:
                        log_message(f"✗ Job {full_job_id} failed")
                else:
                    log_message(f"✗ Job {full_job_id} not found")
            except Exception as e:
                log_message(f"✗ Job processing error: {e}")
        
        thread = threading.Thread(target=process_job, daemon=True)
        thread.start()
        
        return f"Started processing job {full_job_id}\nCheck terminal for progress."
        
    except Exception as e:
        log_message(f"✗ Failed to start job: {e}")
        return f"Error: {str(e)}"


def detect_characters_from_text(
    text_path: str,
    use_ollama: bool,
    ollama_model: str,
    ollama_url: str
) -> Tuple[str, pd.DataFrame]:
    """Detect characters from uploaded text file (EPUB/PDF)"""
    global current_characters, current_epub_text, current_analyzer
    
    try:
        if not text_path or not os.path.exists(text_path):
            return "Please upload a text file first", pd.DataFrame()
        
        log_message(f"Detecting characters from {os.path.basename(text_path)}...")
        
        # Extract text
        extractor = TextExtractor.create_extractor(text_path)
        text = extractor.extract_text()
        current_epub_text = text
        
        # Create analyzer
        current_analyzer = CharacterAnalyzer(
            use_ollama=use_ollama,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            work_dir="./work/character_detection"
        )
        
        # Detect characters
        characters = current_analyzer.detect_characters(text)
        current_characters = characters
        
        # Create DataFrame
        data = []
        for name, traits in sorted(characters.items(), key=lambda x: x[1].appearances, reverse=True):
            data.append({
                "Name": name,
                "Gender": traits.gender,
                "Demeanor": traits.demeanor,
                "Appearances": traits.appearances,
                "Dialogue": traits.dialogue_count,
                "Thoughts": traits.thought_count
            })
        
        df = pd.DataFrame(data)
        
        log_message(f"✓ Detected {len(characters)} characters")
        
        message = f"✓ Detected {len(characters)} characters\n\n"
        message += "Characters saved to work/character_detection/detected_characters.json\n"
        message += "You can now create voice configuration below."
        
        return message, df
        
    except Exception as e:
        log_message(f"✗ Character detection failed: {e}")
        return f"Error: {str(e)}", pd.DataFrame()


def save_character_voice_config(config_json: str) -> Tuple[str, str]:
    """Save character voice configuration and return path"""
    try:
        if not config_json:
            return "Error: No configuration provided", ""
        
        # Parse JSON
        config_data = json.loads(config_json)
        
        # Save to file
        config_path = "work/character_voices.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        log_message(f"✓ Saved character voice configuration to {config_path}")
        return f"Configuration saved to {config_path}", config_path
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {str(e)}", ""
    except Exception as e:
        return f"Error: {str(e)}", ""


def save_emotion_library(library_json: str) -> Tuple[str, str]:
    """Save emotion library configuration and return path"""
    try:
        if not library_json:
            return "Error: No configuration provided", ""
        
        # Parse JSON
        library_data = json.loads(library_json)
        
        # Save to file
        library_path = "work/emotion_library.json"
        os.makedirs(os.path.dirname(library_path), exist_ok=True)
        
        with open(library_path, 'w', encoding='utf-8') as f:
            json.dump(library_data, f, indent=2)
        
        log_message(f"✓ Saved emotion library to {library_path}")
        return f"Emotion library saved to {library_path}", library_path
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {str(e)}", ""
    except Exception as e:
        return f"Error: {str(e)}", ""


def create_voice_config_template() -> str:
    """Create voice configuration template from detected characters"""
    global current_characters
    
    try:
        if not current_characters:
            return "Please detect characters first"
        
        character_names = list(current_characters.keys())
        config_path = "work/character_voices_template.json"
        
        CharacterVoiceMapping.create_template(character_names, config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_json = f.read()
        
        log_message(f"✓ Created voice configuration template")
        return config_json
        
    except Exception as e:
        return f"Error: {str(e)}"


def create_emotion_library_template() -> str:
    """Create emotion library template"""
    try:
        template_path = "work/emotion_library_template.json"
        EmotionLibrary.create_template(template_path)
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_json = f.read()
        
        log_message(f"✓ Created emotion library template")
        return template_json
        
    except Exception as e:
        return f"Error: {str(e)}"


def refresh_terminal() -> str:
    """Refresh terminal output"""
    return get_terminal_output()


# Create Gradio interface
with gr.Blocks(title="Anything to Everything", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Anything to Everything")
    gr.Markdown("Convert EPUB files to audiobooks with character-aware multi-voice support")
    
    with gr.Tabs():
        # Tab 1: Job Creation
        with gr.Tab("📝 Create Job"):
            gr.Markdown("### Upload Files and Configure Job")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Required Files")
                    
                    # Text file selection (EPUB/PDF)
                    gr.Markdown("**Text File (EPUB/PDF):**")
                    with gr.Row():
                        text_upload = gr.File(label="Upload New Text File", file_types=[".epub", ".pdf"])
                        text_dropdown = gr.Dropdown(
                            label="Or Select Existing Text File", 
                            choices=get_uploaded_text_files(),
                            interactive=True
                        )
                    text_path_display = gr.Textbox(label="Selected Text File Path", interactive=False)
                    text_info = gr.Textbox(label="Text File Info", lines=3, interactive=False)
                    
                    # Voice file selection
                    gr.Markdown("**Voice Reference (optional in Character Mode):**")
                    with gr.Row():
                        voice_upload = gr.File(
                            label="Upload New Voice File", 
                            file_types=[".wav", ".mp3", ".flac"]
                        )
                        voice_dropdown = gr.Dropdown(
                            label="Or Select Existing Voice",
                            choices=get_uploaded_voice_files(),
                            interactive=True
                        )
                    voice_path_display = gr.Textbox(label="Selected Voice Path", interactive=False)
                    voice_info = gr.Textbox(label="Voice Info", interactive=False)
                    gr.Markdown("💡 *In Character Mode, voice reference is optional if character config has all voices*")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### Job Settings")
                    output_name = gr.Textbox(label="Output Filename", placeholder="audiobook.m4b")
                    format_choice = gr.Radio(["wav", "m4b"], label="Output Format", value="m4b")
                    priority = gr.Slider(0, 100, value=10, step=1, label="Priority (higher = processed first)")
                    
                    gr.Markdown("#### Processing Options")
                    character_mode = gr.Checkbox(label="Enable Character Mode (Multi-Voice)", value=False)
                    use_ollama = gr.Checkbox(label="Use Ollama for Text Processing", value=False)
                    keep_segments = gr.Checkbox(label="Keep Audio Segments", value=False)
                    
                    gr.Markdown("#### Character & Emotion Configuration")
                    character_config = gr.Textbox(
                        label="Character Config Path (auto-populated from Character tab)", 
                        value="work/character_voices.json",
                        placeholder="work/character_voices.json"
                    )
                    emotion_library = gr.Textbox(
                        label="Emotion Library Path (auto-populated from Character tab)", 
                        value="work/emotion_library.json",
                        placeholder="work/emotion_library.json"
                    )
                    # Emotion audio selection
                    gr.Markdown("**Emotion Audio (optional):**")
                    with gr.Row():
                        emo_audio = gr.Textbox(label="Custom Path", placeholder="path/to/emotion.wav", scale=2)
                        emo_dropdown = gr.Dropdown(
                            label="Or Select Existing",
                            choices=get_uploaded_emotion_files(),
                            interactive=True,
                            scale=1
                        )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        detect_chars = gr.Checkbox(label="Detect Characters (creates template and exits)", value=False)
                        ollama_char = gr.Checkbox(label="Use Ollama for Character Detection", value=False)
                        
                        ollama_model = gr.Textbox(label="Ollama Model", value="aratan/DeepSeek-R1-32B-Uncensored:latest")
                        ollama_url = gr.Textbox(label="Ollama URL", value="http://host.docker.internal:11434")
                        
                        segment_words = gr.Slider(100, 1000, value=500, step=50, label="Words per Segment")
                        max_words = gr.Slider(200, 1200, value=600, step=50, label="Max Words per Segment")
                        min_words = gr.Slider(50, 500, value=100, step=50, label="Min Words per Segment")
                        strip_unknown_tokens = gr.Checkbox(label="Strip Unknown Tokens (recommended)", value=True, info="Remove tokens like === that cause TTS encoding issues")
            
            with gr.Row():
                create_job_btn = gr.Button("Create Job", variant="primary", size="lg")
                job_result = gr.Textbox(label="Result", interactive=False)
                job_id_output = gr.Textbox(label="Job ID", interactive=False, visible=False)
            
            # Wire up file uploads and refreshes
            text_upload.upload(
                upload_text_file,
                inputs=[text_upload],
                outputs=[text_path_display, text_info]
            ).then(
                # Refresh dropdown after upload
                lambda: gr.Dropdown(choices=get_uploaded_text_files()),
                inputs=[],
                outputs=[text_dropdown]
            )
            
            voice_upload.upload(
                upload_voice_file,
                inputs=[voice_upload],
                outputs=[voice_path_display, voice_info]
            ).then(
                # Refresh dropdown after upload
                lambda: gr.Dropdown(choices=get_uploaded_voice_files()),
                inputs=[],
                outputs=[voice_dropdown]
            )
            
            # Wire up dropdown selections
            text_dropdown.change(
                select_existing_text,
                inputs=[text_dropdown],
                outputs=[text_path_display, text_info]
            )
            
            voice_dropdown.change(
                select_existing_voice,
                inputs=[voice_dropdown],  
                outputs=[voice_path_display, voice_info]
            )
            
            # Wire up emotion dropdown
            emo_dropdown.change(
                select_existing_emotion,
                inputs=[emo_dropdown],
                outputs=[emo_audio]
            )
            
            # Wire up job creation
            create_job_btn.click(
                create_single_job,
                inputs=[
                    text_path_display, voice_path_display, output_name, format_choice, priority,
                    detect_chars, ollama_char, character_mode, keep_segments, use_ollama,
                    ollama_model, ollama_url, segment_words, max_words, min_words, strip_unknown_tokens,
                    emo_audio, character_config, emotion_library
                ],
                outputs=[job_id_output, job_result]
            )
        
        # Tab 2: Job Monitor
        with gr.Tab("📊 Job Monitor"):
            gr.Markdown("### Monitor Job Queue and Status")
            
            with gr.Row():
                status_filter = gr.Radio(
                    ["all", "pending", "running", "completed", "failed"],
                    label="Filter by Status",
                    value="all"
                )
                refresh_jobs_btn = gr.Button("🔄 Refresh", size="sm")
            
            jobs_table = gr.Dataframe(
                headers=["Job ID", "Status", "Priority", "EPUB", "Output"],
                label="Job Queue"
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    job_id_input = gr.Textbox(label="Job ID (or prefix)", placeholder="Enter job ID...")
                    with gr.Row():
                        get_details_btn = gr.Button("Get Details", size="sm")
                        start_single_job_btn = gr.Button("▶️ Start Job", variant="primary", size="sm")
                        cancel_job_btn = gr.Button("Cancel Job", variant="stop", size="sm")
                
                with gr.Column(scale=2):
                    job_details_output = gr.Textbox(label="Job Details", lines=20, interactive=False)
            
            with gr.Row():
                max_jobs_input = gr.Number(label="Max Jobs to Process (0 = unlimited)", value=0, precision=0)
                stop_on_error_check = gr.Checkbox(label="Stop on First Error", value=False)
                start_processing_btn = gr.Button("▶️ Start Processing Queue", variant="primary", size="lg")
            
            processing_result = gr.Textbox(label="Processing Status", interactive=False)
            
            # Wire up job monitor
            refresh_jobs_btn.click(
                get_job_list,
                inputs=[status_filter],
                outputs=[jobs_table]
            )
            
            status_filter.change(
                get_job_list,
                inputs=[status_filter],
                outputs=[jobs_table]
            )
            
            get_details_btn.click(
                get_job_details,
                inputs=[job_id_input],
                outputs=[job_details_output]
            )
            
            start_single_job_btn.click(
                start_single_job,
                inputs=[job_id_input],
                outputs=[job_details_output]
            )
            
            cancel_job_btn.click(
                cancel_job,
                inputs=[job_id_input],
                outputs=[job_details_output]
            )
            
            start_processing_btn.click(
                start_processing,
                inputs=[max_jobs_input, stop_on_error_check],
                outputs=[processing_result]
            )
            
            # Auto-refresh every 5 seconds using timer
            job_refresh_timer = gr.Timer(5)
            job_refresh_timer.tick(
                get_job_list,
                inputs=[status_filter],
                outputs=[jobs_table]
            )
        
        # Tab 3: Character Management
        with gr.Tab("🎭 Character Management"):
            gr.Markdown("### Detect and Configure Characters")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Detect Characters")
                    char_text_path = gr.Textbox(label="Text File Path", placeholder="Upload text file (EPUB/PDF) in Create Job tab first")
                    char_use_ollama = gr.Checkbox(label="Use Ollama for Detection", value=True)
                    char_ollama_model = gr.Textbox(label="Ollama Model", value="aratan/DeepSeek-R1-32B-Uncensored:latest")
                    char_ollama_url = gr.Textbox(label="Ollama URL", value="http://host.docker.internal:11434")
                    detect_chars_btn = gr.Button("Detect Characters", variant="primary")
                    
                    char_detection_result = gr.Textbox(label="Detection Result", lines=5, interactive=False)
                
                with gr.Column():
                    gr.Markdown("#### Detected Characters")
                    characters_table = gr.Dataframe(
                        headers=["Name", "Gender", "Demeanor", "Appearances", "Dialogue", "Thoughts"],
                        label="Characters"
                    )
            
            gr.Markdown("#### Step 2: Create Voice Configuration")
            
            with gr.Row():
                create_voice_template_btn = gr.Button("Create Voice Config Template")
                create_emotion_template_btn = gr.Button("Create Emotion Library Template")
            
            with gr.Row():
                with gr.Column():
                    voice_config_editor = gr.Code(
                        label="Character Voice Configuration (JSON)",
                        language="json",
                        lines=20
                    )
                    save_voice_config_btn = gr.Button("Save Voice Configuration", variant="primary")
                    voice_config_result = gr.Textbox(label="Save Result", interactive=False)
                
                with gr.Column():
                    emotion_library_editor = gr.Code(
                        label="Emotion Library (JSON)",
                        language="json",
                        lines=20
                    )
                    save_emotion_library_btn = gr.Button("Save Emotion Library", variant="primary")
                    emotion_library_result = gr.Textbox(label="Save Result", interactive=False)
            
            gr.Markdown("#### Step 3: Upload Emotion Reference Files")
            emotion_files_upload = gr.File(
                label="Upload Emotion Audio Files",
                file_count="multiple",
                file_types=[".wav", ".mp3", ".flac"]
            )
            emotion_upload_result = gr.Textbox(label="Upload Result", interactive=False)
            emotion_dir_display = gr.Textbox(label="Emotion Files Directory", interactive=False)
            
            # Wire up character management
            text_path_display.change(
                lambda x: x,
                inputs=[text_path_display],
                outputs=[char_text_path]
            )
            
            detect_chars_btn.click(
                detect_characters_from_text,
                inputs=[char_text_path, char_use_ollama, char_ollama_model, char_ollama_url],
                outputs=[char_detection_result, characters_table]
            )
            
            create_voice_template_btn.click(
                create_voice_config_template,
                outputs=[voice_config_editor]
            )
            
            create_emotion_template_btn.click(
                create_emotion_library_template,
                outputs=[emotion_library_editor]
            )
            
            save_voice_config_btn.click(
                save_character_voice_config,
                inputs=[voice_config_editor],
                outputs=[voice_config_result, character_config]
            )
            
            save_emotion_library_btn.click(
                save_emotion_library,
                inputs=[emotion_library_editor],
                outputs=[emotion_library_result, emotion_library]
            )
            
            emotion_files_upload.upload(
                upload_emotion_files,
                inputs=[emotion_files_upload],
                outputs=[emotion_dir_display, emotion_upload_result]
            )
        
        # Tab 4: Terminal
        with gr.Tab("💻 Terminal"):
            gr.Markdown("### System Logs and Output")
            
            terminal_output = gr.Textbox(
                label="Terminal Output",
                lines=30,
                max_lines=30,
                interactive=False,
                show_copy_button=True
            )
            
            with gr.Row():
                refresh_terminal_btn = gr.Button("🔄 Refresh Terminal", size="sm")
                clear_terminal_btn = gr.Button("🗑️ Clear Terminal", size="sm")
            
            # Wire up terminal
            refresh_terminal_btn.click(
                refresh_terminal,
                outputs=[terminal_output]
            )
            
            clear_terminal_btn.click(
                lambda: (terminal_logs.clear(), ""),
                outputs=[terminal_output]
            )
            
            # Auto-refresh terminal every 2 seconds using timer
            terminal_refresh_timer = gr.Timer(2)
            terminal_refresh_timer.tick(
                refresh_terminal,
                outputs=[terminal_output]
            )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EPUB to Audiobook Converter Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    
    args = parser.parse_args()
    
    log_message("=== EPUB to Audiobook Converter Web UI ===")
    log_message(f"Starting server on {args.host}:{args.port}")
    
    app.queue()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )
