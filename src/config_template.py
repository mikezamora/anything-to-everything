"""
Configuration file template for EPUB to Audiobook converter
Copy this file and customize for your needs
"""

# Text Processing
TEXT_PROCESSING = {
    "segment_words": 500,        # Target words per segment
    "max_words": 600,            # Maximum words per segment
    "min_words": 100,            # Minimum words per segment
    "use_ollama": False,         # Enable Ollama text cleanup
    "ollama_model": "aratan/DeepSeek-R1-32B-Uncensored:latest",    # Ollama model to use
    "ollama_url": "http://host.docker.internal:11434",  # Ollama API URL
}

# TTS Generation
TTS_GENERATION = {
    "temperature": 0.8,          # Generation temperature (0.0-1.0)
    "top_p": 0.8,                # Top-p sampling (0.0-1.0)
    "top_k": 30,                 # Top-k sampling
    "repetition_penalty": 10.0,  # Repetition penalty
    "length_penalty": 0.0,       # Length penalty
    "num_beams": 3,              # Number of beams for beam search
    "max_text_tokens": 120,      # Max text tokens per TTS segment
}

# Audio Settings
AUDIO_SETTINGS = {
    "interval_silence": 200,     # Silence between sentences (ms)
    "segment_silence": 500,      # Silence between segments (ms)
}

# Emotion Settings
EMOTION_SETTINGS = {
    "emo_alpha": 1.0,            # Emotion blend strength (0.0-1.0)
    "use_emo_text": False,       # Auto-detect emotion from text
    # Emotion vector: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    "emo_vector": None,          # e.g., [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.3]
}

# Model Settings
MODEL_SETTINGS = {
    "config": "checkpoints/config.yaml",  # Path to config file
    "model_dir": "checkpoints",           # Path to model directory
    "use_fp16": False,                    # Use FP16 precision
    "device": None,                       # Device (None=auto, "cuda:0", "cpu")
    "use_cuda_kernel": True,              # Use CUDA kernel for BigVGAN
    "use_deepspeed": False,               # Use DeepSpeed
}

# Output Settings
OUTPUT_SETTINGS = {
    "work_dir": "./work",        # Working directory for temporary files
    "keep_segments": False,      # Keep individual segment audio files
    "output_format": "wav",      # Output format: 'wav' or 'm4b'
    "verbose": False,            # Verbose output
}
