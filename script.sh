# Process single file
python -m tasks.transcriber.main --audio path/to/audio.wav --language chinese

# Process directory
python -m tasks.transcriber.main --audio path/to/audio/directory --language chinese

# Use different model
python -m tasks.transcriber.main --audio path/to/audio.wav --model "openai/whisper-large-v3" --language chinese
