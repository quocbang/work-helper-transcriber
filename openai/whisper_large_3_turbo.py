# tasks/transcriber/whisper_transcriber.py
from typing import Dict, List, Optional

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        max_duration: int = 30,
        target_sampling_rate: int = 16000,
        language: str = "chinese",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_duration = max_duration
        self.target_sampling_rate = target_sampling_rate
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Initializing Whisper model: {model_name}")
        print(f"Using device: {self.device}")

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

        print("Model initialization complete")

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def transcribe(self, audio_path: str) -> List[Dict]:
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=None)

            if len(audio) == 0:
                raise ValueError(f"Audio file '{audio_path}' is empty")

            if sr != self.target_sampling_rate:
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=self.target_sampling_rate
                )

            # Process audio in chunks
            chunk_size = self.max_duration * self.target_sampling_rate
            transcriptions = []

            for chunk_start in range(0, len(audio), chunk_size):
                chunk = audio[chunk_start : chunk_start + chunk_size]
                if len(chunk) == 0:
                    continue

                # Calculate timestamp
                start_time = chunk_start / self.target_sampling_rate
                end_time = (chunk_start + len(chunk)) / self.target_sampling_rate

                # Process chunk
                inputs = self.processor(
                    chunk,
                    sampling_rate=self.target_sampling_rate,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=self.language, task="transcribe"
                )

                predicted_ids = self.model.generate(
                    inputs.input_features,
                    forced_decoder_ids=forced_decoder_ids,
                )

                chunk_text = self.processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]

                # Store result with timestamps
                transcriptions.append(
                    {
                        "start": self.format_timestamp(start_time),
                        "end": self.format_timestamp(end_time),
                        "text": chunk_text.strip(),
                    }
                )

            return transcriptions

        except Exception as e:
            raise RuntimeError(f"Transcription failed for {audio_path}: {str(e)}")
