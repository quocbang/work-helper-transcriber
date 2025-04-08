import argparse
from pathlib import Path

from openai.whisper_large_3_turbo import WhisperTranscriber


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper")
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to audio file or directory"
    )
    parser.add_argument(
        "--language", type=str, default="chinese", help="Language of the audio"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Whisper model name",
    )

    args = parser.parse_args()

    try:
        # Initialize transcriber
        transcriber = WhisperTranscriber(model_name=args.model, language=args.language)

        audio_path = Path(args.audio)

        if audio_path.is_file():
            # Single file processing
            print(f"Processing file: {audio_path}")
            results = transcriber.transcribe(str(audio_path))
            for result in results:
                if "text" in result:
                    result["text"] = result["text"].replace("\n", " ")
                    print(
                        f"Start: {result['start']} End: {result['end']} Transcription: {result['text']}"
                    )

        elif audio_path.is_dir():
            # Directory processing
            print(f"Processing directory: {audio_path}")
            audio_files = list(audio_path.glob("*.wav")) + list(
                audio_path.glob("*.mp3")
            )

            for file in audio_files:
                print(f"\nProcessing: {file}")
                try:
                    result = transcriber.transcribe(str(file))
                    print("Transcription:")
                    print(result)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

        else:
            raise FileNotFoundError(f"Path not found: {audio_path}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
