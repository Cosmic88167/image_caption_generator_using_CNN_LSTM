"""Standalone TTS worker — avoids COM threading issues on Windows when called from Gradio."""
import sys
import pyttsx3
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python tts_worker.py <text> <output_wav_path>", file=sys.stderr)
        sys.exit(1)
    text = sys.argv[1]
    output_path = sys.argv[2]
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        if os.path.exists(output_path):
            print("OK")
        else:
            print("FAIL: file not created")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    main()

