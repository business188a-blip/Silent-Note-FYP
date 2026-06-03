import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> int:
    print("=" * 50)
    print("  SilentNote Model Cache Warmup")
    print("=" * 50)

    print("\n[1/3] Loading Whisper model...")
    from modules.transcriber import Transcriber

    Transcriber().load_model(on_progress=print)
    print("      Whisper is ready.")

    print("\n[2/3] Loading NLP models...")
    from modules.action_extractor import ActionExtractor
    from modules.summarizer import Summarizer

    Summarizer().load()
    ActionExtractor().load()
    print("      NLP models are ready.")

    print("\n[3/3] Loading emotion model...")
    from modules.emotion_detector import EmotionDetector

    EmotionDetector().load()
    print("      Emotion detection is ready.")

    print("\n" + "=" * 50)
    print("  Local models/cache warmed.")
    print("  Now run: python main.py")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
