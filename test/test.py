from dartmouth_ai_backend.object_detection import ObjectDetector
from dartmouth_ai_backend.language_detection import LanguageDetector
from dartmouth_ai_backend.named_entity_recognition import NamedEntityRecognizer
from dartmouth_ai_backend.speech_recognition import SpeechRecognizer

from pathlib import Path


def test_language_detection():
    with open("test/de.txt") as f:
        de_text = f.read()

    r = LanguageDetector().detect(de_text)
    assert r["language"], "de"

    with open("test/en.txt") as f:
        en_text = f.read()

    r = LanguageDetector().detect(en_text)
    assert r["language"], "en"


def test_named_entity_recognition():
    with open("test/wiki.txt") as f:
        raw_text = f.read()

    ner = NamedEntityRecognizer()
    result = ner.recognize(raw_text)
    assert result


def test_object_detection():
    result = ObjectDetector().detect(
        Path(__file__).parent.resolve() / "object_detection_sample.jpg"
    )
    assert result


def test_speech_recognition():
    result = SpeechRecognizer(model="tiny").transcribe(
        str(Path(__file__).parent.resolve() / "speech_recognition_sample.flac")
    )
    print(result)

    result = SpeechRecognizer(model="medium").transcribe(
        str(Path(__file__).parent.resolve() / "speech_translation_sample.mp3"),
        task="translate"
    )
    print(result)
    assert result


if __name__ == "__main__":
    test_language_detection()
    test_named_entity_recognition()
    test_object_detection()
    test_speech_recognition()
