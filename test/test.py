from dartmouth_ai_backend.language_detection import LanguageDetector
from dartmouth_ai_backend.named_entity_recognition import NamedEntityRecognizer

import json

def test_language_detection():
    with open("test/de.txt") as f:
        de_text = f.read()

    r = LanguageDetector().detect(de_text)
    assert r[0], "de"

    with open("test/en.txt") as f:
        en_text = f.read()

    r = LanguageDetector().detect(en_text)
    assert r[0], "en"


def test_named_entity_recognition():
    with open("test/wiki.txt") as f:
        raw_text = f.read()

    ner = NamedEntityRecognizer()
    result = ner.recognize(raw_text)
    assert result


if __name__ == "__main__":
    test_language_detection()
    test_named_entity_recognition()
