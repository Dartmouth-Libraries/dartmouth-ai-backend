""" Language detection """
import spacy
import spacy_fastlang  # Ignore warning about unused import!


class LanguageDetector:
    """Detects the language of a text"""

    def __init__(self):
        self.__nlp = spacy.load("en_core_web_trf")
        self.__nlp.add_pipe("language_detector")

    def detect(self, text):
        detected = self.__nlp(text)
        return {"language": detected._.language, "score": detected._.language_score}


if __name__ == "__main__":
    pass
