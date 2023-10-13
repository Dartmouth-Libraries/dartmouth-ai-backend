import spacy
import spacy_fastlang  # Ignore warning about unused import!


class LanguageDetector:
    """Detects the language a text is written in."""

    def __init__(self):
        self.__nlp = spacy.blank("xx")
        self.__nlp.add_pipe("language_detector")

    def detect(self, text: str) -> dict:
        """Detects the language a text is written in.

        Args:
            text (string): The text to be analyzed.

        Returns:
            dict: A dictionary containing the keys language and score.
        """
        detected = self.__nlp(text)
        return {"language": detected._.language, "score": detected._.language_score}


if __name__ == "__main__":
    pass
