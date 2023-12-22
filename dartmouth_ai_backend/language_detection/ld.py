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

    @staticmethod
    def how_to_cite(format="bibtex") -> str:
        if format != "bibtex":
            return NotImplemented
        return """@article{joulin2016fasttext,
    title={FastText.zip: Compressing text classification models},
    author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
    journal={arXiv preprint arXiv:1612.03651},
    year={2016}
}"""


if __name__ == "__main__":
    pass
