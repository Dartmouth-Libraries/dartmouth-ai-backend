""" Language detection """
import spacy
import spacy_fastlang  # Ignore warning about unused import!


class LanguageDetector:
    def __init__(self, lang='en'):
        if lang != 'en':
            raise NotImplementedError("Language Detection is currently "
                                      "only supported for English texts.")

        self.__nlp = spacy.load("en_core_web_trf")
        self.__nlp.add_pipe("language_detector")

    def detect(self, text):
        detected = self.__nlp(text)
        return detected._.language, detected._.language_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='ld',
        description='Language detection',
        epilog='Detects the language of a given text.'
        )

    parser.add_argument('textfile', metavar='txt', type=str,
                        help='The name of the text file to process.')

    args = parser.parse_args()

    with open(args.textfile) as f:
        raw_text = f.read()

    ld = LanguageDetector()
    print(*ld.detect(raw_text))
