""" Sentiment Analysis """
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob # Ignore warning about unused import!


class SentimentAnalyzer:
    def __init__(self, lang='en'):
        if lang != 'en':
            raise NotImplementedError("Sentiment Analysis is currently "
                                      "only supported for English texts.")

        self.__nlp = spacy.load("en_core_web_trf")
        self.__nlp.add_pipe('spacytextblob')

    def analyze(self, text):
        analyzed = self.__nlp(text)
        return analyzed._.blob.polarity, analyzed._.blob.subjectivity


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='sentiment_analysis',
        description='Sentiment Analysis',
        epilog='Detects the sentiment in a given text.'
        )

    parser.add_argument('textfile', metavar='txt', type=str,
                        help='The name of the text file to process.')

    args = parser.parse_args()

    with open(args.textfile) as f:
        raw_text = f.read()

    sa = SentimentAnalyzer()
    print(*sa.analyze(raw_text))
