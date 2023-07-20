""" Named Entity Recognition """
import spacy
from spacy import displacy


class NamedEntityRecognizer:
    def __init__(self, lang='en'):
        if lang != 'en':
            raise NotImplementedError("Named Entity Recognition is currently "
                                      "only supported for English texts.")

        self.__nlp = spacy.load("en_core_web_trf")

    def recognize(self, text):
        recognized = self.__nlp(text)
        tags = set()

        for entity in recognized.ents:
            print(entity.text, entity.label_)
            tags.add(entity.label_)

        for tag in sorted(tags):
            print(f"{tag}: ", spacy.explain(tag))

        return displacy.render(recognized, style="ent")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='NER',
        description='Named Entity Recognition',
        epilog='Extract named entities, like country names, events, or dates, from a given text.'
        )

    parser.add_argument('textfile', metavar='txt', type=str,
                        help='The name of the text file to process.')
    parser.add_argument('-o', '--output')

    args = parser.parse_args()

    with open(args.textfile) as f:
        raw_text = f.read()

    ner = NamedEntityRecognizer()
    html_text = ner.recognize(raw_text)
    with open(args.output, 'w') as f:
        f.write(html_text)
    print("Done.")
