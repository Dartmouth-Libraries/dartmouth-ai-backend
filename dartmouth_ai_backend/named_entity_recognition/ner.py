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
            tags.add(entity.label_)

        result = dict()
        result['tag_key'] = {
            tag: spacy.explain(tag)
            for tag in sorted(tags)
        }
        result['html'] = displacy.render(recognized, style="ent")

        return result


if __name__ == "__main__":
    pass
