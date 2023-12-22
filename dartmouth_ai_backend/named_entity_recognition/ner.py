""" Named Entity Recognition """
import spacy
from spacy import displacy


class NamedEntityRecognizer:
    def __init__(self, lang="en"):
        if lang != "en":
            raise NotImplementedError(
                "Named Entity Recognition is currently "
                "only supported for English texts."
            )

        self.__nlp = spacy.load("en_core_web_trf", disable=["parser"])

    def recognize(self, text):
        recognized = self.__nlp(text)
        tags = set()

        for entity in recognized.ents:
            tags.add(entity.label_)

        result = dict()
        result["tag_key"] = {tag: spacy.explain(tag) for tag in sorted(tags)}
        result["html"] = displacy.render(recognized, style="ent")

        return result

    @staticmethod
    def how_to_cite(format="bibtex") -> str:
        if format != "bibtex":
            return NotImplemented
        return """@misc{spacy,
    author = {Honnibal, Matthew and Montani, Ines and Van Landeghem, Sofie and Boyd, Adriane},
    doi = {10.5281/zenodo.1212303},
    title = {spaCy: Industrial-strength Natural Language Processing in Python}
}"""


if __name__ == "__main__":
    pass
