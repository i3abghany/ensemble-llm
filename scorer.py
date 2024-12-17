import re


class Scorer:
    def _split_sentences(self, text):
        return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    def evaluate(self, query, passage):
        raise NotImplementedError
