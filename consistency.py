import re
from scorer import Scorer
from sentence_transformers import SentenceTransformer, util


class ConsistencyScorer(Scorer):
    def __init__(self, model, max_length):
        self.model = model
        self.max_length = max_length

    def _get_relevance(self, query, passage):
        eq = self.model.encode(query, convert_to_tensor=True)
        ep = self.model.encode(passage, convert_to_tensor=True)
        return util.pytorch_cos_sim(eq, ep).item()

    def _get_inter_sentence_relevalnce(self, sentences):
        res = []
        for s1, s2 in zip(sentences, sentences[1:]):
            res.append(self._get_relevance(s1, s2))
        return sum(res) / len(res) if res else 0

    def evaluate(self, query, passage):
        sentences = self._split_sentences(passage)
        overall_relevance = self._get_relevance(query, passage)
        inter_sentence_relevance = self._get_inter_sentence_relevalnce(sentences)
        return overall_relevance * 0.8 + inter_sentence_relevance * 0.2


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(type(model))
    con = ConsistencyScorer(model, 32768)
    query = "What is the meaning of life?"

    passage_gpt4 = open("responses/gpt4.txt").read()
    print("GPT-4:", con.evaluate(query, passage_gpt4))

    passage_mistral = open("responses/mistral.txt").read()
    print("Mistral:", con.evaluate(query, passage_mistral))
