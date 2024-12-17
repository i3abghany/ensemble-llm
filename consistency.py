import re
from sentence_transformers import SentenceTransformer, util


class CoherenceScorer:
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
        return sum(res) / len(res)

    def _split_sentences(self, text):
        return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    def get_coherence_score(self, query, passage):
        sentences = self._split_sentences(passage)
        overall_relevance = self._get_relevance(query, passage)
        inter_sentence_relevance = self._get_inter_sentence_relevalnce(sentences)
        return overall_relevance * 0.4 + inter_sentence_relevance * 0.6


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    coh = CoherenceScorer(model, 32768)
    query = "What is the meaning of life?"

    passage_gpt4 = open("responses/gpt4.txt").read()
    print("GPT-4:", coh.get_coherence_score(query, passage_gpt4))

    passage_mistral = open("responses/mistral.txt").read()
    print("Mistral:", coh.get_coherence_score(query, passage_mistral))
