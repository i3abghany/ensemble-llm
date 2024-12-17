import torch
import re
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class CoherenceScorer:
    def __init__(self, model, max_length):
        self.model = model
        self.max_length = max_length
        self._init_prefixes()

    def _init_prefixes(self):
        self._task_name_to_instruct = {
            "Overall": "Given a question, how well does the passage answer the question?",
            "Inter-Sentence": "Given two sentences, how well does the second sentence follow the first?",
        }
        self.overall_query_prefix = "Instruct" + self._task_name_to_instruct["Overall"] + "\nQuery: "
        self.inter_sentence_query_prefix = "Instruct" + self._task_name_to_instruct["Inter-Sentence"] + "\nQuery: "
        self.passage_prefix = ""

    def _get_relevance(self, query, passage):
        qe = self.model.encode([query], instruction=self.query_prefix, max_length=self.max_length)
        pe = self.model.encode([passage], instruction=self.passage_prefix, max_length=self.max_length)

        qe, pe = F.normalize(qe, p=2, dim=-1), F.normalize(pe, p=2, dim=-1)
        return torch.mm(qe, pe.T).item()

    def _get_inter_sentence_relevalnce(self, sentences):
        res = []
        self.query_prefix = self.inter_sentence_query_prefix
        for s1, s2 in zip(sentences, sentences[1:]):
            res.append(self._get_relevance(s1, s2))

        return sum(res) / len(res)

    def _split_sentences(self, text):
        return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    def get_coherence_score(self, query, passage):
        sentences = self._split_sentences(passage)
        self.query_prefix = self.overall_query_prefix
        overall_relevance = self._get_relevance(query, passage)
        inter_sentence_relevance = self._get_inter_sentence_relevalnce(sentences)
        return overall_relevance * inter_sentence_relevance


if __name__ == "__main__":
    model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
    coh = CoherenceScorer(model, 32768)
    query = "What is the meaning of life?"

    passage_gpt4 = open("responses/gpt4.txt").read()
    print("GPT-4:", coh.get_coherence_score(query, passage_gpt4))

    passage_mistral = open("responses/mistral.txt").read()
    print("Mistral:", coh.get_coherence_score(query, passage_mistral))
