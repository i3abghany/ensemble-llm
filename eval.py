from sentence_transformers import SentenceTransformer, util

from scorer import Scorer
from nli import ContradictionScorer
from consistency import ConsistencyScorer
from rep import RepetitionScorer

CONSISTENCY_MODEL = "all-MiniLM-L6-v2"
CONSISTENCY_MAX_LENGTH = 32768
NLI_MODEL = "facebook/bart-large-mnli"
NGRAM_N = 4


class ResponseEvaluator(Scorer):
    def __init__(self):
        self.nli_scorer = ContradictionScorer(NLI_MODEL)
        self.consistency_model = SentenceTransformer(CONSISTENCY_MODEL)
        self.consistency_scorer = ConsistencyScorer(self.consistency_model, CONSISTENCY_MAX_LENGTH)
        self.repetition_scorer = RepetitionScorer(NGRAM_N)

    def _combine_scores(self, contradiction_score, consistency_score, repetition_score):
        print("Contradiction score:", contradiction_score)
        print("Consistency score:", consistency_score)
        print("Repetition score:", repetition_score)
        return contradiction_score * 0.5 + consistency_score * 0.3 + repetition_score * 0.2

    def evaluate(self, query, passage):
        contradiction_score = self.nli_scorer.evaluate(query, passage, inter_sentence=True, n_samples=10)
        consistency_score = self.consistency_scorer.evaluate(query, passage)
        _, repetition_score = self.repetition_scorer.evaluate(query, passage)
        return self._combine_scores(contradiction_score, consistency_score, repetition_score)


if __name__ == "__main__":
    query = "Can you provide specific details, such as dates or studies, to support the assertion that humans can breathe underwater without assistance?"
    passage_gpt4 = open("responses/gpt4.txt").read()
    passage_mistral = open("responses/mistral.txt").read()
    passage_gemma = open("responses/gemma.txt").read()

    ev = ResponseEvaluator()
    print("GPT-4:", ev.evaluate(query, passage_gpt4))
    print("Mistral:", ev.evaluate(query, passage_mistral))
    print("Gemma:", ev.evaluate(query, passage_gemma))
