from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scorer import Scorer
import torch

import random


class ContradictionScorer(Scorer):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def inference(self, query, passage):
        inputs = self.tokenizer.encode_plus(query, passage, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        contradiction_prob = probs[0][0].item()
        return contradiction_prob

    def evaluate(self, query, passage, inter_sentence=False, n_samples=3):
        """
        Evaluate the contradiction score of a passage given a query.

        Args:
            query (str): The query to evaluate the passage against.
            passage (str): The passage to evaluate.
            inter_sentence (bool): Whether to evaluate inter-sentence contradiction.
            n_samples (int): Number of sentence pairs to sample for inter-sentence contradiction (-1 means all).
        """
        contradiction_prob = self.inference(query, passage)
        sentences = self._split_sentences(passage)
        if inter_sentence:
            inter_sentence_score = self._evaluate_inter_sentence(query, sentences, n_samples)
            return contradiction_prob * 0.7 + inter_sentence_score * 0.3
        return contradiction_prob

    def _sample_sentence_pairs(self, sentences, n_samples):
        sentence_pairs = [(a, b) for a, b in zip(sentences, sentences[1:])]
        if n_samples == -1:
            return sentence_pairs
        return random.sample(sentence_pairs, min(n_samples, len(sentence_pairs)))

    def _evaluate_inter_sentence(self, query, sentences, n_samples):
        sentence_pairs = self._sample_sentence_pairs(sentences, n_samples)
        contradiction_probs = [self.inference(query, a + " " + b) for a, b in sentence_pairs]
        return sum(contradiction_probs) / len(contradiction_probs)


if __name__ == "__main__":
    query = "What is the meaning of life?"
    passage_gpt4 = open("responses/gpt4.txt").read()
    passage_mistral = open("responses/mistral.txt").read()

    cont = ContradictionScorer("facebook/bart-large-mnli")
    print("GPT-4:", cont.evaluate(query, passage_gpt4))
    print("Mistral:", cont.evaluate(query, passage_mistral))
