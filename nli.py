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

    def evaluate(self, query, passage):
        contradiction_prob = self.inference(query, passage)
        sentences = self._split_sentences(passage)
        return contradiction_prob

    def _sample_sentence_pairs(self, sentences):
        sentence_pairs = [(a, b) for a, b in zip(sentences, sentences[1:])]
        return random.sample(sentence_pairs, min(3, len(sentence_pairs)))


if __name__ == "__main__":
    query = "What is the meaning of life?"
    passage_gpt4 = open("responses/gpt4.txt").read()
    passage_mistral = open("responses/mistral.txt").read()

    cont = ContradictionScorer("facebook/bart-large-mnli")
    print("GPT-4:", cont.evaluate(query, passage_gpt4))
    print("Mistral:", cont.evaluate(query, passage_mistral))
