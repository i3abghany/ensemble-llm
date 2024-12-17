from collections import Counter
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


class RepetitionScorer:
    def __init__(self, n=4):
        self.n = n

    def repetition_score(self, text):
        words = nltk.word_tokenize(text.lower())
        ngrams = list(nltk.ngrams(words, self.n))
        ngram_counts = Counter(ngrams)
        repeated = {k: v for k, v in ngram_counts.items() if v > 1}
        redundancy_score = sum(repeated.values()) / len(ngram_counts) if ngram_counts else 0

        return repeated, 1 - redundancy_score


if __name__ == "__main__":
    rep = RepetitionScorer(n=4)

    passage_gpt4 = open('responses/gpt4.txt').read()
    _, score = rep.repetition_score(passage_gpt4)

    print("GPT-4:")
    print("Repetition score:", score)

    passage_mistral = open('responses/mistral.txt').read()

    _, score = rep.repetition_score(passage_mistral)

    print("Mistral:")
    print("Repetition score:", score)