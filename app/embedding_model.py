import spacy


class EmbeddingModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def calculate_embedding(self, input_word: str):
        word = self.nlp(input_word)
        return word.vector.tolist()