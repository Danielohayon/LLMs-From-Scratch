
import torch

class Tokenizer():
    def __init__(self, text):
        tokens = sorted(list(set(text)))
        self.letter_to_index = dict([ (token, idx) for idx, token in enumerate(tokens)])
        self.index_to_letter = dict([ (idx, token) for token, idx in self.letter_to_index.items() ])

    def encode(self, text):
        return torch.tensor([self.letter_to_index[char] for char in text])

    def decode(self, indexes):
        return "".join([self.index_to_letter[i] for i in indexes])


