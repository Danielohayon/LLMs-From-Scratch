import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from tokenizer import Tokenizer




class Text8(Dataset):
    def __init__(self, max_length, tokenizer, usecase="train"):
        super(Text8, self).__init__()
        self.usecase = usecase
        self.dataset = load_dataset("afmck/text8")
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.letter_to_index)
        self.dataset_length = len(self.dataset[self.usecase][0]["text"])
        self.tokenized = torch.tensor(
            [
                self.tokenizer.encode(i)
                for i in self.dataset[self.usecase][0]["text"]
            ]
        )
        self.max_length = max_length

    def __len__(self):
        return self.dataset_length - self.max_length - 1

    def __getitem__(self, i):
        return (
            self.tokenized[i : i + self.max_length],
            self.tokenized[i + 1 : i + self.max_length + 1],
        )
