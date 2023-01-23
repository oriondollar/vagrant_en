import torch
from vagrant.utils import encode_smiles, one_hot, tokenizer

class StringPreprocessor:
    """
    Preprocessing code preparing SMILES or SELFIES strings for training
    """
    def __init__(self, vocab, max_length):
        self.vocab = vocab
        self.max_length = max_length+2 ### Including <start> and <end> tokens

    def preprocess(self, smiles):
        tokenized_smiles = [tokenizer(smi) for smi in smiles]
        encoded_smiles = []
        for i, tok_smi in enumerate(tokenized_smiles):
            encoded_smi = torch.tensor(encode_smiles(tok_smi, self.max_length, self.vocab))
            encoded_smiles.append(encoded_smi)
        encoded_smiles = torch.stack(encoded_smiles)
        return encoded_smiles.long()
