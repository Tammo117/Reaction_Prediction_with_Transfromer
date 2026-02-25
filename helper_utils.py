import re
import numpy as np
from collections import Counter
import torch
import torch.nn as nn


class tokenize_index_pairs:
    """
    Class to tokenize and index translation pairs of reactants and products
     - Builds a vocabulary from the translation pairs
     - Converts reactant and product SMILES strings into sequences of indices
     - Handles special tokens for start/end of sequence and padding
     - Computes length statistics to determine max sequence length for padding/truncation
     - Uses a regex to tokenize SMILES strings into meaningful components (atoms, bonds, etc.)
    """
    
    def __init__(self, translation_pairs):
        self.translation_pairs = translation_pairs
        self.SMILES_REGEX = r"(\[[^\]]+\]|Br?|Cl?|Si?|Se?|Na?|Li?|Ca?|Mg?|Al?|@@?|=|#|\(|\)|\.|\/|\\|\+|\-|\d|[A-Za-z])"
        stats = self.compute_length_statistics()
        self.MAX_LENGTH = stats["99_percentile"] + 2  # Add 2 for <sos> and <eos> tokens

    def run(self):
        vocab, word2idx, idx2word = self.build_vocab()
        prepared_pairs = []
        for reactant, product in self.translation_pairs:
            reactant_indices = self.prepare_sequence(reactant, word2idx, add_special_tokens=False)
            product_indices = self.prepare_sequence(product, word2idx)
            prepared_pairs.append((reactant_indices, product_indices))

        print(f"Prepared {len(prepared_pairs)} translation pairs with max sequence length {self.MAX_LENGTH}")
        print(f"Example original pair (reactant indices, product indices): {self.translation_pairs[0]}")
        print(f"Example prepared pair (reactant indices, product indices): {prepared_pairs[0]}")

        return prepared_pairs, vocab, word2idx, idx2word, self.MAX_LENGTH
    
    def compute_length_statistics(self):
        lengths = []

        for reactant, product in self.translation_pairs:
            r_len = len(self.tokenize_smiles(reactant))
            p_len = len(self.tokenize_smiles(product))
            lengths.append(r_len)
            lengths.append(p_len)

        lengths = np.array(lengths)

        stats = {
            "max": int(np.max(lengths)),
            "mean": float(np.mean(lengths)),
            "95_percentile": int(np.percentile(lengths, 95)),
            "99_percentile": int(np.percentile(lengths, 99)),
        }

        return stats

    def tokenize_smiles(self, smiles):
        return re.findall(self.SMILES_REGEX, smiles)
    
    def build_vocab(self,min_freq=3):
        """
        Build vocabulary from sentences
        """
        counter = Counter()  # Counter to count word frequencies in all sentences
        for reactant, product in self.translation_pairs:
            counter.update(self.tokenize_smiles(reactant))
            counter.update(self.tokenize_smiles(product))
        
        # Start vocab with special tokens for translation
        # <pad>: padding token, <unk>: unknown token, <sos>: start of sequence, <eos>: end of sequence
        vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + [w for w, c in counter.items() if c >= min_freq]
        
        # Create a mapping from word to unique index
        word2idx = {w: i for i, w in enumerate(vocab)}
        # Create a mapping from index back to word (inverse of word2idx)
        idx2word = {i: w for i, w in enumerate(vocab)}
        
        # Return the vocab list and the two dictionaries
        return vocab, word2idx, idx2word

    def prepare_sequence(self, sentence, word2idx, add_special_tokens=True):
        """
        Convert a sentence to a list of indices with special tokens
        """
        tokens = self.tokenize_smiles(sentence)
        
        if add_special_tokens:
            # Add <sos> at the beginning and <eos> at the end
            tokens = ['<sos>'] + tokens + ['<eos>']
        
        # Convert tokens to indices
        indices = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) < self.MAX_LENGTH:
            # Pad with <pad> tokens
            indices = indices + [word2idx['<pad>']] * (self.MAX_LENGTH - len(indices))
        else:
            # Truncate if too long
            indices = indices[:self.MAX_LENGTH]
        
        return indices
    

class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings using sinusoidal patterns.
    
    Since transformers don't have inherent notion of sequence order (unlike RNNs),
    we add positional encodings to give the model information about where each
    token appears in the sequence.
    """
    def __init__(self, max_len, d_model):
        """
        Initialize positional encoding matrix.
        
        Args:
            max_len (int): Maximum sequence length the model will handle
                          (e.g., 100 for sentences up to 100 tokens)
            d_model (int): Dimension of the model's embeddings 
                          (e.g., 256 or 512 - must match embedding size)
        
        Creates a fixed sinusoidal pattern matrix of shape [max_len, d_model]
        where each row represents the positional encoding for that position.
        """
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trained, but saved with model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (Tensor): Token embeddings of shape [batch_size, seq_len, d_model]
                       where seq_len <= max_len from initialization
        
        Returns:
            Tensor: Positional encodings of shape [batch_size, seq_len, d_model]
                   (same shape as input, ready to be added to embeddings)
        
        Example:
            If x represents embeddings for "I love cats" (3 tokens):
            - Input x shape: [batch_size, 3, 256]
            - Output shape: [batch_size, 3, 256]
            - Returns positions 0, 1, 2 encoded as 256-dim vectors
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]
    
def show_model_layers(model):
    """
    Display the 4 main layers of the TranslationEncoder model.
    """
    print("\n" + "=" * 70)
    print(f" {model.__class__.__name__} - Main Layers")
    print("=" * 70)
    print(f"\n{'Layer':<30} {'Type':<25} {'Parameters':>15}")
    print("-" * 70)
    
    # Show the 4 main layers
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_type = module.__class__.__name__
        print(f"{name:<30} {module_type:<25} {params:>15,}")
    
    print("-" * 70)
    total = sum(p.numel() for p in model.parameters())
    print(f"{'TOTAL':<30} {'':<25} {total:>15,}")
    print("=" * 70)