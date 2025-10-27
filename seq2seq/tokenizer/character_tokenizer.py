from .tokenizer import Tokenizer

import torch


class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = True):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        If verbose is True, prints out the vocabulary.

        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = """aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}’•–í€óá«»… º◦©ö°äµ—ø­·òãñ―½¼γ®⇒²▪−√¥£¤ß´úª¾є™，ﬁõ  �►□′″¨³‑¯≈ˆ§‰●ﬂ⇑➘①②„≤±†✜✔➪✖◗¢ไทยếệεληνικαåşıруский 한국어汉语ž¹¿šćþ‚‛─÷〈¸⎯×←→∑δ■ʹ‐≥τ;∆℡ƒð¬¡¦βϕ▼⁄ρσ⋅≡∂≠π⎛⎜⎞ω∗"""
        for idx, char in enumerate(self.characters):
            self.vocab[char] = idx
    
        if verbose:
            print("Vocabulary:", self.vocab)

        #raise NotImplementedError("Need to implement vocab initialization")

    def encode(self, text: str) -> torch.Tensor:

        tokens = []

        for char in text.lower(): 
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                print("damn bro did not make sure the chars are in vocab")

        return torch.tensor(tokens)
    
        # raise NotImplementedError(
        #     "Need to implement encoder that converts text to tensor of tokens."
        # )

    def decode(self, tokens: torch.Tensor) -> str:
        reverse = {v: k for k, v in self.vocab.items()}
        decoded = []

        for token in tokens:
            if token.item() in reverse:
                decoded.append(reverse[token.item()])
            else:
                print("damn bro did not make sure the tokens are in vocab")

        return "".join(decoded)

        # raise NotImplementedError(
        #     "Need to implement decoder that converts tensor of tokens to text."
        # )

# x = CharacterTokenizer(True)
# encoding = x.encode("Dasari? Is that like the water brand?")
# decoding = x.decode(encoding)
# print(encoding)
# print(decoding)