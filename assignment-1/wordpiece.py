from collections import Counter
from typing import Dict, List, Tuple, Set


def load_data(file_path: str) -> List[str]:
    """
    Loads the data from the specified file path and returns a list of lines from
    the file, stripped of whitespace.
    """

    with open(file_path) as f:
        return [line.strip() for line in f]


class WordPieceTokenizer:
    """
    A tokenizer that uses the WordPiece algorithm to tokenize text.
    """

    def __init__(self):
        self.word_frequencies: Counter[str] = Counter()
        self.vocabulary: Set[str] = set()
        self.words_to_tokens: Dict[str, List[str]] = {}
        self.merge_rules: Dict[Tuple[str, str], str] = {}
        self.PREFIX_SYMBOL = "##"

    def pre_tokenize(self, corpus: List[str]) -> Counter[str]:
        """
        Updates the word frequencies with counts for each word in the corpus.

        Returns:
            A Counter containing the word frequencies in the corpus.
        """

        assert corpus, "😳 Corpus must be non-empty"

        word_frequencies: Counter[str] = Counter()
        for line in corpus:
            word_frequencies.update(line.split())

        return word_frequencies

    def build_vocabulary(self) -> None:
        """
        Builds the base vocabulary from the word frequencies. The base vocabulary
        is a set containing all the characters used in the corpus, and we add
        a special prefix symbol (##) to each character that is not the start of
        a word.
        """

        assert (
            self.word_frequencies
        ), "😳 Word frequencies must be initialized before building the vocabulary"
        assert not self.vocabulary, "😳 Vocabulary must be empty before building it"

        for word in self.word_frequencies:
            self.vocabulary.add(word[0])

            for char in word[1:]:
                self.vocabulary.add(self.PREFIX_SYMBOL + char)

    def compute_pair_scores(self) -> Dict[Tuple[str, str], float]:
        """
        Computes the scores for each pair of adjacent tokens in the vocabulary.
        Scores are calculated using the formula:
                score(ab) = frequency(ab) / frequency(a) * frequency(b)
        where ab is an adjacent pair of tokens, and a and b are the tokens in the
        pair. This formula is taken from the original WordPiece paper.

        Returns:
            A dictionary mapping each pair of adjacent tokens to its score.
        """

        assert self.word_frequencies, "😳 Word frequencies must be initialized"

        token_frequencies: Counter[str] = Counter()
        pair_frequencies: Counter[Tuple[str, str]] = Counter()

        for word, frequency in self.word_frequencies.items():
            tokens_in_word = self.words_to_tokens[word]

            # If the word itself is a single token, we don't need to continue
            # processing the pair frequencies. Just add its frequency to the
            # token frequencies and continue.
            if len(tokens_in_word) == 1:
                token_frequencies[tokens_in_word[0]] += frequency
                continue

            # Process each pair of adjacent tokens in the word
            for i in range(len(tokens_in_word) - 1):
                # Get the pair and add its frequency to the pair frequencies
                pair = (tokens_in_word[i], tokens_in_word[i + 1])
                pair_frequencies[pair] += frequency

                # Add the first token in the pair to the token frequencies
                token_frequencies[tokens_in_word[i]] += frequency

            # Add the last token in the word to the token frequencies to account
            # for the for loop stopping before the last token
            token_frequencies[tokens_in_word[-1]] += frequency

        pair_scores = {
            pair: frequency / (token_frequencies[pair[0]] * token_frequencies[pair[1]])
            for pair, frequency in pair_frequencies.items()
        }

        return pair_scores

    def merge_tokens(self, pair: Tuple[str, str]) -> None:
        """
        Merge the pair of tokens into a single token. The new token is the
        concatenation of the two tokens in the pair, and the merge rule is
        stored in the merge rules dictionary.

        For example, if the two tokens were "##a" and "##b", the new token would
        be "##ab", and the merge rule would be ("##a", "##b") -> "##ab". If the
        two tokens were "h" and "##u", the new token would be "hu", and the merge
        rule would be ("h", "##u") -> "hu".
        """

        assert pair, "😳 Pair must be non-empty"

        # Get the two tokens to merge
        token1, token2 = pair

        # Merge the two tokens into a single token
        if token2.startswith(self.PREFIX_SYMBOL):
            # If the second token is a prefix token, we need to remove the prefix
            # symbol from the second token before merging
            merged_token = token1 + token2[len(self.PREFIX_SYMBOL) :]
        else:
            merged_token = token1 + token2

        print(pair, "->", merged_token)

        for word, tokens_in_word in self.words_to_tokens.items():
            # If the word itself is a single token, it cannot possibly contain
            # the pair of tokens we are trying to merge. Just continue.
            if len(tokens_in_word) == 1:
                continue

            i = 0
            while i < len(tokens_in_word) - 1:
                if tokens_in_word[i] == token1 and tokens_in_word[i + 1] == token2:
                    tokens_in_word = (
                        tokens_in_word[:i] + [merged_token] + tokens_in_word[i + 2 :]
                    )
                else:
                    i += 1

            self.words_to_tokens[word] = tokens_in_word

        # Add the merge rule to the merge rules dictionary
        self.merge_rules[pair] = merged_token

        # Update the vocabulary
        self.vocabulary.add(merged_token)

    def train(self, corpus: List[str]) -> None:
        # Pre-tokenize the corpus to get the word frequencies
        self.word_frequencies = self.pre_tokenize(corpus)

        # Build the base vocabulary
        self.build_vocabulary()

        # Split each word into individual characters to start the training process
        self.words_to_tokens = {
            word: [
                char if i == 0 else self.PREFIX_SYMBOL + char
                for i, char in enumerate(word)
            ]
            for word in self.word_frequencies.keys()
        }

        # Merge the most frequent pair of adjacent tokens in the vocabulary
        # until we have learned 4000 merge rules (i.e. the vocabulary has size
        # 4000 excluding the base vocabulary)
        # while len(self.merge_rules) < 4000:
        while len(self.vocabulary) < 70:
            # Compute the scores for each pair of adjacent tokens in the vocabulary
            pair_scores = self.compute_pair_scores()

            # Get the pair with the highest score
            best_pair = max(pair_scores, key=pair_scores.get)

            # Merge the pair
            self.merge_tokens(best_pair)

    def tokenize_word(self, word: str) -> List[str]:
        """
        Tokenizes the given word using the trained tokenizer. For WordPiece,
        tokenization is done by greedily matching the longest possible token
        in the vocabulary at each step.

        Args:
            word: The word to tokenize.

        Returns:
            A list of tokens in the word.
        """

        assert word, "😳 Word must be non-empty"
        assert self.vocabulary, "😳 Vocabulary must be initialized"

        # Stores the tokens in the word
        tokens: List[str] = []

        while len(word) > 0:
            # We initialize i to the length of the word and decrement it until
            # we find a substring match from the beginning of the word to i
            # that is in the vocabulary
            i = len(word)
            while i > 0 and word[:i] not in self.vocabulary:
                i -= 1

            # No match found, which means the word is not in the vocabulary and
            # cannot be tokenized.
            if i == 0:
                return ["[UNK]"]

            # Match found! Add the token to the list of tokens
            tokens.append(word[:i])

            # Remove the token from the word and continue
            word = word[i:]

            # After the first iteration, if we still have parts of the word
            # left to tokenize, we need to add the prefix symbol to the token.
            # This is because the remaining parts of the word are not the
            # beginning of a word, so we need to add the prefix symbol to
            # correctly tokenize them.
            if len(word) > 0:
                word = self.PREFIX_SYMBOL + word

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the given text using the trained tokenizer. For WordPiece,
        tokenization is done by greedily matching the longest possible token
        in the vocabulary at each step.

        Args:
            text: The text to tokenize.

        Returns:
            A list of tokens in the text.
        """

        assert text, "😳 Text must be non-empty"
        assert self.vocabulary, "😳 Vocabulary must be initialized"

        # Pre-tokenize the text to get the word frequencies
        word_frequencies = self.pre_tokenize([text])
        # Get the words themselves
        words = list(word_frequencies.keys())

        # Tokenize each word into a list of list of tokens
        tokenized_words = [self.tokenize_word(word) for word in words]

        return [token for tokenized_word in tokenized_words for token in tokenized_word]


if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        corpus = load_data("test.txt")

        tokenizer = WordPieceTokenizer()
        tokenizer.train(corpus)

        print(tokenizer.vocabulary)

        print(tokenizer.tokenize("Hugging"))
        print(tokenizer.tokenize("HOgging"))
        print(tokenizer.tokenize("This is the Hugging Face course!"))

        exit()

    corpus = load_data("BPE-data.txt")

    # Split the corpus into training and test data
    training_data, test_data = corpus[:4000], corpus[4000:]

    # Initialize the tokenizer and train it on the training data
    tokenizer = WordPieceTokenizer()
    tokenizer.train(training_data)

    print(f"Vocabulary size: {len(tokenizer.vocabulary)}")

    # Tokenize the test data
    tokenized_test_data = [tokenizer.tokenize(line) for line in test_data]
    for line, tokens in zip(test_data, tokenized_test_data)[:10]:
        print(line)
        print(tokens)
        print()
    
    print(
        tokenizer.tokenize(
            "Analysts were expecting the opposite, a deepening of the deficit."
        )
    )
    print(
        tokenizer.tokenize(
            "Five minutes later, a second person arrived, aged around thirty, with knife wounds."
        )
    )
