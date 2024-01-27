from collections import Counter
from typing import Dict, List, Tuple

from bpe import BytePairEncodingTokenizer
from utils import load_data, produce_scatterplot


class WordPieceTokenizer(BytePairEncodingTokenizer):
    """
    A tokenizer that uses the WordPiece algorithm to tokenize text. This class
    inherits from the BytePairEncodingTokenizer class, which implements the core
    functionality of the Byte Pair Encoding algorithm. Since we were told to
    pretokenize our corpus just like we did for BPE, we can reuse the
    functionality and override the train and tokenize methods to implement the
    WordPiece algorithm.
    """

    def __init__(self):
        super().__init__()  # see bpe.py for more details

    def compute_pair_scores(self) -> Dict[Tuple[str, str], float]:
        """
        Computes the scores for each pair of adjacent tokens in the vocabulary.
        Scores are calculated using the formula:
                score(ab) = frequency(ab) / frequency(a) * frequency(b)
        where ab is an adjacent pair of tokens, and a and b are the tokens in
        the pair. This formula is taken from the original WordPiece paper.

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

    def train(self, corpus: List[str]) -> None:
        # Pre-tokenize the corpus to get the word frequencies
        self.word_frequencies = super().pre_tokenize(corpus)

        # Build the base vocabulary
        super().build_vocabulary()

        # Split each word into individual characters to start the training
        # process
        self.words_to_tokens = {
            word: list(word) for word in self.word_frequencies.keys()
        }

        # Calculate the initial corpus length
        tokenized_corpus = [self.tokenize(line) for line in corpus]
        initial_corpus_length = sum(len(tokens) for tokens in tokenized_corpus)

        # Lists to store vocabulary sizes and corpus lengths
        vocabulary_sizes = [len(self.vocabulary)]
        corpus_lengths = [initial_corpus_length]

        # Track the number of iterations. We only want to tokenize the corpus
        # every 100 iterations, so we use this to keep track of when to do so.
        num_iterations = 0

        # Merge the most frequent pair of adjacent tokens in the vocabulary
        # until we have learned 4000 merge rules
        while len(self.merge_rules) < 4000:
            # Compute the scores for each pair of adjacent tokens in the
            # vocabulary
            pair_scores = self.compute_pair_scores()

            # Get the pair with the highest score
            best_pair = max(pair_scores, key=pair_scores.get)

            print(best_pair)

            # Merge the pair
            super().merge_tokens(best_pair)

            if num_iterations % 100 == 0:
                # Record the size of the vocabulary
                vocabulary_sizes.append(len(self.vocabulary))

                # Tokenize the training data to get the corpus length in tokens
                tokenized_corpus = [self.tokenize(line) for line in corpus]

                corpus_length = sum(len(tokens) for tokens in tokenized_corpus)

                # Record the corpus length in tokens
                corpus_lengths.append(corpus_length)

            num_iterations += 1

        # Plot the vocabulary size and corpus length over time
        produce_scatterplot(
            vocabulary_sizes,
            corpus_lengths,
            title="Vocabulary Size vs Training Corpus Length Over Iterations",
            x_label="Vocabulary Size",
            y_label="Training Corpus Length",
            file_name="wordpiece_training.png",
        )

    def tokenize_word(self, word: str) -> List[str]:
        """
        Tokenizes the given word using the trained tokenizer. For WordPiece,
        tokenization is done by greedily matching the longest possible token in
        the vocabulary at each step.

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
            # we find a substring match from the beginning of the word to i that
            # is in the vocabulary
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

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the given text using the trained tokenizer. For WordPiece,
        tokenization is done by greedily matching the longest possible token in
        the vocabulary at each step.

        Args:
            text: The text to tokenize.

        Returns:
            A list of tokens in the text.
        """

        assert text, "😳 Text must be non-empty"
        assert self.vocabulary, "😳 Vocabulary must be initialized"

        # Pre-tokenize the text to get the word frequencies
        word_frequencies = super().pre_tokenize([text])
        # Get the words themselves
        words = list(word_frequencies.keys())

        # Tokenize each word into a list of list of tokens
        tokens_in_words = [self.tokenize_word(word) for word in words]

        return [token for tokens_in_word in tokens_in_words for token in tokens_in_word]


if __name__ == "__main__":
    corpus = load_data("data/BPE-data.txt")

    # Split the corpus into training and test data
    training_data, test_data = corpus[:4000], corpus[4000:]

    # Initialize the tokenizer and train it on the training data
    tokenizer = WordPieceTokenizer()
    tokenizer.train(training_data)

    # Tokenize the training data
    tokenized_training_data = [tokenizer.tokenize(line) for line in training_data]

    # Print the number of tokens in the training data
    print(
        f"Number of tokens in training data: {sum(len(tokens) for tokens in tokenized_training_data)}"
    )

    # Tokenize the test data
    tokenized_test_data = [tokenizer.tokenize(line) for line in test_data]

    # Print the number of tokens in the test data
    print(
        f"Number of tokens in test data: {sum(len(tokens) for tokens in tokenized_test_data)}"
    )

    print("Analysts were expecting the opposite, a deepening of the deficit.")
    print(
        tokenizer.tokenize(
            "Analysts were expecting the opposite, a deepening of the deficit."
        )
    )

    print(
        "Five minutes later, a second person arrived, aged around thirty, with knife wounds."
    )
    print(
        tokenizer.tokenize(
            "Five minutes later, a second person arrived, aged around thirty, with knife wounds."
        )
    )
