from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
from ngram import load_data


class BytePairEncodingTokenizer:
    """
    A tokenizer that uses the Byte Pair Encoding algorithm to tokenize text.
    """

    def __init__(self):
        self.word_frequencies: Counter[str] = Counter()
        self.vocabulary: Set[str] = set()
        self.words_to_tokens: Dict[str, List[str]] = {}
        self.merge_rules: Dict[Tuple[str, str], str] = {}
        self.SPACE_SYMBOL = "Ġ"

    def pre_tokenize(self, corpus: List[str]) -> Counter[str]:
        """
        Updates the word frequencies with counts for each word in the corpus.
        Each word that is not the first word in a line is prefixed with a space
        symbol (Ġ) to distinguish it from the first word in a line, and then
        added to the Counter.

        Returns:
            A Counter containing the word frequencies in the corpus.
        """

        assert corpus, "😳 Corpus must be non-empty"

        word_frequencies: Counter[str] = Counter()
        for line in corpus:
            words = line.split()
            # Increment the count for the first word in the line without a space
            word_frequencies[words[0]] += 1
            # Update the Counter with the rest of the words prefixed with a
            # space
            word_frequencies.update(self.SPACE_SYMBOL + word for word in words[1:])

        return word_frequencies

    def build_vocabulary(self) -> None:
        """
        Builds the base vocabulary from the word frequencies. The base
        vocabulary is a set containing all the characters used in the corpus.
        """

        assert (
            self.word_frequencies
        ), "😳 Word frequencies must be initialized before building the vocabulary"
        assert not self.vocabulary, "😳 Vocabulary must be empty before building it"

        for word in self.word_frequencies.keys():
            for char in word:
                self.vocabulary.add(char)

    def compute_pair_frequencies(self) -> Dict[Tuple[str, str], int]:
        """
        Compute the frequencies of adjacent token pairs in the vocabulary.

        This method iterates through each word in the word frequencies
        dictionary, splits the word into its constituent tokens, and counts the
        frequency of each adjacent token pair. This is a critical step in the
        BPE training algorithm, as it identifies the most common pairs of tokens
        to merge in subsequent iterations of the algorithm.

        Returns:
            A dictionary of token pairs with their corresponding frequency
            counts. Each key is a tuple containing a pair of tokens, and the
            value is the frequency of that pair within the entire corpus.
        """

        assert self.word_frequencies, "😳 Word frequencies must be initialized"
        assert self.vocabulary, "😳 Vocabulary must be initialized"

        pair_frequencies: Dict[Tuple[str, str], int] = defaultdict(int)

        for word, frequency in self.word_frequencies.items():
            # Get the tokens in the word
            tokens_in_word = self.words_to_tokens[word]

            # If the word itself is a token, skip it as it cannot be adjacent to
            # any other tokens in the corpus. This is because the word itself is
            # separated from the rest of the corpus by spaces, so it can't be
            # merged any further.
            if len(tokens_in_word) == 1:
                continue

            # Iterate through each pair of adjacent tokens in the word
            for i in range(len(tokens_in_word) - 1):
                pair = (tokens_in_word[i], tokens_in_word[i + 1])
                # This pair occurs frequency times, since it is in the word
                pair_frequencies[pair] += frequency

        return pair_frequencies

    def merge_tokens(self, pair: Tuple[str, str]) -> None:
        """
        Merges the two tokens in the pair into a single token.

        This method updates the vocabulary to reflect the merged token, and
        updates the tokens dictionary to reflect the merged token in each word
        in the corpus. This is a critical step in the BPE training algorithm, as
        it merges the most common pair of tokens in the corpus.

        Args:
            pair: A tuple containing the pair of tokens to merge.
        """

        assert self.vocabulary, "😳 Vocabulary must be initialized"
        assert self.words_to_tokens, "😳 Tokens must be initialized"

        # Get the two tokens to merge
        token_1, token_2 = pair

        # Merge the two tokens into a single token
        merged_token = token_1 + token_2

        # Update the tokens dictionary to reflect the merged token
        for word, tokens_in_word in self.words_to_tokens.items():
            # If the word itself is a token, skip it as it cannot possibly
            # contain the two tokens to merge.
            if len(tokens_in_word) == 1:
                continue

            i = 0
            while i < len(tokens_in_word) - 1:
                # If the current token and the next token are the two tokens to
                # merge, replace them with the merged token
                if (tokens_in_word[i], tokens_in_word[i + 1]) == pair:
                    tokens_in_word = (
                        tokens_in_word[:i] + [merged_token] + tokens_in_word[i + 2 :]
                    )
                else:
                    i += 1

            # Update the tokens dictionary with the new tokens in the word
            self.words_to_tokens[word] = tokens_in_word

        # Update the merge rules to reflect the merged token
        self.merge_rules[pair] = merged_token

        # Update the vocabulary to reflect the merged token
        self.vocabulary.add(merged_token)

    def train(self, corpus: List[str]) -> None:
        """
        Trains the tokenizer on the given corpus. This involves splitting each
        word in the corpus into individual tokens, and then merging the most
        common pair of tokens in the corpus. This process is repeated until the
        most common pair of tokens in the corpus occurs less than two times. In
        other words, we merge all adjacent tokens in the corpus that occur at
        least twice.
        """

        # Pre-tokenize the corpus to get the word frequencies
        self.word_frequencies = self.pre_tokenize(corpus)

        # Initialize the vocabulary with all the characters used in the corpus
        self.build_vocabulary()

        # Print the size of the initial vocabulary
        print(f"Initial vocabulary size: {len(self.vocabulary)}")

        # Split each word into individual characters to start the training
        # process
        self.words_to_tokens = {
            word: list(word) for word in self.word_frequencies.keys()
        }

        # Compute the frequencies of adjacent token pairs in the vocabulary
        pair_frequencies = self.compute_pair_frequencies()
        # Get the most frequent pair of tokens in the corpus
        most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)

        while pair_frequencies[most_frequent_pair] > 2:
            # Merge the most frequent pair of tokens in the corpus
            self.merge_tokens(most_frequent_pair)

            # Recompute the pair frequencies
            pair_frequencies = self.compute_pair_frequencies()
            # Get the most frequent pair of tokens in the corpus
            most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the given text using the trained tokenizer. For Byte-Pair
        Encoding, this involves splitting each word in the text into individual
        tokens, and then merging the most common pair of tokens in the corpus.

        Args:
            text: The text to tokenize.

        Returns:
            A list of tokens in the text.
        """

        assert self.vocabulary, "😳 Vocabulary must be initialized"
        assert self.words_to_tokens, "😳 Tokens must be initialized"

        # Pre-tokenize the text to get the word frequencies
        word_frequencies = self.pre_tokenize([text])
        # Get the words themselves
        words = list(word_frequencies.keys())
        # Split each word into individual characters
        tokens = [list(word) for word in words]

        # Merge tokens until we can't merge them anymore
        for (token_1, token_2), merged_token in self.merge_rules.items():
            for idx, tokens_in_word in enumerate(tokens):
                curr_token = 0
                while curr_token < len(tokens_in_word) - 1:
                    # Found an adjacent pair of tokens that can be merged
                    if (
                        tokens_in_word[curr_token] == token_1
                        and tokens_in_word[curr_token + 1] == token_2
                    ):
                        # Replace the two tokens with the merged token within
                        # the list of tokens in the word
                        tokens_in_word = (
                            tokens_in_word[:curr_token]
                            + [merged_token]
                            + tokens_in_word[curr_token + 2 :]
                        )
                    else:
                        curr_token += 1

                tokens[idx] = tokens_in_word

        # Return the tokens in the text
        return [token for tokens_in_word in tokens for token in tokens_in_word]


if __name__ == "__main__":
    corpus = load_data("data/BPE-data.txt")

    # Split the corpus into training and test data
    training_data, test_data = corpus[:4000], corpus[4000:]

    # Initialize the tokenizer and train it on the training data
    tokenizer = BytePairEncodingTokenizer()
    tokenizer.train(training_data)

    # Print the number of tokens in the vocabulary
    print(f"Vocabulary size after training: {len(tokenizer.vocabulary)}")

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
