from collections import Counter, defaultdict
import math
from typing import List, Set, Tuple


def load_data(file_path: str) -> List[str]:
    """
    Loads the data from the specified file path and returns a list of lines from
    the file, stripped of whitespace.
    """

    # Read the file and split it into lines
    with open(file_path) as f:
        return [line.strip() for line in f]


class NGramModel:
    """
    Represents an N-gram model, where N is specified by the user.
    """

    def __init__(
        self,
        n: int,
        smoothing_constant: float = 1.0,
        lambdas: Tuple[float, float, float] = None,
    ):
        self.n = n
        self.ngram_counts: dict[Tuple[str], int] = defaultdict(int)
        self.context_counts: dict[Tuple[str], int] = defaultdict(int)
        self.vocabulary: Set[str] = set()
        self.smoothing_constant = smoothing_constant
        # Lambdas are the weights for each n-gram model when using interpolation.
        self.lambdas = lambdas

    # Preprocess the data by tokenizing it and prepending N - 1 <START> tokens
    # and appending a <STOP> token to each line. Returns a list of tokenized
    # lines corresponding to each line in the input data.
    def preprocess_data(
        self, data: List[str], is_training_data=False
    ) -> List[List[str]]:
        tokenized_lines = []
        for line in data:
            # Split the line into tokens
            tokens = line.split()

            # Prepend <START> token N - 1 times, where N - 1 is the model order.
            # For example, for a bigram model, prepend <START> once since we
            # need to see one word back. For a trigram model, prepend <START>
            # twice, etc.
            # Also, regardless of the model order, we always append <STOP> once.
            tokens = ["<START>"] * (self.n - 1) + tokens + ["<STOP>"]

            tokenized_lines.append(tokens)

        if not is_training_data:
            # If this is false, this is not our first time preprocessing the
            # data. This means that we have already built the vocabulary, so
            # we want to preprocess the data according to the vocabulary
            # corresponding to our training data. This means we want to UNKify
            # the data using the same words we UNKified the training data with.
            # That way, we evaluate the model on the same data distribution that
            # we trained it on. If we UNKify the dev and test data with different
            # words, then we're evaluating the model on a different data
            # distribution than we trained it on.
            assert (
                self.vocabulary
            ), "😳 Vocabulary is empty. Did you forget to train the model?"
            processed_lines = [
                [
                    token if token in self.vocabulary or token == "<START>" else "<UNK>"
                    for token in line
                ]
                for line in tokenized_lines
            ]

            return processed_lines

        # Count the number of times each token appears in the data.
        token_counts = Counter([token for line in tokenized_lines for token in line])

        # Replace tokens that appear less than 3 times with <UNK>. This helps
        # us handle out-of-vocabulary (OOV) words.
        processed_lines = [
            [token if token_counts[token] >= 3 else "<UNK>" for token in line]
            for line in tokenized_lines
        ]

        # Build the vocabulary for the training data. This is the set of unique
        # tokens in the training data. <STOP> and <UNK> are included in the
        # vocabulary, but <START> is not.
        self.vocabulary = set(
            token if token_counts[token] >= 3 else "<UNK>" for token in token_counts
        )

        # Remove the <START> token from the vocabulary (if it exists. Note
        # that for a unigram model, we don't prepend <START> tokens). We don't
        # want to predict <START> tokens, since they can only appear at the
        # beginning of a sentence.
        self.vocabulary.discard("<START>")

        return processed_lines

    # Return the n-grams in a given sentence, where n = self.n.
    # For example, if self.n = 3, return the set of trigrams for the sentence.
    # If the sentence was ["<START>", "<START>", "I", "love", "CSE", "447",
    # "<STOP>"], this function would return the following set of tuples:
    # - ("<START>", "<START>", "I")
    # - ("<START>", "I", "love")
    # - ("I", "love", "CSE")
    # - ("love", "CSE", "447")
    # - ("CSE", "447", "<STOP>")
    def ngrams(self, sentence: List[str]) -> List[Tuple[str]]:
        ngrams = []
        for i in range(len(sentence) - (self.n - 1)):
            # Get the next n words in the sentence
            ngram = tuple(sentence[i : i + self.n])
            ngrams.append(ngram)

        return ngrams

    # Train the N-gram model on the specified training data.
    def train(self, training_data: List[str]) -> None:
        # Preprocess the data to prepare it for training. This means
        # tokenizing each sentence and adding N - 1 <START> tokens and a <STOP>
        # token to each sentence. Tokens that appear less than 3 times in the
        # training data are replaced with <UNK>.
        # Next, build the vocabulary for the training data. This is the set of
        # unique tokens in the training data. <STOP> and <UNK> are included in
        # the vocabulary, but <START> is not.
        self.training_data = self.preprocess_data(training_data, is_training_data=True)

        # Update the n-gram counts and context counts based on the training data.
        # Recall that the n-gram counts are the number of times each n-gram
        # appears in the training data, and the context counts are the number
        # of times each context appears in the training data.
        for sentence in self.training_data:
            for ngram in self.ngrams(sentence):
                self.ngram_counts[ngram] += 1
                # Get the first n - 1 words of the n-gram. This is the context,
                # which is the words leading up to the last word of the n-gram.
                context = ngram[:-1]
                self.context_counts[context] += 1

    # Return the probability of the specified n-gram under the N-gram model.
    # Conceptually, this means the probability of the last word of the n-gram
    # appearing given the context (i.e. the words leading up to the last word
    # of the n-gram). For example, if the n-gram is ("I", "love", "CSE"), then
    # we want to return P("CSE" | "I", "love").
    # The formula we use to calculate this is simply the number of times the
    # n-gram appears in the training data divided by the number of times the
    # context appears in the training data. This is the maximum likelihood
    # estimate (MLE) of the probability.
    def ngram_probability(
        self, ngram: Tuple[str], use_laplace_smoothing=False
    ) -> float:
        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[ngram[:-1]]

        # If we want to apply smoothing, we use the Laplace smoothing formula.
        # This means we add 1 to the numerator and the size of the vocabulary
        # to the denominator.
        if use_laplace_smoothing:
            # (Count + k) / (Context count + k * V) where 0 < k < 1
            return (ngram_count + self.smoothing_constant) / (
                context_count + self.smoothing_constant * len(self.vocabulary)
            )

        # If the context count is 0, then we have never seen this context in
        # the training data. This means that we have no information about what
        # word should come next, so we return 0.
        elif context_count == 0:
            return 0.0

        # Otherwise, we return the MLE of the probability.
        return ngram_count / context_count

    # Return the log probability of the specified sentence under the N-gram model.
    # Conceptually, this means the probability of our model predicting the
    # sentence. Note that this will be a negative number, since the probability
    # of a sentence is between 0 and 1, and the log of a number between 0 and 1
    # is negative.
    # Returns -infinity if the probability of the sentence being predicted by
    # the model is 0. This can happen if the sentence contains an n-gram that
    # has never appeared in the training data. We return -infinity because the
    # log of 0 is -infinity.
    def sentence_log_probability(
        self, processed_sentence: List[str], use_laplace_smoothing=False
    ) -> float:
        sentence_probability_log_sum = 0.0
        for ngram in self.ngrams(processed_sentence):
            probability = self.ngram_probability(ngram, use_laplace_smoothing)

            # If the probability is 0, then the sentence probability is 0. This
            # is because the log of 0 is undefined, so we can't add it to the
            # log sum. And since the probability is 0, it means that the ngram
            # has never appeared in the training data, so the sentence has 0
            # probability of being generated by the model. (Of course, smoothing
            # prevents this from happening, but we don't apply smoothing when
            # calculating the perplexity of the dev and test data.)
            if probability == 0.0:
                return float("-inf")

            sentence_probability_log_sum += math.log(probability)

        return sentence_probability_log_sum

    # Return the perplexity of the specified data under the N-gram model.
    def perplexity(self, data: List[str], use_laplace_smoothing=False) -> float:
        # Preprocess the data, except this time we want to UNKify the data
        # using the same words we UNKified the training data with. This is
        # because we want to evaluate the model on the same data distribution
        # that we trained it on. If we UNKify the dev and test data with
        # different words, then we are evaluating the model on a different
        # data distribution than we trained it on.
        processed_data = self.preprocess_data(data, is_training_data=False)

        # Calculate the log probability of each sentence in the data. Then,
        # sum these log probabilities together.
        log_probability_sum = 0.0
        # Count the number of tokens in the data. This will be used to calculate
        # the perplexity.
        token_count = 0
        for sentence in processed_data:
            log_probability = self.sentence_log_probability(
                sentence, use_laplace_smoothing
            )

            # If the log probability of seeing the sentence is -infinity, then
            # the perplexity is infinity. This is because perplexity is
            # e^(-sum of log probabilities for all sentences / token count).
            # So if the log probability is -infinity, then the perplexity will
            # e^(infinity / token count), which is infinity.
            if log_probability == float("-inf"):
                return float("inf")

            log_probability_sum += log_probability

            # Add the number of tokens in the sentence to the token count.
            # We subtract N - 1 from the length of the sentence because we
            # prepend N - 1 <START> tokens to each sentence. We don't want
            # to count these tokens when calculating the perplexity, since
            # they are not part of the original sentence and our model doesn't
            # predict them.
            token_count += len(sentence) - (self.n - 1)

        # Calculate the perplexity using the formula e^(-log probability / token count).
        return math.exp(-log_probability_sum / token_count)

    """
    INTERPOLATION FUNCTIONS (only works with trigram models)
    """

    # Return the log probability of the specified sentence under the N-gram model
    # using interpolation. NOTE: This function only works with trigram models.
    def sentence_log_probability_with_interpolation(
        self,
        processed_sentence: List[str],
        unigram_model: "NGramModel",
        bigram_model: "NGramModel",
    ) -> float:
        assert self.n == 3, "😳 We can only use interpolation with trigram models."
        assert self.lambdas, "😳 We need to specify lambda values to use interpolation."

        sentence_probability_log_sum = 0.0
        for ngram in self.ngrams(processed_sentence):
            # Calculate the probability of the n-gram under the trigram model.
            trigram_probability = self.ngram_probability(ngram)

            # Calculate the probability of the n-gram under the bigram model.
            bigram_probability = bigram_model.ngram_probability(ngram[-2:])

            # Calculate the probability of the n-gram under the unigram model.
            unigram_probability = unigram_model.ngram_probability(ngram[-1:])

            # Calculate the interpolated probability of the n-gram.
            interpolated_probability = (
                self.lambdas[0] * unigram_probability
                + self.lambdas[1] * bigram_probability
                + self.lambdas[2] * trigram_probability
            )

            if interpolated_probability == 0.0:
                return float("-inf")

            sentence_probability_log_sum += math.log(interpolated_probability)

        return sentence_probability_log_sum

    # Return the perplexity of the specified data under the N-gram model using
    # interpolation. NOTE: This function only works with trigram models.
    def perplexity_with_interpolation(
        self,
        data: List[str],
        unigram_model: "NGramModel",
        bigram_model: "NGramModel",
    ) -> float:
        assert self.n == 3, "😳 We can only use interpolation with trigram models."

        processed_data = self.preprocess_data(data, is_training_data=False)

        log_probability_sum = 0.0
        token_count = 0
        for sentence in processed_data:
            log_probability = self.sentence_log_probability_with_interpolation(
                sentence, unigram_model, bigram_model
            )

            if log_probability == float("-inf"):
                return float("inf")

            log_probability_sum += log_probability
            token_count += len(sentence) - (self.n - 1)

        return math.exp(-log_probability_sum / token_count)


"""
MODEL EVALUATION FUNCTIONS -- Used for evaluating the perplexity of the models
on the training, dev, and test data.
"""


# Evaluates the trigram model with interpolation on the given dataset. Prints
# the perplexity of the model on the dataset for all possible lambda values.
# NOTE: This function only works with trigram models.
def evaluate_trigram_model_with_interpolation(
    data: List[str], data_name: str, unigram_model: NGramModel, bigram_model: NGramModel
) -> None:
    print(
        f"Running grid search to find best lambda values for interpolation on {data_name} set..."
    )
    best_perplexity = float("inf")
    best_lambdas = None
    # Grid search over all possible lambda values
    for lambda_1 in [0.1, 0.3, 0.5, 0.9]:
        for lambda_2 in [0.1, 0.3, 0.5, 0.9]:
            # Round lambda_3 to 1 decimal place to avoid floating point errors
            lambda_3 = round(1 - lambda_1 - lambda_2, 1)

            if lambda_3 <= 0:
                continue

            print(f"Lambda values: {lambda_1}, {lambda_2}, {lambda_3}")

            model = NGramModel(3, lambdas=(lambda_1, lambda_2, lambda_3))
            model.train(training_data)
            perplexity = model.perplexity_with_interpolation(
                data, unigram_model, bigram_model
            )

            print(f"Perplexity: {perplexity}")
            print()

            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_lambdas = (lambda_1, lambda_2, lambda_3)

    print(f"Best lambdas for {data_name} set: {best_lambdas}")
    print(f"Best perplexity: {best_perplexity}")


# Evaluates the given n-gram model on the given dataset. Prints the perplexity
# of the model on the dataset without any smoothing, with Laplace smoothing,
# and with interpolation.
def evaluate_model(model: NGramModel, data: List[str], data_name: str) -> None:
    print(f"{model.n}-gram model perplexity on {data_name} set...")
    print(f"No smoothing:\t\t\t{model.perplexity(data)}")
    print(
        f"Laplace smoothing (k={model.smoothing_constant}):\t{model.perplexity(data, use_laplace_smoothing=True)}"
    )
    print()


if __name__ == "__main__":
    # Load the training, dev, and test data
    training_data = load_data("data/1b_benchmark.train.tokens")
    dev_data = load_data("data/1b_benchmark.dev.tokens")
    test_data = load_data("data/1b_benchmark.test.tokens")

    # Create an instance of the NGramModel for unigrams, bigrams, trigrams
    unigram_model = NGramModel(1)
    bigram_model = NGramModel(2)
    trigram_model = NGramModel(3)

    # Train the models on the training data
    unigram_model.train(training_data)
    bigram_model.train(training_data)
    trigram_model.train(training_data)

    # Evaluate the models on the training data
    evaluate_model(unigram_model, training_data, "training")
    evaluate_model(bigram_model, training_data, "training")
    evaluate_model(trigram_model, training_data, "training")

    # Evaluate the models on the dev data
    evaluate_model(unigram_model, dev_data, "dev")
    evaluate_model(bigram_model, dev_data, "dev")
    evaluate_model(trigram_model, dev_data, "dev")

    # Evaluate the models on the test data
    evaluate_model(unigram_model, test_data, "test")
    evaluate_model(bigram_model, test_data, "test")
    evaluate_model(trigram_model, test_data, "test")

    # Evaluate the trigram model with interpolation on the dev data
    evaluate_trigram_model_with_interpolation(
        dev_data, "dev", unigram_model, bigram_model
    )
