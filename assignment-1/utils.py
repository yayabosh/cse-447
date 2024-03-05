import csv
from typing import List

import matplotlib.pyplot as plt


def load_data(file_path: str) -> List[str]:
    """
    Loads the data from the specified file path and returns a list of lines from
    the file, stripped of whitespace.
    """

    # Read the file and split it into lines
    with open(file_path) as f:
        return [line.strip() for line in f]


def save_to_csv(
    vocabulary_sizes: List[int],
    corpus_lengths: List[int],
    filename: str = "bpe_training_data.csv",
) -> None:
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Vocabulary Size", "Corpus Length"])
        for vocab_size, corpus_length in zip(vocabulary_sizes, corpus_lengths):
            writer.writerow([vocab_size, corpus_length])


def produce_scatterplot(
    vocabulary_sizes: List[int],
    corpus_lengths: List[int],
    title: str,
    x_label: str,
    y_label: str,
    file_name: str,
) -> None:
    # Call this function after your training loop, using the lists you populated
    save_to_csv(vocabulary_sizes, corpus_lengths)

    # Plot the scatterplot after training
    plt.figure(figsize=(10, 6))
    # Plot the lines first to ensure they are in the background
    plt.plot(
        vocabulary_sizes,
        corpus_lengths,
        linestyle="-",
        linewidth=1,
        color="grey",
        alpha=0.5,
    )

    # Then plot the scatter points on top
    plt.scatter(vocabulary_sizes, corpus_lengths)
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True)  # Show gridlines
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Highlight the starting and ending points
    plt.scatter(
        vocabulary_sizes[0],
        corpus_lengths[0],
        color="green",
        s=100,
        label="Start",
        zorder=5,
    )
    plt.scatter(
        vocabulary_sizes[-1],
        corpus_lengths[-1],
        color="red",
        s=100,
        label="End",
        zorder=5,
    )
    plt.legend(fontsize=12)  # Add a legend with large font

    plt.tight_layout()  # Adjust the padding between and around subplots
    plt.savefig(file_name)
