import matplotlib.pyplot as plt

file_name = "training_validation_accuracies.png"

training_accuracies = [
    0.8632947976878613,
    0.9320809248554913,
    0.945664739884393,
    0.9559248554913294,
    0.9635838150289018,
    0.9645953757225434,
    0.9708092485549132,
    0.974421965317919,
    0.9773121387283237,
    0.9799132947976879,
]

validation_accuracies = [
    0.9151376146788991,
    0.8853211009174312,
    0.9243119266055045,
    0.9243119266055045,
    0.930045871559633,
    0.926605504587156,
    0.9277522935779816,
    0.9277522935779816,
    0.930045871559633,
    0.9334862385321101,
]


def plot(training_accuracies, validation_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(training_accuracies) + 1),
        training_accuracies,
        linestyle="-",
        linewidth=1,
        color="blue",
        alpha=0.5,
        label="Training Accuracy",
    )
    plt.plot(
        range(1, len(validation_accuracies) + 1),
        validation_accuracies,
        linestyle="-",
        linewidth=1,
        color="red",
        alpha=0.5,
        label="Validation Accuracy",
    )
    plt.title("Training and Validation Accuracies", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig(file_name)


plot(training_accuracies, validation_accuracies)
