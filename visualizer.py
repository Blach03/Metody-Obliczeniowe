import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_prefix):
    x = np.loadtxt(f"{file_prefix}_x.csv", delimiter=",")
    u_true = np.loadtxt(f"{file_prefix}_u_true.csv", delimiter=",")
    u_pred = np.loadtxt(f"{file_prefix}_u_pred.csv", delimiter=",")
    loss_history = np.loadtxt(f"{file_prefix}_loss_history.csv", delimiter=",")
    return x, u_true, u_pred, loss_history

def plot_results(x, u_true, u_pred, title):
    plt.figure()
    plt.plot(x, u_true, label="Exact Solution")
    plt.plot(x, u_pred, label="Predicted Solution", linestyle='--')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_error(x, u_true, u_pred, title):
    plt.figure()
    error = np.abs(u_true - u_pred)
    plt.plot(x, error, label="Error")
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_loss(loss_history, title):
    epochs = np.arange(0, len(loss_history) * 10, 10)
    plt.figure()
    plt.plot(epochs, loss_history, label="Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_all_errors(data_files1, data_files2):
    plt.figure()
    for file_prefix in data_files1:
        x, u_true, u_pred, _ = load_data(file_prefix)
        error = np.abs(u_true - u_pred)
        label = file_prefix.replace("results_", "").replace("_", " ").title()
        plt.plot(x, error, label=label)
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title('Comparison of Errors for omega = 1')
    plt.legend()
    plt.show()

    plt.figure()
    for file_prefix in data_files2:
        x, u_true, u_pred, _ = load_data(file_prefix)
        error = np.abs(u_true - u_pred)
        label = file_prefix.replace("results_", "").replace("_", " ").title()
        plt.plot(x, error, label=label)
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title('Comparison of Errors for omega = 15')
    plt.legend()
    plt.show()

def plot_all_loss(data_files1, data_files2):
    plt.figure()
    for file_prefix in data_files1:
        _, _, _, loss_history = load_data(file_prefix)
        min_loss = [min(loss_history[i:i+10]) for i in range(0, len(loss_history), 10)]
        epochs = np.arange(0, len(min_loss) * 100, 100)
        label = file_prefix.replace("results_", "").replace("_", " ").title()
        plt.plot(epochs, min_loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Comparison of Losses for omega = 1')
    plt.legend()
    plt.show()

    plt.figure()
    for file_prefix in data_files2:
        _, _, _, loss_history = load_data(file_prefix)
        min_loss = [min(loss_history[i:i+10]) for i in range(0, len(loss_history), 10)]
        epochs = np.arange(0, len(min_loss) * 100, 100)
        label = file_prefix.replace("results_", "").replace("_", " ").title()
        plt.plot(epochs, min_loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Comparison of Losses for omega = 15')
    plt.legend()
    plt.show()

data_files1 = [
    "results_omega_1_layers_2_neurons_16",
    "results_omega_1_layers_4_neurons_64",
    "results_omega_1_layers_5_neurons_128"
]

data_files2 = [
    "results_omega_15_layers_2_neurons_16",
    "results_omega_15_layers_4_neurons_64",
    "results_omega_15_layers_5_neurons_128"
]


for file_prefix in data_files1:
    x, u_true, u_pred, loss_history = load_data(file_prefix)
    plot_results(x, u_true, u_pred, f"Solution: {file_prefix.replace('_', ' ').title()}")
    plot_error(x, u_true, u_pred, f"Error: {file_prefix.replace('_', ' ').title()}")
    plot_loss(loss_history, f"Loss: {file_prefix.replace('_', ' ').title()}")

for file_prefix in data_files2:
    x, u_true, u_pred, loss_history = load_data(file_prefix)
    plot_results(x, u_true, u_pred, f"Solution: {file_prefix.replace('_', ' ').title()}")
    plot_error(x, u_true, u_pred, f"Error: {file_prefix.replace('_', ' ').title()}")
    plot_loss(loss_history, f"Loss: {file_prefix.replace('_', ' ').title()}")


plot_all_errors(data_files1, data_files2)


plot_all_loss(data_files1, data_files2)