import json
import sys
import matplotlib.pyplot as plt
import numpy as np

def load_log(file_path):
    # get training and validation loss data from json
    with open(file_path, 'r', encoding='utf8') as f:
        return np.asarray(json.load(f))

def plot_loss(trn_loss, val_loss):
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12

    epochs = np.arange(len(val_loss))

    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.plot(epochs, trn_loss, label='Training Loss', color='blue')

    plt.grid(which='both', color='gray', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(edgecolor='white')
    plt.title('Training and Validation Loss Over Epochs')
    plt.tight_layout()
    plt.show()

def main(file_path):
    log_data = load_log(file_path)
    print("Minimum loss values (training, validation):", np.min(log_data, axis=0))

    trn_loss = log_data[:, 0]
    val_loss = log_data[:, 1]

    plot_loss(trn_loss, val_loss)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot_loss.py <log_file.json>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    main(log_file_path)
