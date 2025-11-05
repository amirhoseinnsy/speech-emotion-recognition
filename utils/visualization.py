import matplotlib.pyplot as plt
import re

def plot_losses(log_lines):
    train_losses = []
    val_losses = []

    loss_pattern = re.compile(r"Train Loss: ([0-9.]+) .* Val Loss: ([0-9.]+)")

    for line in log_lines:
        match = loss_pattern.search(line)
        if match:
            train_losses.append(float(match.group(1)))
            val_losses.append(float(match.group(2)))

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
