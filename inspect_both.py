import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_please(epoch=None, folder=None):
    df_true = pd.read_csv("problem.csv")
    df_pred = pd.read_csv("solution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    axes[0].scatter(df_true[df_true.line == 0].x, df_true[df_true.line == 0].y, color='red', s=10)
    axes[0].scatter(df_true[df_true.line == 1].x, df_true[df_true.line == 1].y, color='green', s=10)
    axes[0].set_title("Ground Truth")

    axes[1].scatter(df_pred[df_pred.line == 0].x, df_pred[df_pred.line == 0].y, color='red', s=10)
    axes[1].scatter(df_pred[df_pred.line == 1].x, df_pred[df_pred.line == 1].y, color='green', s=10)
    axes[1].set_title("Predicted")

    for ax in axes:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True)
        ax.set_aspect('equal')

    plt.tight_layout()
    if epoch is None:
        plt.show()
    else:
        if folder is None:
            folder = "dump"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/{epoch}.png")
        plt.close()


