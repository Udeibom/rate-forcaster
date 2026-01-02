import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")


def plot_actual_vs_predicted(y_true, y_pred, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Exchange Rate")
    plt.xlabel("Time")
    plt.ylabel("USD/NGN")
    plt.tight_layout()
    plt.savefig(save_dir / "actual_vs_predicted.png")
    plt.close()


def plot_error_over_time(y_true, y_pred, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    errors = y_true - y_pred

    plt.figure(figsize=(10, 4))
    plt.plot(errors)
    plt.axhline(0)
    plt.title("Prediction Error Over Time")
    plt.xlabel("Time")
    plt.ylabel("Error (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(save_dir / "error_over_time.png")
    plt.close()
