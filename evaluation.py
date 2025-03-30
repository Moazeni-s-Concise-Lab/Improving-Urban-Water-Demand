import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import torch

def evaluate_model(model, dataloader, conf, device):
    model.eval()
    preds, truevals = [], []
    upperpreds, lowerpreds = [], []
    pred_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).cpu().squeeze()
            preds.append(pred[:, 1])
            truevals.append(y.cpu())
            pred_loss += torch.mean((y.cpu() - pred[:, 1]) ** 2) / len(dataloader)
            upperpreds.append(pred[:, 2] + conf)
            lowerpreds.append(pred[:, 0] - conf)

    preds = np.concatenate(preds)
    truevals = np.concatenate(truevals)
    upperpreds = np.concatenate(upperpreds)
    lowerpreds = np.concatenate(lowerpreds)

    metrics = {
        "MSE": mean_squared_error(truevals, preds),
        "RMSE": np.sqrt(mean_squared_error(truevals, preds)),
        "R2": r2_score(truevals, preds),
        "NRMSE": np.sqrt(mean_squared_error(truevals, preds)) / np.mean(truevals),
        "MAPE": mean_absolute_percentage_error(truevals, preds),
        'Coverage': reliability(truevals, lowerpreds, upperpreds)
    }
    

    print(f"\nFinal R² Score: {metrics['R2']:.4f}")
    print(f"Prediction Interval Coverage (Reliability): {metrics['Coverage']:.4f}\n")

    return preds, truevals, upperpreds, lowerpreds, pred_loss, metrics





def reliability(y_true, y_lower, y_upper):
    return ((y_true >= y_lower) & (y_true <= y_upper)).mean()

def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def export_predictions(preds, upper, lower, truevals, filename):
    df = pd.DataFrame({
        "TrueDemand": truevals,
        "0.05 Quantile": upper,
        "0.5 Quantile": preds,
        "0.95 Quantile": lower,
    })
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
