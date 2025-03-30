####################################################################################################################
# % Code for the paper:
# % Improving urban water demand forecast using conformal prediction-based hybrid machine learning models
# % By Oluwabunmi Iwakin; Farrah Moazeni, PhD
# % Lehigh University, omi222@lehigh.edu, moazeni@lehigh.edu
####################################################################################################################


import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=32, output=3, nlayers=1, drop=0.25):
        super(CNNLSTM, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 32, kernel_size=1, padding="same"),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(32, 32, kernel_size=1),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        self.bilstm = nn.LSTM(160, hidden_size=hidden_size, num_layers=nlayers, bidirectional=True)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * hidden_size, output)
        )

    def forward(self, x):
        x = self.conv_layer(x.unsqueeze(1))
        x = torch.flatten(x, 1)
        x, _ = self.bilstm(x)
        return self.head(x)


# def reliability(y_true, y_lower, y_upper):
#     return (y_true >= y_lower) & (y_true <= y_upper).float().mean()

def reliability(y, y_low, y_high):
    return torch.mean(((y >= y_low) & (y <= y_high)).type(torch.float64) )


def conformity_loss(qpred, y_true, alpha=0.1):
    quants = torch.tensor([0.05, 0.5, 0.95], device=qpred.device).float()
    conf_q = min(np.ceil((len(y_true)+1)*(1-alpha))/len(y_true), 0.9)
    err = y_true - qpred

    calib_scores = torch.max(qpred[:, 0] - y_true[:, 0], y_true[:, 0] - qpred[:, -1])
    calib = torch.quantile(calib_scores, conf_q, interpolation="higher")

    new_qpred = qpred.clone()
    new_qpred[:, 0] -= calib
    new_qpred[:, -1] += calib

    err = y_true - new_qpred
    loss = torch.max(quants * err, (quants - 1) * err).sum(dim=1).mean()
    factor = reliability(y_true, qpred[:, 2:3], qpred[:, 0:1])

    return loss / (factor + alpha/2), calib_scores


def train(model, trainloader, valloader, lr, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = conformity_loss
    best_loss = float("inf")
    best_case = []

    training_loss = []
    eval_loss = []

    print(f"Training {model.__class__.__name__} with Adam Optimizer...")
    start = time.time()

    for epoch in tqdm(range(epochs)):
        model.train()
        tr_loss = 0
        calib_case = torch.tensor([], device=device)

        for X_train, y_train in trainloader:
            y_pred = model(X_train)
            optimizer.zero_grad()
            loss, scores = loss_func(y_pred, y_train.unsqueeze(1))
            tr_loss += loss
            loss.backward()
            optimizer.step()

        training_loss.append(tr_loss.item() / len(trainloader))

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for X_val, y_val in valloader:
                val_pred = model(X_val)
                val_loss, scores = loss_func(val_pred, y_val.unsqueeze(1))
                calib_case = torch.hstack([calib_case, scores])
                val_loss /= len(valloader)
            eval_loss.append(val_loss.item())

        if val_loss < best_loss:
            best_loss = val_loss
            best_case = calib_case
            torch.save(model.state_dict(), 'bestcoveragemodel2.pt')

    print(f"\nTotal Training Time: {(time.time() - start)/60:.2f} mins.")
    return training_loss, eval_loss, best_case


import torch

def conformal_predict(model, dataloader, device, loss_func):
    """
    Generate predictions with conformal calibration scores.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): PyTorch dataloader for validation/test set.
        device (torch.device): Device to perform computation on.
        loss_func (function): Conformity loss function used during training.

    Returns:
        Tuple[List[Tensor], Tensor]: 
            - predictions from the model (shape: [N, 3] for quantiles)
            - calibration scores from the conformity loss function
    """
    model.eval()
    predictions = []
    calib_case = torch.tensor([], device=device)

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu())  # move to CPU for output
            _, scores = loss_func(y_pred, y_batch.unsqueeze(1))
            calib_case = torch.hstack([calib_case, scores])

    predictions = torch.cat(predictions, dim=0)  # full batch predictions
    return predictions, calib_case
