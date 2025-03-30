import torch
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import load_and_preprocess_data
from feature_engineering import create_features
from dataset import get_data_loaders
from models import CNNLSTM, train, conformal_predict, conformity_loss
from evaluation import evaluate_model, plot_loss, export_predictions

if __name__ == "__main__":
    csv_path = "../data/TelfordDemandData.csv"
    window = 12
    batch_size = 32
    lr = 5e-4
    epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    seed = 1932
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load and preprocess
    df = load_and_preprocess_data(csv_path)
    X, y = create_features(df, window)

    # Train/Val/Test Split
    # idx = np.random.permutation(len(X))
    # val_split = int(len(X) * 0.8)
    # X_train, X_val = [X[i] for i in idx[:val_split]], [X[i] for i in idx[val_split:]]
    # y_train, y_val = [y[i] for i in idx[:val_split]], [y[i] for i in idx[val_split:]]

    # test_split = int(len(X) * 0.8)
    # X_test, y_test = X[test_split:], y[test_split:]
    
    distro = range(len(X)) 
    train_idx = int((len(X)) *0.8)

    X_train = [X[i] for i in distro[:train_idx]]
    X_test = [X[i] for i in distro[train_idx:]]
    y_train = [y[i] for i in distro[:train_idx]]
    y_test = [y[i] for i in distro[train_idx:]]

    # Shuffle Xtrain and val windows
    shuffled_idx = np.random.permutation(range(len(X_train)))
    indexer = lambda searchList, idx: [searchList[i] for i in idx]
    val_idx = int(len(X_train)* 0.8)
    X_val = indexer(X_train, shuffled_idx[val_idx:])
    y_val = indexer(y_train, shuffled_idx[val_idx:])

    X_train = indexer(X_train, shuffled_idx[:val_idx])
    y_train = indexer(y_train, shuffled_idx[:val_idx])

    trainloader, valloader, testloader = get_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size, device
    )

    # Model
    model = CNNLSTM(hidden_size=32, output=3, drop=0.0).to(device)
    train_loss, val_loss, calib_case = train(model, trainloader, valloader, lr, epochs, device)
    
    # model.load_state_dict(torch.load('bestcoveragemodel2.pt'))
    # _, calib_case = conformal_predict(model, trainloader, device, conformity_loss)
    alpha = 0.1
    conf_q = np.minimum(np.ceil((len(calib_case)+1)*(1-alpha))/len(calib_case), 1-alpha)
    conf = torch.quantile(calib_case, conf_q, interpolation="higher")

    # Evaluate
    preds, truevals, upper, lower, _, metrics = evaluate_model(model, testloader, conf, device)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    plot_loss(train_loss, val_loss)

    filename = f"{metrics['NRMSE']:.2f}NRMSE-{metrics['MAPE']:.2f}MAPE_Prediction.csv"
    export_predictions(preds, upper, lower, truevals, filename)
