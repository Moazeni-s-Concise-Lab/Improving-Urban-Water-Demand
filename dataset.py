####################################################################################################################
# % Code for the paper:
# % Improving urban water demand forecast using conformal prediction-based hybrid machine learning models
# % By Oluwabunmi Iwakin; Farrah Moazeni, PhD
# % Lehigh University, omi222@lehigh.edu, moazeni@lehigh.edu
####################################################################################################################

import torch
from torch.utils.data import Dataset

class DemandDataset(Dataset):
    def __init__(self, features, targets, device=None):
        self.X = features
        self.y = targets
        self.device = device

    def __getitem__(self, index):
        return (
            torch.tensor(self.X[index], device=self.device).float(),
            torch.tensor(self.y[index], device=self.device).float()
        )

    def __len__(self):
        return len(self.y)

def get_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, device="cpu"):
    trainset = DemandDataset(X_train, y_train, device=device)
    valset = DemandDataset(X_val, y_val, device=device)
    testset = DemandDataset(X_test, y_test, device=device)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False)

    return trainloader, valloader, testloader
