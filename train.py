import wandb
import torch
from sklearn.preprocessing import MinMaxScaler

from dataset import TrainDataset, extract_data
from model import HybridLSTMGRU
import evaluation as eval
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim


def train_config(dataset, config, log_count=50, verbose=False):
    eval.set_seed(seed=config["seed"])

    if config["scaler"] == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = None
    train_data, val_data, _ = extract_data(
        dataset,
        config["pays"],
        config["prop"],
        config["type"],
        saison=config["saison"],
        ville=config["ville"],
        scaler=scaler,
    )
    dataset_train = TrainDataset(
        train_data, seq_len=config["seq_len"], shift=config["shift"]
    )
    train_dataloader = DataLoader(
        dataset_train, batch_size=config["batch_size"], shuffle=True
    )

    model = HybridLSTMGRU().to(config["device"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_results = {
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "seq_len": config["seq_len"],
        "epoch": 0,
        "val_rmse": float("inf"),
        "val_mae": None,
        "val_mape": None,
        "train_rmse": None,
        "train_mae": None,
        "train_mape": None,
    }

    for epoch in range(config["epochs"]):
        model.train()
        for X, y in train_dataloader:
            X = X.unsqueeze(-1).to(config["device"])
            y = y.unsqueeze(-1).to(config["device"])

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        if (epoch - 1) % log_count == 0 or epoch == config["epochs"] - 1:
            model.eval()
            with torch.no_grad():
                train_rmse, train_mae, train_mape = eval.evaluate(
                    model, train_data, None, config["seq_len"], scaler
                )
                val_rmse, val_mae, val_mape = eval.evaluate(
                    model, val_data, train_data, config["seq_len"], scaler
                )
                if verbose:
                    print(f"Epoch {epoch}/{config['epochs']}")
                    print(
                        f"Training - RMSE: {train_rmse:.4f},  MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}"
                    )
                    print(
                        f"Validation - RMSE: {val_rmse:.4f},  MAE: {val_mae:.4f}, MAPE: {val_mape:.4f}"
                    )

                if val_rmse < best_results["val_rmse"]:
                    best_results["epoch"] = epoch
                    best_results["val_rmse"] = val_rmse
                    best_results["val_mae"] = val_mae
                    best_results["val_mape"] = val_mape
                    best_results["train_rmse"] = train_rmse
                    best_results["train_mae"] = train_mae
                    best_results["train_mape"] = train_mape

    return model, best_results
