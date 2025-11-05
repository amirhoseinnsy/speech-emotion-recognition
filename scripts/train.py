import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
import logging
from datetime import datetime
import yaml
import random
import numpy as np
import os

from utils.metrics import metrics
log_data = {"logs": []}

def log_message(message, log_file="config/logging_GPT.yaml"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    log_data["logs"].append(log_entry)
    logging.info(message)
    with open("debug_log.txt", "a") as dbg:
        dbg.write(log_entry + "\n")
    with open(log_file, 'a') as f:
        yaml.dump({"logs": [log_entry]}, f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(
    model,
    train_data,
    val_data,
    device,
    optimizer,
    epochs,
    save_epoch=1,
    patience=5,
    log_file="config/log.yaml"
):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    train_acc = metrics()
    loss_avg_train = metrics()
    val_acc = metrics()
    loss_avg_val = metrics()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    start = datetime.now()
    with trange(1, epochs + 1, desc="Training", unit="epoch") as epoch_bar:
        for epoch in epoch_bar:
            model.train()
            train_acc.reset()
            loss_avg_train.reset()

            for feats, labels in tqdm(train_data, desc=f"Train {epoch}", leave=False):
                feats, labels = feats.to(device), labels.to(device)
                optimizer.zero_grad()

                logits = model(feats)  # [batch, num_classes]
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == labels).float().mean()

                train_acc.step(acc.item(), feats.size(0))
                loss_avg_train.step(loss.item(), feats.size(0))

            model.eval()
            val_acc.reset()
            loss_avg_val.reset()
            with torch.inference_mode():
                for feats, labels in tqdm(val_data, desc=f"Val {epoch}", leave=False):
                    feats, labels = feats.to(device), labels.to(device)

                    logits = model(feats)
                    loss = loss_fn(logits, labels)

                    preds = logits.argmax(dim=-1)
                    acc = (preds == labels).float().mean()

                    val_acc.step(acc.item(), feats.size(0))
                    loss_avg_val.step(loss.item(), feats.size(0))

            msg = (
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {loss_avg_train.average:.4f} | Train Acc: {train_acc.average:.4f} | "
                f"Val Loss: {loss_avg_val.average:.4f} | Val Acc: {val_acc.average:.4f}"
            )
            log_message(msg, log_file=log_file)
            epoch_bar.set_postfix({
                "train_loss": f"{loss_avg_train.average:.4f}",
                "train_acc":  f"{train_acc.average:.4f}",
                "val_loss":   f"{loss_avg_val.average:.4f}",
                "val_acc":    f"{val_acc.average:.4f}",
            })

            if loss_avg_val.average < best_val_loss:
                best_val_loss = loss_avg_val.average
                epochs_no_improve = 0
                if epoch % save_epoch == 0:
                    os.makedirs("models/saved_models", exist_ok=True)
                    torch.save(model.state_dict(), f"models/saved_models/model_epoch_{epoch}.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log_message("Early stopping triggered.", log_file=log_file)
                    break

    os.makedirs("models/saved_models", exist_ok=True)
    torch.save(model.state_dict(), "models/saved_models/Final.pth")
    log_message(f"Training complete. Total time: {datetime.now() - start}", log_file=log_file)
    return model

