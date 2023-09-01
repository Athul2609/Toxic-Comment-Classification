import torch
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
import warnings
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import os


def make_prediction(model, loader, output_csv="submission.csv"):
    preds = []
    ids = []
    model.eval()

    for x, y, id in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            predictions = model(x)
            # Convert MSE floats to integer predictions
            # predictions = (predictions > 0.5).float()
            predictions = predictions.long().squeeze(1)
            preds.append(predictions.cpu().numpy())
            ids.append(id)

    column_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Create a DataFrame
    df = pd.DataFrame(preds, columns=column_names)
    df.insert(0, 'id', ids)
    df.to_csv(output_csv, index=False)
    model.train()
    print("Done with predictions")

def check_accuracy_and_save(loader, model, device="cuda", save_dir="./"):
    model.eval()
    all_preds, all_labels = [], []

    for x, y, id in tqdm(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            predictions = model(x)
        predictions = torch.sigmoid(predictions)
        print(predictions)
        predictions = (predictions > 0.5).long()
        predictions = predictions.long()
        y = y.long()
        # Add predictions and labels to lists
        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    # Concatenate the predictions and labels
    all_preds = np.concatenate(all_preds, axis=0, dtype=np.int64)
    all_labels = np.concatenate(all_labels, axis=0, dtype=np.int64)
    print(all_preds.shape)
    print(all_labels.shape)

    # Calculate accuracy using sklearn's accuracy_score
    accuracy = accuracy_score(all_labels, all_preds)
    acc_values = [accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(all_preds.shape[1])]

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Coloumn wise accuracy: {acc_values}")
    model.train()

    # Save predictions and labels as numpy files
    np.save(os.path.join(save_dir, "all_preds.npy"), all_preds)
    np.save(os.path.join(save_dir, "all_labels.npy"), all_labels)
    print(f"Predictions and labels saved to {save_dir}")

    return all_preds, all_labels

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr