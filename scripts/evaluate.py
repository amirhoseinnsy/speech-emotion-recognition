import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for feats, labels in dataloader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            preds = torch.argmax(F.softmax(logits, dim=-1), dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print("Classification Report:")
    for i, cls in enumerate(class_names):
        print(f"{cls:>7} → Precision: {report[cls]['precision']:.3f}, Recall: {report[cls]['recall']:.3f}, F1: {report[cls]['f1-score']:.3f}")

    print(f"\nOverall Accuracy: {report['accuracy']:.3f}")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return report, cm
