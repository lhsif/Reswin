import itertools
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
from torch.utils.data import DataLoader

from config import config
from dataset import NiiDataset, split_dataset
from model import swin_transformer_3d, resnet_3d_50, csn_r101, slow_r50, r2plus1d_r50, x3d, slowfast_r50,mvit


def ensemble_predictions(model1, model2, data_loader):
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    all_labels = []
    all_probs1 = []
    all_probs2 = []

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            logits1 = model1(data)
            logits2 = model2(data)

            probabilities1 = torch.softmax(logits1, dim=1)[:, 1].cpu().numpy()
            probabilities2 = torch.softmax(logits2, dim=1)[:, 1].cpu().numpy()

            all_probs1.extend(probabilities1.tolist())
            all_probs2.extend(probabilities2.tolist())
            all_labels.extend(labels.tolist())

    return all_labels, all_probs1, all_probs2


def plot_roc(labels, preds, model_name):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f"./data/modelckpt/{model_name}_auc_plot.png")
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def compute_aupr(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)
    return aupr


# Load your trained models
model1 = swin_transformer_3d()
model2 = resnet_3d_50()  # Replace this with the actual ResNet model

# Load model weights after training
model1.load_state_dict(torch.load("./data/modelckpt/swin32/max_f1_score_weight.pt"))
model2.load_state_dict(torch.load("./data/modelckpt/resnet32/max_f1_score_weight.pt"))

folders = ["CT-0", "CT-1"]
_, val_folders = split_dataset(folders, train_ratio=config["train_ratio"], seed=config["seed"],
                                data_root="./data")
print("Validation folders:", val_folders)
val_dataset = NiiDataset(val_folders, task=config["task"])

val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_labels, all_probs1, all_probs2 = ensemble_predictions(model1, model2, val_loader)

# Find the best weights for the ensemble
best_auc = 0
best_f1 = 0
best_weights = (0.5, 0.5)

for w1 in np.arange(0, 1.01, 0.01):
    w2 = 1 - w1
    weights = (w1, w2)

    ensemble_probabilities = (weights[0] * np.array(all_probs1)) + (weights[1] * np.array(all_probs2))
    auc_score = roc_auc_score(all_labels, ensemble_probabilities)
    f1_score_val = f1_score(all_labels, [1 if p > 0.5 else 0 for p in ensemble_probabilities])

    if auc_score > best_auc and f1_score_val > best_f1:
        best_auc = auc_score
        best_f1 = f1_score_val
        best_weights = weights

print(f"Best AUC: {best_auc:.4f}, Best F1: {best_f1:.4f}, Best Weights: {best_weights}")

# Calculate ensemble probabilities with the best weights
ensemble_probabilities = (best_weights[0] * np.array(all_probs1)) + (best_weights[1] * np.array(all_probs2))

# Plot ROC curve
plot_roc(all_labels, ensemble_probabilities, "ensemble")

# Compute confusion matrix
cnf_matrix = confusion_matrix(all_labels, [1 if p > 0.5 else 0 for p in ensemble_probabilities])

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['No-Covid19', 'Covid19'], title='Confusion Matrix')
plt.show()

# Print the performance metrics
accuracy = (cnf_matrix[0, 0] + cnf_matrix[1, 1]) / np.sum(cnf_matrix)
precision = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[0, 1])
recall = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0])
aupr = compute_aupr(all_labels, ensemble_probabilities)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {best_f1:.4f}")
print(f"AUPR: {aupr:.4f}")