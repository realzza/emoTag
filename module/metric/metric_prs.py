import torch
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score, recall_score

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def uar(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        uar = recall_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), labels=[0, 1, 2, 3, 4], average='macro', zero_division=1)
    return uar    
    
def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def auc(output, target):
    with torch.no_grad():
        pred = torch.softmax(output, dim=1)
        assert pred.shape[0] == len(target)
    target = target.cpu().detach().numpy()
    output = np.max(output.cpu().detach().numpy(), axis=1)
    return roc_auc_score(target, output)