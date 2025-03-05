import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import tensorflow as tf
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LinearLR, StepLR
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from src.models.mlp import MLPClassifier
import numpy as np
from src.data.x_vector_pytorch.models.x_vector import X_vector
from src.models.vfd_resnet_model import resnet18
from src.models.custom_resnet_model import custom_resnet18
from src.models.se_resnet_model import custom_se_resnet18
from src.models.lcnn_model import LCNNModel
from src.training.loss_functions import vfd_loss_function, non_vfd_loss_function, vfd_loss_function_binary, non_vfd_loss_function_binary
from src.training.arguments import MODELS
from src.training.invariables import CLASSES, DEV, BINARY_CLASS_LABELS
import re


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)


    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print("torch.use_deterministic_algorithms is not available. Consider upgrading PyTorch for full determinism.")

def get_model(model, classification_type):
    # Set correct model based on passed arguments
    print(f'Set up the {model} model...')

    
    # Set model targets
    if "binary" in classification_type:
        num_classes = 1
    else:
        num_classes = int(re.findall(pattern=r'\d+', string=classification_type)[0])

    # Get model
    if (model == "resnet"):
        model = custom_resnet18(num_classes=num_classes)
    elif (model == "se-resnet"):
        model = custom_se_resnet18(num_classes=num_classes)
    elif (model == "lcnn"):
        model = LCNNModel(num_classes=num_classes)
    elif (model == "x-vector"):
        model = X_vector(input_dim=60, num_classes=num_classes)
    elif (model == "vfd-resnet"):
        model = resnet18(num_classes=num_classes)
    elif (model == "fingerprints"):
        model = MLPClassifier()
    return model


def get_optimizer_scheduler_loss_function(model, my_model, classification_type):
    if model == "vfd-resnet":
        lr = 1e-2
        optimizer = Adam(my_model.parameters(), lr=lr)
        #scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        scheduler = LinearLR(optimizer=optimizer, total_iters=100)
        loss_function =  vfd_loss_function
        if "binary" in classification_type:
            loss_function =  vfd_loss_function_binary

    elif model == "fingerprints":

        optimizer = Adam(my_model.parameters(), lr=0.001)
        scheduler = LinearLR(optimizer=optimizer, total_iters=100)
        loss_function = torch.nn.BCEWithLogitsLoss().to(DEV)

    elif model in MODELS:
        optimizer = Adam(my_model.parameters(), lr=0.001, betas=(0.9, 0.98))
        scheduler = LinearLR(optimizer=optimizer, total_iters=100)
        loss_function =  non_vfd_loss_function
        if "binary" in classification_type:
            loss_function =  non_vfd_loss_function_binary
    else:
        raise ValueError(f'Function \"get_optimizer_scheduler_loss_function\" in {__file__}: Model not specified or Wrong model Specified. Please provide a valid model option!')
    
    return optimizer, scheduler, loss_function

def get_metric(performance_metric):
    if performance_metric == "accuracy":
        metric = "Validating_Accuracy"
    elif performance_metric == "f1_score":
        metric = "Validating_F1_Score"
    elif performance_metric == "precision":
        metric = "Validating_Precision"
    elif performance_metric == "recall":
        metric = "Validating_Recall"
    elif performance_metric == "auroc":
        metric = "Validating_AUROC"
    return metric


def save_confusion_matrix_to_excel(conf_matrix, destination_url, classification_type):

    conf_matrix = conf_matrix.cpu().numpy()
    if "binary" in classification_type:
        labels = BINARY_CLASS_LABELS
    else:
        labels = CLASSES[classification_type]

    conf_matrix_df = pd.DataFrame(conf_matrix, columns=[f"Pred_{i}" for i in labels],
                    index=[f"True_{i}" for i in labels])
    conf_matrix_df.to_excel(f'{destination_url}/testing_confusion_matrix.xlsx', index=True)
    print("Confusion matrix saved...")    


def save_heatmap(conf_matrix, destination_url, classification_type):

    if "binary" in classification_type:
        labels = BINARY_CLASS_LABELS
    else:
        labels = CLASSES[classification_type]

    plt.figure(figsize=(8, 6))

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[f"Pred {i}" for i in labels],
                yticklabels=[f"True {i}" for i in labels])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(f'{destination_url}/testing_heatmap.png', bbox_inches="tight")
    print("Heatmaps of Confusion matrices saved...")