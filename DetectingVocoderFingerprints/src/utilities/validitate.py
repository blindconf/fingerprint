import re
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import device, no_grad
from tqdm import tqdm
from tabulate import tabulate
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix, AUROC
from torch.cuda import is_available
from src.datasets.utility import get_datasets, COLLATE_FN, BATCH_SIZE
from src.training.utility import get_model, save_confusion_matrix_to_excel, save_heatmap
from torch.nn import DataParallel
from torch import Generator
from torch.utils.data import DataLoader


DEV = device("cuda" if is_available() else "cpu")
DEVICE_IDS = [0, 1, 2]

def test_model(model, test_loader, classification_type, url_dir_to_save_model):
    """
    Test the model on the test dataset, compute evaluation metrics, and save the confusion matrix and heatmap.
    """
    print("\nTesting the best model...")

    # Load best model
    model.load_state_dict(torch.load(f'{url_dir_to_save_model}/best_model.pth', map_location="cuda"), strict=False)
    model.to("cuda")
    model.eval()

    # Define metrics
    '''
    if "binary" in classification_type:
        task = "binary"
        num_classes = None
        prob_func = torch.sigmoid  # Sigmoid for probability
        preds_func = lambda signals: (signals > 0.5).float()
    else:
        task = "multiclass"
        num_classes = int(re.findall(pattern=r'\d+', string=classification_type)[0])
        prob_func = lambda signals: torch.nn.functional.softmax(signals, dim=1)
        preds_func = lambda signals: torch.argmax(signals, dim=1)
        '''
    # Initialize metrics
    prob_func = torch.nn.functional.sigmoid
    preds_func = lambda signals: (signals > 0.5).float()
    task = "binary"
    accuracy = Accuracy(task=task).to("cuda")
    f1 = F1Score(task=task).to("cuda")
    precision = Precision(task=task).to("cuda")
    recall = Recall(task=task).to("cuda")
    confusion_matrix = ConfusionMatrix(task=task).to("cuda")
    auroc = AUROC(task=task).to("cuda")

    # Reset metrics
    accuracy.reset(), f1.reset(), precision.reset()
    recall.reset(), confusion_matrix.reset(), auroc.reset()

    # Variables for debugging
    all_logits = []
    all_probabilities = []
    all_preds = []
    all_labels = []

    with no_grad():
        for waveforms, labels in tqdm(test_loader, desc="Testing batches"):
            waveforms, labels = waveforms.to("cuda"), labels.to("cuda")
            labels = labels.float().unsqueeze(1)  # Ensure correct shape for binary classification
            
            # Forward pass
            outputs, _ = model(waveforms)  # Model outputs logits
            probabilities = prob_func(outputs)  # Convert logits to probabilities
            preds = preds_func(probabilities)  # Convert probabilities to class labels (0 or 1)

            # Debugging: Log outputs, probabilities, and predictions
            all_logits.extend(outputs.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update Metrics
            accuracy.update(preds, labels)
            f1.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            confusion_matrix.update(preds, labels)
            auroc.update(probabilities, labels)

    # Compute metrics
    testing_accuracy = accuracy.compute().item()
    testing_f1_score = f1.compute().item()
    testing_precision = precision.compute().item()
    testing_recall = recall.compute().item()
    testing_confusion_matrix = confusion_matrix.compute()
    testing_auroc = auroc.compute().item()

    # Debugging: Print distribution of logits, probabilities, and predictions
    print("\n--- Debugging Information ---")
    print(f"Logits Distribution: min={min(all_logits)}, max={max(all_logits)}, mean={sum(all_logits)/len(all_logits)}")
    print(f"Probabilities Distribution: min={min(all_probabilities)}, max={max(all_probabilities)}, mean={sum(all_probabilities)/len(all_probabilities)}")
    unique_preds = set([int(pred[0]) for pred in all_preds])  # Convert arrays to scalars
    print(f"Unique Predictions: {unique_preds}")

    unique_labels = set([int(label[0]) for label in all_labels])  # Convert arrays to scalars
    print(f"Unique True Labels: {unique_labels}")
    print("------------------------------\n")
    
    # Store metrics in a dictionary
    testing_scores_dict = {
        "Testing_Accuracy": testing_accuracy,
        "Testing_F1_Score": testing_f1_score,
        "Testing_Precision": testing_precision,
        "Testing_Recall": testing_recall,
        "Testing_AUROC": testing_auroc,
    }

    # Print test metrics
    print("\n")
    table = [[key, value] for key, value in testing_scores_dict.items()]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
    print("\n")
    # Save confusion matrix and heatmap
    url_dir_to_save_model = '/home/hessos4l/Downloads/DetectingVocoderFingerprints'
    df = pd.DataFrame({"True Labels": all_labels, "Predictions": all_preds, "Probabilities": all_probabilities})
    df.to_excel(f'{url_dir_to_save_model}/test_predictions.xlsx', index=False)
    save_confusion_matrix_to_excel(testing_confusion_matrix, url_dir_to_save_model, classification_type)
    save_heatmap(testing_confusion_matrix.cpu().numpy(), url_dir_to_save_model, classification_type)
    print("Scores and visualizations saved...")

    return testing_scores_dict



# Get Dataloader
model = "resnet"
train_ds, validate_ds, test_ds = get_datasets(
model=model,
classification_type="binary-10", 
proportional="non-proportional", 
seed=40)

generator = Generator().manual_seed(40)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE[model], num_workers=24, persistent_workers=False, pin_memory=True, generator=generator, collate_fn=COLLATE_FN[model])

model = get_model(
    model=model,
    classification_type="binary-10",
)

my_model = DataParallel(model, device_ids=DEVICE_IDS).to(DEV)

test_model(
    model=my_model,
    test_loader=test_loader,
    classification_type="binary-10",
    url_dir_to_save_model="/home/hessos4l/Downloads/DetectingVocoderFingerprints/trained_models/resnet/binary-10/40"

)