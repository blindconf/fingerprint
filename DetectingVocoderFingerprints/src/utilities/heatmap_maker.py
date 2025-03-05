import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.training.invariables import MULTI_CLASS_10_LABELS, BINARY_CLASS_LABELS


def save_heatmap_from_excel(excel_file, sheet_name, destination_url, classification_type):
    """
    Creates a heatmap from a confusion matrix loaded from an Excel file.

    Args:
        excel_file (str): Path to the Excel file containing the confusion matrix.
        sheet_name (str): Name of the sheet in the Excel file where the confusion matrix is located.
        destination_url (str): Directory where the heatmap will be saved.
        classification_type (str): Type of classification, either "multiclass" or "binary".

    Raises:
        ValueError: If an invalid classification type is provided.
    """
    # Load the confusion matrix from the Excel file
    try:
        conf_matrix = pd.read_excel(excel_file, sheet_name=sheet_name, header=None).iloc[1:, 1:]
        conf_matrix = conf_matrix.apply(pd.to_numeric, errors="coerce").fillna(0).values  # Ensure numeric data
    except Exception as e:
        raise ValueError(f"Error reading the Excel file: {e}")

    # Validate classification type and assign labels
    if classification_type == "multiclass":
        labels = MULTI_CLASS_10_LABELS
    elif classification_type == "binary":
        labels = BINARY_CLASS_LABELS
    else:
        raise ValueError(
            f'Function "save_heatmap_from_excel": Invalid classification type provided. Expected "binary" or "multiclass".'
        )

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[f"Pred_{i}" for i in labels],
                yticklabels=[f"True_{i}" for i in labels])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Save the heatmap to the specified directory
    heatmap_path = f'{destination_url}/testing_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()

    print(f"Heatmap of confusion matrix saved to {heatmap_path}")

