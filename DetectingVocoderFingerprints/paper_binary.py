import pandas as pd
import os
import numpy as np
from scipy.stats import ranksums  # Wilcoxon Rank-Sum Test

# List of folder names (seeds)
seeds = ['1', '21', '40', '80', '1000']
folders = ['x-vector', 'lcnn', 'resnet', 'se-resnet', 'vfd-resnet', 'fingerprints']  # Include fingerprints and vfd-resnet

# Path to the parent directory where folders are located
parent_dir = '/home/hessos4l/Downloads/DetectingVocoderFingerprints/trained_models/'

# Create an empty list to store the results for the final summary table
final_data = []
p_values_data = []  # Separate table for P-values

# Define the fingerprints scores to calculate the average and std deviation
'''
fingerprints_scores = [
    [0.99866, 0.99866, 0.99847, 0.99885],  # Accuracy, F1, Precision, Recall for seed 1
    [0.99867, 0.99867, 0.99848, 0.99886],  # Seed 2
    [0.99865, 0.99865, 0.99846, 0.99884],  # Seed 3
    [0.99856, 0.99856, 0.99837, 0.99875],  # Seed 4
    [0.99896, 0.99896, 0.99897, 0.99895]   # Seed 5
]
'''
# Initialize list to store extracted scores
fingerprints_scores = []

for seed in seeds:
    folder_path = os.path.join(parent_dir, 'fingerprints', 'binary-10/non-proportional', seed) 
    file_path = os.path.join(folder_path, 'testing_scores.xlsx') 

    if os.path.exists(file_path):  # Ensure the file exists
        df = pd.read_excel(file_path)  # Load the Excel file
        # Assuming the metrics are in the first row (index 0)
        metrics_1 = df.iloc[0][['Testing_Accuracy', 'Testing_F1_Score', 'Testing_Precision', 'Testing_Recall']].tolist()
        fingerprints_scores.append(metrics_1)  # Store extracted values
    else:
        print(f"Warning: File not found - {file_path}")

# Print the extracted fingerprint scores
print("Extracted Fingerprint Scores:")
for i, scores in enumerate(fingerprints_scores):
    print(f"Seed {seeds[i]}: {scores}")

# Loop through each parent (method)
for parent in folders:
    sum_df = None
    count = 0
    metrics_list = []  # To collect the metrics for each method to compute standard deviation

    for seed in seeds:
        if parent == 'vfd-resnet':
            folder_path = os.path.join(parent_dir, parent, 'binary-10/non-proportional_old', seed)
        else:
            folder_path = os.path.join(parent_dir, parent, 'binary-10/non-proportional', seed)
        file_path = os.path.join(folder_path, 'testing_scores.xlsx')

        if os.path.exists(file_path):  # Check if the file exists in the folder
            # Read the Excel file
            df = pd.read_excel(file_path)
            '''
            if parent == 'fingerprints':
                # For 'fingerprints', set 'Metric' as index and transpose it
                df = df.set_index('Metric').transpose()
            '''
            # If this is the first file, initialize sum_df
            if sum_df is None:
                sum_df = df.copy()  # Initialize with the first DataFrame
            else:
                sum_df += df  # Sum the DataFrames

            count += 1
            # Collect the metrics for standard deviation computation
            if parent == 'fingerprints':
                metrics_list.append(fingerprints_scores[count-1])  # Collect fingerprint scores
            else:
                metrics_list.append(df.loc[0, ['Testing_Accuracy', 'Testing_F1_Score', 'Testing_Precision', 'Testing_Recall']])

    # Now calculate the average DataFrame if count > 0
    average_df = sum_df / count if count > 0 else None

    if average_df is not None:
        # Extract the relevant metrics: Accuracy, F1 Score, Precision, Recall
        if parent == 'fingerprints':
            # Calculate the mean for fingerprints metrics across the five seeds
            avg_fingerprints = [sum(x) / len(x) for x in zip(*fingerprints_scores)]
            metrics = avg_fingerprints  # Store the average of all the fingerprint metrics
        else:
            # For other methods, extract relevant values
            metrics = average_df.loc[0, ['Testing_Accuracy', 'Testing_F1_Score', 'Testing_Precision', 'Testing_Recall']]

        # Calculate the standard deviation for each metric (across the seeds)
        std_dev = np.std(metrics_list, axis=0)

        # Perform Wilcoxon Rank-Sum Test against fingerprints
        if parent != 'fingerprints':
            p_values = []
            for i in range(4):  # Loop through the metrics: Accuracy, F1, Precision, Recall
                stat, p_value = ranksums([x[i] for x in fingerprints_scores], [x[i] for x in metrics_list])
                p_values.append(p_value)

            # Save the p-values in a separate table
            p_values_data.append({
                'Methods': parent,
                'P-Value (Accuracy)': f"{p_values[0]:.3f}",
                'P-Value (F1)': f"{p_values[1]:.3f}",
                'P-Value (Precision)': f"{p_values[2]:.3f}",
                'P-Value (Recall)': f"{p_values[3]:.3f}",
            })

            # Prepare a dictionary for the row data to add to the final table
            final_data.append({
                'Methods': parent,
                'Accuracy': f"{metrics[0]:.3f} ({std_dev[0]:.3f})",
                'F1 Score': f"{metrics[1]:.3f} ({std_dev[1]:.3f})",
                'Precision': f"{metrics[2]:.3f} ({std_dev[2]:.3f})",
                'Recall': f"{metrics[3]:.3f} ({std_dev[3]:.3f})",
            })

# Add a row for the 'fingerprints' method explicitly
avg_fingerprints = [sum(x) / len(x) for x in zip(*fingerprints_scores)]  # Calculate the mean values
std_fingerprints = np.std(fingerprints_scores, axis=0)  # Calculate the standard deviation for fingerprints
final_data.append({
    'Methods': 'fingerprints',
    'Accuracy': f"{avg_fingerprints[0]:.3f} ({std_fingerprints[0]:.3f})",
    'F1 Score': f"{avg_fingerprints[1]:.3f} ({std_fingerprints[1]:.3f})",
    'Precision': f"{avg_fingerprints[2]:.3f} ({std_fingerprints[2]:.3f})",
    'Recall': f"{avg_fingerprints[3]:.3f} ({std_fingerprints[3]:.3f})",
})

# Create DataFrames
metrics_df = pd.DataFrame(final_data)
p_values_df = pd.DataFrame(p_values_data)

# Print metrics table
print("### Metrics Table (Mean ± Standard Deviation) ###")
print(metrics_df.to_string(index=False))

# Print p-values table separately
print("\n### P-Values (Wilcoxon Rank-Sum Test) ###")
print(p_values_df.to_string(index=False))
