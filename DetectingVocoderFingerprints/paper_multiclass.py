import pandas as pd
import os
from scipy.stats import ranksums

# List of folder names (seeds)
seeds = ['1', '21', '40', '80', '1000']
folders = ['x-vector', 'lcnn', 'resnet', 'se-resnet', 'vfd-resnet', 'fingerprints']

# Path to the parent directory where folders are located
parent_dir = '/home/hessos4l/Downloads/DetectingVocoderFingerprints/trained_models/'

# Create an empty list to store the results for the final summary table
final_data = []
fingerprint_scores = []
method_scores = {}

# Loop through each parent (method) to calculate mean and std for each metric
for parent in folders:
    metric_values = {'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': []}
    
    for seed in seeds:
        folder_path = os.path.join(parent_dir, parent, 'multiclass-10', seed)
        file_path = os.path.join(folder_path, 'testing_scores.xlsx')

        if os.path.exists(file_path):  # Check if the file exists
            df = pd.read_excel(file_path)

            if parent == 'fingerprints':
                df = df.set_index('Metric').transpose()
                metric_values['Accuracy'].append(df.loc['Score', 'Accuracy'])
                metric_values['Precision'].append(df.loc['Score', 'Precision'])
                metric_values['Recall'].append(df.loc['Score', 'Recall'])
                metric_values['F1 Score'].append(df.loc['Score', 'F1 Score'])
            else:
                metric_values['Accuracy'].append(df.loc[0, 'Testing_Accuracy'])
                metric_values['Precision'].append(df.loc[0, 'Testing_Precision'])
                metric_values['Recall'].append(df.loc[0, 'Testing_Recall'])
                metric_values['F1 Score'].append(df.loc[0, 'Testing_F1_Score'])

    # Compute mean and standard deviation
    if metric_values['Accuracy']:
        mean_std_metrics = {metric: (sum(values) / len(values), pd.Series(values).std()) for metric, values in metric_values.items()}

        if parent == 'fingerprints':
            fingerprint_scores = metric_values['Accuracy']  # Store fingerprint scores separately
        else:
            method_scores[parent] = metric_values['Accuracy']  # Store method scores for WRST comparison

        # Prepare row data
        final_data.append({
            'Methods': parent,
            'Accuracy': f"{mean_std_metrics['Accuracy'][0]:.3f} ({mean_std_metrics['Accuracy'][1]:.3f})",
            'F1 Score': f"{mean_std_metrics['F1 Score'][0]:.3f} ({mean_std_metrics['F1 Score'][1]:.3f})",
            'Precision': f"{mean_std_metrics['Precision'][0]:.3f} ({mean_std_metrics['Precision'][1]:.3f})",
            'Recall': f"{mean_std_metrics['Recall'][0]:.3f} ({mean_std_metrics['Recall'][1]:.3f})",
            'p-value': "-" if parent == 'fingerprints' else ""  # Placeholder for p-values
        })

# Perform Wilcoxon Rank-Sum Test (WRST) for each method vs. fingerprints
for row in final_data:
    method = row['Methods']
    if method != 'fingerprints' and len(fingerprint_scores) == 5 and len(method_scores[method]) == 5:
        W_stat, p_value = ranksums(fingerprint_scores, method_scores[method])
        row['p-value'] = f"{p_value:.4f}"
          # Interpretation
        if p_value < 0.05:
            print(f"Fingerprint method significantly outperforms {method} (p < 0.05)")
        else:
            print(f"No significant difference between Fingerprint and {method} (p >= 0.05)")
# Create the final DataFrame with the collected data
final_df = pd.DataFrame(final_data)

# Print or save the final summary table
print(final_df)
