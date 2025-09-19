import pandas as pd
import os 
import re
import numpy as np

name_mapping = {
    # LJSPEECH
    'ljspeech_avocodo': 'Avo',
    'ljspeech_bigvgan': 'BVG',
    'ljspeech_fast_diff_tacotron': 'FastDiff',
    'ljspeech_hifiGAN': 'HF-G',
    'ljspeech_hnsf': 'NSF',
    'ljspeech_melgan_large': 'MG-L',
    'ljspeech_multi_band_melgan': 'MB-MG',
    'ljspeech_parallel_wavegan': 'PWG',
    'ljspeech_pro_diff': 'ProDiff',
    'ljspeech_waveglow': 'WGlow', 
    'Real': 'Real',
    'Avg.': 'Avg.',

    'FastDiff_tacotron-1': 'FastDiff',
    'I_avocodo-1': 'Avo',
    'J_bigvgan-1': 'BVG',
    'ProDiff-1': 'ProDiff',
    'ljspeech_hifiGAN-1': 'HF-G',
    'ljspeech_hnsf-1': 'NSF',
    'ljspeech_melgan_large-1': 'MG-L',
    'ljspeech_multi_band_melgan-1': 'MB-MG',
    'ljspeech_parallel_wavegan-1': 'PWG',
    'ljspeech_waveglow-1': 'WGlow',
    '1_FastDiff_tacotron_test': 'FastDiff',
    '1_I_avocodo_test': 'Avo',
    '1_J_bigvgan_test': 'BVG',
    '1_ProDiff_test': 'ProDiff',
    '1_ljspeech_hifiGAN_test': 'HF-G',
    '1_ljspeech_hnsf_test': 'NSF',
    '1_ljspeech_melgan_large_test': 'MG-L',
    '1_ljspeech_multi_band_melgan_test': 'MB-MG',
    '1_ljspeech_parallel_wavegan_test': 'PWG',
    '1_ljspeech_waveglow_test': 'WGlow',
    '1_real_test': 'Real',
    '1_all': 'All',
    'FastDiff_tacotron-24.0': 'FastDiff',
    'I_avocodo-24.0': 'Avo',
    'J_bigvgan-24.0': 'BVG',
    'ProDiff-24.0': 'ProDiff',
    'ljspeech_hifiGAN-24.0': 'HF-G',
    'ljspeech_hnsf-24.0': 'NSF',
    'ljspeech_melgan_large-24.0': 'MG-L',
    'ljspeech_multi_band_melgan-24.0': 'MB-MG',
    'ljspeech_parallel_wavegan-24.0': 'PWG',
    'ljspeech_waveglow-24.0': 'WGlow',
    '24.0_FastDiff_tacotron_test': 'FastDiff',
    '24.0_I_avocodo_test': 'Avo',
    '24.0_J_bigvgan_test': 'BVG',
    '24.0_ProDiff_test': 'ProDiff',
    '24.0_ljspeech_hifiGAN_test': 'HF-G',
    '24.0_ljspeech_hnsf_test': 'NSF',
    '24.0_ljspeech_melgan_large_test': 'MG-L',
    '24.0_ljspeech_multi_band_melgan_test': 'MB-MG',
    '24.0_ljspeech_parallel_wavegan_test': 'PWG',
    '24.0_ljspeech_waveglow_test': 'WGlow',
    '24.0_real_test': 'Real',
    '24.0_all': 'All',

    #JSUT
    'jsut_hnsf-24.0': 'NSF',
    'jsut_multi_band_melgan': 'MB-MG',
    'jsut_parallel_wavegan': 'PWG',
    'real-24.0': 'NSF',
    'Avg.': 'Avg.',
    '24.0_jsut_hnsf_test': 'NSF',
    '24.0_jsut_multi_band_melgan_test': 'MB-MG',
    '24.0_jsut_parallel_wavegan_test': 'PWG',
    'real-24.0': 'Real',
    '1_jsut_multi_band_melgan_test': 'MB-MG',
    '1_jsut_parallel_wavegan_test': 'PWG',
    '1_jsut_waveglow_test': 'WGlow',
    '1_jsut_hnsf_test': 'NSF',
    'jsut_multi_band_melgan-1': 'MB-MG',
    'jsut_parallel_wavegan-1': 'PWG',
    'jsut_hnsf-1': 'NSF',
    'real-1': 'Real',

    # ASVSpoof
    "1_A01_test": "A01",
    "1_A02_test": "A02",
    "1_A03_test": "A03",
    "1_A04_test": "A04",
    "1_A05_test": "A05",
    "1_A06_test": "A06",
    "1_A07_test": "A07",
    "1_A08_test": "A08",
    "1_A09_test": "A09",
    "1_A10_test": "A10",
    "1_A11_test": "A11",
    "1_A12_test": "A12",
    "1_A13_test": "A13",
    "1_A14_test": "A14",
    "1_A15_test": "A15",
    "1_A16_test": "A16",
    "1_A17_test": "A17",
    "1_A18_test": "A18",
    "1_A19_test": "A19",
    "1_bonafide_test": "Bonafide",
    "1_all": "Avg.",
    "A01-1": "A01",
    "A02-1": "A02",
    "A03-1": "A03",
    "A04-1": "A04",
    "A05-1": "A05",
    "A06-1": "A06",
    "A16-1": "A16",
    "A19-1": "A19",

    "24.0_A01_test": "A01",
    "24.0_A02_test": "A02",
    "24.0_A03_test": "A03",
    "24.0_A04_test": "A04",
    "24.0_A05_test": "A05",
    "24.0_A06_test": "A06",
    "24.0_A07_test": "A07",
    "24.0_A08_test": "A08",
    "24.0_A09_test": "A09",
    "24.0_A10_test": "A10",
    "24.0_A11_test": "A11",
    "24.0_A12_test": "A12",
    "24.0_A13_test": "A13",
    "24.0_A14_test": "A14",
    "24.0_A15_test": "A15",
    "24.0_A16_test": "A16",
    "24.0_A17_test": "A17",
    "24.0_A18_test": "A18",
    "24.0_A19_test": "A19",
    "24.0_bonafide_test": "Bonafide",
    "24.0_all": "Avg.",
    "A01-24.0": "A01",
    "A02-24.0": "A02",
    "A03-24.0": "A03",
    "A04-24.0": "A04",
    "A05-24.0": "A05",
    "A06-24.0": "A06",
    "A16-24.0": "A16",
    "A19-24.0": "A19",

    "C1": "C1",
    "C2": "C2",
    "C3": "C3",
    "C4": "C4",
    "C5": "C5",
    "C6": "C6",
    "C7": "C7",
}

def dynamic_name_mapping(name):
    # Pattern: A16-3 or A16-4-7 -> A16
    m = re.match(r'^(A\d{2})-', name)
    if m:
        return m.group(1)
    # Pattern: 3_A07_test or 4-7_A07_test -> A07
    m = re.match(r'^[\d\-]+_(A\d{2})_test$', name)
    if m:
        return m.group(1)
    # Pattern: 3_bonafide_test or 4-7_bonafide_test -> Bonafide
    if re.match(r'^[\d\-]+_bonafide_test$', name, re.IGNORECASE):
        return "Bonafide"
    # Pattern: 3_all or 4-7_all -> Avg.
    if re.match(r'^[\d\-]+_all$', name, re.IGNORECASE):
        return "Avg."
    # Fallback: return original name
    return name

def matrix_gen(file_path):
    # Load all sheets
    # print(file_path)
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names  # List of sheet names
    # print(sheets)
    # Create a dictionary to store AUROC values from all sheets
    auroc_data = {}

    # Loop through each sheet and extract data
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet, dtype={'AUC': float})  # Read current sheet
        df.set_index("vs_model", inplace=True)  # Set vocoder names as index
        # Ensure numbers retain all decimal places when displayed
        pd.set_option("display.float_format", lambda x: f"{x:.10f}")  # Adjust precision as needed

        # If some columns are automatically read as objects, convert them to float
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].astype(float)

        auroc_data[sheet] = df.iloc[:, 0]  # Extract AUROC values
        # break
    # Convert dictionary into DataFrame (rows = vocoder names, columns = compared vocoders)
    auroc_matrix = pd.DataFrame(auroc_data)

    # Define replacement mappings for rows and columns
    # print(auroc_matrix)

    # Rename rows and columns
    auroc_matrix.rename(index=name_mapping, inplace=True)
    auroc_matrix.rename(columns=name_mapping, inplace=True)

    # Define the correct order of vocoders, including "Real" and "All"
    # print(sheets[0])
    if len(sheets) == 6 and sheets[0] != "C1":
        desired_order = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A17', 'A18', 'Bonafide', 'Avg.']
    elif len(sheets) == 6:
        desired_order = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Real', 'Avg.']
    elif len(sheets) == 2 and sheets[0] == "A16":
        desired_order = ['A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16','A17', 'A18', 'A19','Bonafide', 'Avg.']
    elif len(sheets) > 6:
        desired_order = ['FastDiff', 'ProDiff', 'MG-L', 'Avo', 'BVG', 'HF-G', 'MB-MG', 'PWG', 'WGlow', 'NSF', 'Real', 'Avg.']
    else:
        desired_order = ['MB-MG', 'PWG', 'Real', 'Avg.']
    # print(desired_order)
    # Check for duplicate labels in the index and columns
    # print(auroc_matrix.index.duplicated().sum(), auroc_matrix.index)  # Count of duplicate index labels
    # print(auroc_matrix.columns.duplicated().sum(), auroc_matrix.columns)  # Count of duplicate column labels
    
    # Reindex rows and columns to match the desired order

    auroc_matrix.index = [dynamic_name_mapping(idx) for idx in auroc_matrix.index]

    auroc_matrix.columns = [dynamic_name_mapping(col) for col in auroc_matrix.columns]

    auroc_matrix = auroc_matrix.reindex(index=desired_order, columns=desired_order)
    # Drop columns if they exist
    auroc_matrix = auroc_matrix.drop(columns=[c for c in ["Real", "Avg.", "C7", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A17", "A18", "Bonafide"] if c in auroc_matrix.columns])
    # Round all values to 2 decimal places
    # Set display format for floats to 2 decimals
    pd.set_option('display.float_format', '{:.2f}'.format)   
    # Replace all NaN values with "-"
    # auroc_matrix = auroc_matrix.fillna("-")

    print(auroc_matrix)
    '''
    # **Fix "FastDiff" Column Being NaN**
    if auroc_matrix["FastDiff"].isnull().all():
        print("Warning: 'FastDiff' column is completely NaN. Fixing it now.")
        if "FastDiff" in auroc_matrix.index:
            auroc_matrix["FastDiff"] = auroc_matrix.loc["FastDiff"]  # Copy row values into column
    
    # **Fix "Real" and "All" Rows Missing**
    for name in ["Real", "All"]:
        if auroc_matrix.loc[name].isnull().all():
            print(f"Warning: '{name}' row is completely NaN. Filling with 1.0 (default).")
            auroc_matrix.loc[name] = 1.0  # Default to 1.0 for missing data
    print(auroc_matrix)
    '''
    return auroc_matrix

if __name__ == "__main__":

    # Define file path
    base_paths = {'encodec': [
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/ljspeech/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/21/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/ljspeech/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/80/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/ljspeech/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/40/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/ljspeech/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1000/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/ljspeech/"
    ],
    'lpf': 
    [
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1/low_pass_filter_Avg_Spec_aucs/ljspeech/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/21/low_pass_filter_Avg_Spec_aucs/ljspeech/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/80/low_pass_filter_Avg_Spec_aucs/ljspeech/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/40/low_pass_filter_Avg_Spec_aucs/ljspeech/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1000/low_pass_filter_Avg_Spec_aucs/ljspeech/" 
    ],
    'encodec_jsut': [
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/jsut/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/21/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/jsut/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/80/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/jsut/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/40/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/jsut/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1000/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/jsut/"
    ],
    'lpf_jsut': 
    [
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1/low_pass_filter_Avg_Spec_aucs/jsut/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/21/low_pass_filter_Avg_Spec_aucs/jsut/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/80/low_pass_filter_Avg_Spec_aucs/jsut/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/40/low_pass_filter_Avg_Spec_aucs/jsut/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1000/low_pass_filter_Avg_Spec_aucs/jsut/" 
    ], 
    'asv_spoof_lpf': 
    [
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1/low_pass_filter_Avg_Spec_aucs/asvspoof/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/21/low_pass_filter_Avg_Spec_aucs/asvspoof/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/80/low_pass_filter_Avg_Spec_aucs/asvspoof/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/40/low_pass_filter_Avg_Spec_aucs/asvspoof/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1000/low_pass_filter_Avg_Spec_aucs/asvspoof/" 
    ],
    'asv_spoof_encodec': 
    [
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/asvspoof/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/21/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/asvspoof/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/80/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/asvspoof/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/40/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/asvspoof/",
    "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1000/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/asvspoof/" 
    ],
    'lpf_rebuttal':
    [
    "/USERSPACE/pizarm5k/github_fingerprint/fingerprint/aucs/ljspeech/1/low_pass_filter_Avg_Spec_aucs"
    ]
    }
    # file_path = "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1/low_pass_filter_Avg_Spec_aucs/ljspeech/mahalanobis_vcds=paper_pert=None_1_5_param=1.0_nfft=128_hoplen=2_trend=False_cutoff=None_ntrain=10480_ntest=2620.xlsx"
    # Initialize variables
    sum_df = None
    count = 0
    filter = "lpf_rebuttal"
    '''
    for i in base_paths[filter]:
        if filter == "lpf":
            file_path = os.path.join(i, "mahalanobis_vcds=paper_pert=None_1_5_param=1.0_nfft=128_hoplen=2_trend=False_cutoff=None_ntrain=10480_ntest=2620.xlsx")
        elif filter == 'encodec':
            file_path = os.path.join(i, "correlation_vcds=paper_pert=None_1_5_param=24.0_nfft=2048_hoplen=128_trend=False_cutoff=None_ntrain=10480_ntest=2620.xlsx")
        elif filter == 'encodec_jsut':
            file_path = os.path.join(i, "correlation_vcds=paper_pert=None_1_5_param=24.0_nfft=2048_hoplen=128_trend=False_cutoff=None_ntrain=4000_ntest=1000.xlsx")
        elif filter == 'lpf_jsut':
            file_path = os.path.join(i, "mahalanobis_vcds=paper_pert=None_1_5_ake_param=1.0_nfft=128_hoplen=2_trend=False_cutoff=None_ntrain=4000_ntest=1000.xlsx")
        elif filter == 'asv_spoof_lpf':
            file_path = os.path.join(i, "mahalanobis_vcds=paper_pert=None_1_5_None_param=1.0_nfft=128_hoplen=2_trend=False_cutoff=None_ntrain=6012_ntest=1504.xlsx")
        elif filter == 'asv_spoof_encodec':
            file_path = os.path.join(i, "correlation_vcds=paper_pert=None_1_5_None_param=24.0_nfft=2048_hoplen=128_trend=False_cutoff=None_ntrain=6012_ntest=1504.xlsx")
        elif filter == "lpf_rebuttal":
            'mahalanobis_param=1.0_nfft=128_hoplen=2_trend=False_corrtype0_1.0.xlsx'
            'mahalanobis_param=1.0_nfft=128_hoplen=2_trend=False_corrtype1_0.2.xlsx'
            'mahalanobis_param=1.0_nfft=128_hoplen=2_trend=False_corrtype1.xlsx'
            file_path = os.path.join(i, "mahalanobis_param=1.0_nfft=128_hoplen=2_trend=False_corrtype1.xlsx")
            # file_path = os.path.join(i, "correlation_vcds=paper_pert=None_1_5_None_param=24.0_nfft=128_hoplen=2_trend=False_cutoff=None_ntrain=6012_ntest=1504.xlsx")
    
        # file_path = "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/90/EncodecFilter-compute_samplewise=False_Avg_Spec_aucs/asvspoof_val/correlation_vcds=paper_pert=None_1_5_None_param=24.0_nfft=2048_hoplen=128_trend=False_cutoff=None_ntrain=4668_ntest=246.xlsx"
        # print(file_path)
        auroc_matrix = matrix_gen(file_path)

        # break
        if sum_df is None:
            sum_df = auroc_matrix.copy()  # Initialize with the first DataFrame
        else:
            sum_df += auroc_matrix  # Sum the DataFrames
        
        count += 1
        # Print the final AUROC matrix
        # print("\nFixed AUROC Matrix:")
        # print(auroc_matrix)

        # Print the final row and column names
        # print("\nFinal Row Names (Vocoder Reference Models):")
        # print(list(auroc_matrix.index))

        # print("\nFinal Column Names (Compared Vocoders):")
        # print(list(auroc_matrix.columns))
        break

        # break
        if sum_df is None:
            sum_df = auroc_matrix.copy()  # Initialize with the first DataFrame
        else:
            sum_df += auroc_matrix  # Sum the DataFrames
        
        count += 1
        # Print the final AUROC matrix
        # print("\nFixed AUROC Matrix:")
        # print(auroc_matrix)

        # Print the final row and column names
        # print("\nFinal Row Names (Vocoder Reference Models):")
        # print(list(auroc_matrix.index))

        # print("\nFinal Column Names (Compared Vocoders):")
        # print(list(auroc_matrix.columns))

    # Compute the average DataFrame
    average_df = sum_df / count if count > 0 else None
    # print(sum_df)
    # Display the result
    # Print in a format that can be pasted into Excel
    if average_df is not None:
        # print(average_df)
        print(average_df.T.to_string(index=True, header=True, col_space=10))
        # print(average_df.to_csv(sep='\t', index=True, header=True))
    print(count)
    '''
    
    # Low-pass-filter
    file_path = "/USERSPACE/pizarm5k/github_fingerprint/fingerprint/aucs/codecfake/1/low_pass_filter_Avg_Spec_aucs/mahalanobis_param=1.0_nfft=400_hoplen=160_trend=False_corrtype0_1.0.xlsx"
    auroc_matrix = matrix_gen(file_path)
    # Band-pass-filter
    file_path = "/USERSPACE/pizarm5k/github_fingerprint/fingerprint/aucs/codecfake/1/band_pass_filter_Avg_Spec_aucs/mahalanobis_param=5-6_nfft=400_hoplen=160_trend=False_corrtype0_1.0.xlsx"
    auroc_matrix_2 = matrix_gen(file_path)

    # Example: make sure both matrices are numeric, keep NaN for missing
    auroc_matrix_numeric = auroc_matrix.apply(pd.to_numeric, errors='coerce')
    auroc_matrix_2_numeric = auroc_matrix_2.apply(pd.to_numeric, errors='coerce')

    # Function to merge two values as string
    # Merge function
    def merge_entries(x, y):
        if pd.isnull(x) and pd.isnull(y):
            return "-"          # Both missing → single "-"
        x_str = f"{x:.2f}" if pd.notnull(x) else "-"
        y_str = f"{y:.2f}" if pd.notnull(y) else "-"
        return f"{x_str} / {y_str}"

    # Use applymap with np.vectorize for element-wise operation
    merged = pd.DataFrame(np.vectorize(merge_entries)(auroc_matrix.values, auroc_matrix_2.values),
                        index=auroc_matrix.index,
                        columns=auroc_matrix.columns)

    print(merged)