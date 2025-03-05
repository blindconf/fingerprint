import pandas as pd
import os 


name_mapping = {
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

    'jsut_hnsf-24.0': 'NSF',
    'jsut_multi_band_melgan-24.0': 'MB-MG',
    'jsut_parallel_wavegan-24.0': 'PWG',
    'real-24.0': 'NSF',
    'All': 'All',
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
}

def matrix_gen(file_path):
    # Load all sheets
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
        # print(df)
        # If some columns are automatically read as objects, convert them to float
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].astype(float)
        #print(df)
        #print(fasf)

        auroc_data[sheet] = df.iloc[:, 0]  # Extract AUROC values
        
        # break
    # Convert dictionary into DataFrame (rows = vocoder names, columns = compared vocoders)
    auroc_matrix = pd.DataFrame(auroc_data).T  
    # print(auroc_matrix)
    # Define replacement mappings for rows and columns


    # Rename rows and columns
    auroc_matrix.rename(index=name_mapping, inplace=True)
    auroc_matrix.rename(columns=name_mapping, inplace=True)
    # print(auroc_matrix)

    # Define the correct order of vocoders, including "Real" and "All"
    if len(sheets) > 6:
        desired_order = ['FastDiff', 'ProDiff', 'MG-L', 'Avo', 'BVG', 'HF-G', 'MB-MG', 'PWG', 'WGlow', 'NSF', 'Real', 'All']
    else:
        desired_order = ['MB-MG', 'PWG','NSF', 'Real', 'All']
    # print(desired_order)
    # Check for duplicate labels in the index and columns
    # print(auroc_matrix.index.duplicated().sum(), auroc_matrix.index)  # Count of duplicate index labels
    # print(auroc_matrix.columns.duplicated().sum(), auroc_matrix.columns)  # Count of duplicate column labels

    # Reindex rows and columns to match the desired order
    auroc_matrix = auroc_matrix.reindex(index=desired_order, columns=desired_order)
    # print(auroc_matrix)
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
    ]
    }
    # file_path = "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/aucs/1/low_pass_filter_Avg_Spec_aucs/ljspeech/mahalanobis_vcds=paper_pert=None_1_5_param=1.0_nfft=128_hoplen=2_trend=False_cutoff=None_ntrain=10480_ntest=2620.xlsx"
    # Initialize variables
    sum_df = None
    count = 0
    filter = "lpf_jsut"
    for i in base_paths[filter]:
        if filter == "lpf":
            file_path = os.path.join(i, "mahalanobis_vcds=paper_pert=None_1_5_param=1.0_nfft=128_hoplen=2_trend=False_cutoff=None_ntrain=10480_ntest=2620.xlsx")
        elif filter == 'encodec':
            file_path = os.path.join(i, "correlation_vcds=paper_pert=None_1_5_param=24.0_nfft=2048_hoplen=128_trend=False_cutoff=None_ntrain=10480_ntest=2620.xlsx")
        elif filter == 'encodec_jsut':
            file_path = os.path.join(i, "correlation_vcds=paper_pert=None_1_5_param=24.0_nfft=2048_hoplen=128_trend=False_cutoff=None_ntrain=4000_ntest=1000.xlsx")
        elif filter == 'lpf_jsut':
            file_path = os.path.join(i, "mahalanobis_vcds=paper_pert=None_1_5_ake_param=1.0_nfft=128_hoplen=2_trend=False_cutoff=None_ntrain=4000_ntest=1000.xlsx")
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

    # Compute the average DataFrame
    average_df = sum_df / count if count > 0 else None
    # print(sum_df)
    # Display the result
    # Print in a format that can be pasted into Excel
    if average_df is not None:
        print(average_df.to_string(index=True, header=True, col_space=10))
        # print(average_df.to_csv(sep='\t', index=True, header=True))
    print(count)