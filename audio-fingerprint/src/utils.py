import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torchaudio
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
from sklearn.model_selection import train_test_split
import pandas as pd


def get_caching_paths(cache_dir: str, 
                      method_name: str,
                      args: dict) -> dict:
    if args.transformation == "Avg_Spec":
        fingerprint_path = f"{cache_dir}/{method_name}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_{args.transformation}_ntrain={args.num_train}_fingerprint.pickle"
        invcov_path = f"{cache_dir}/{method_name}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_{args.transformation}_ntrain={args.num_train}_invcov.pickle"
        # trend path is only accessed if args.trend_correction==True
        trend_path = f"{cache_dir}/{method_name}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_{args.transformation}_ntrain={args.num_train}_trend.pickle"
    elif args.transformation == "Avg_Mel":
        fingerprint_path = f"{cache_dir}/{method_name}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_nmels={args.nmels}_trend={args.trend_correction}_{args.transformation}_ntrain={args.num_train}_fingerprint.pickle"
        invcov_path = f"{cache_dir}/{method_name}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_nmels={args.nmels}_trend={args.trend_correction}_{args.transformation}_ntrain={args.num_train}_invcov.pickle"
        # trend path is only accessed if args.trend_correction==True
        trend_path = f"{cache_dir}/{method_name}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_nmels={args.nmels}_trend={args.trend_correction}_{args.transformation}_ntrain={args.num_train}_trend.pickle"
    elif args.transformation == "Avg_MFCC":
        fingerprint_path = f"{cache_dir}/{method_name}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_nmels={args.nmels}_nmfcc={args.nmfcc}_trend={args.trend_correction}_{args.transformation}_ntrain={args.num_train}_fingerprint.pickle"
        invcov_path = f"{cache_dir}/{method_name}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_nmels={args.nmels}_nmfcc={args.nmfcc}_trend={args.trend_correction}_{args.transformation}_ntrain={args.num_train}_invcov.pickle"
        # trend path is only accessed if args.trend_correction==True
        trend_path = f"{cache_dir}/{method_name}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_nmels={args.nmels}_nmfcc={args.nmfcc}_trend={args.trend_correction}_{args.transformation}_ntrain={args.num_train}_trend.pickle"
    return {"fingerprint": fingerprint_path, "invcov": invcov_path, "trend": trend_path}

def get_auc_path(args:dict, snr_val) -> str:
    BUTTER_FILTERS = ["LFilterFilter", "FiltFiltFilter"]   

    if args.filter_type in BUTTER_FILTERS:
        raise NotImplementedError("Butterworth paths need to be set properly; Not implemented yet.")
        # auc_dir = f'aucs/{args.filter_type}_{butter_order}_{butter_filter_type}_{args.transformation}_aucs/
    else:
        auc_dir = f'aucs/{args.seed}/{args.filter_type}_{args.transformation}_aucs/{args.corpus}'
        if args.filter_type == "EncodecFilter":
            auc_dir = f'aucs/{args.seed}/{args.filter_type}-compute_samplewise={args.encodec_samplewise}_{args.transformation}_aucs/{args.corpus}'
    perturbation = args.perturbation if args.perturbation=="noise" else f"{args.perturbation}_{args.encodec_qr}"
    
    if args.transformation == "Avg_Spec":
        auc_path = f"{auc_dir}/{args.scorefunction}_vcds={args.vocoders}_pert={perturbation}_{snr_val}_param={args.filter_param}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_cutoff={args.cutoff}_ntrain={args.num_train}_ntest={args.num_test}.xlsx"
    elif args.transformation == "Avg_Mel":
        auc_path = f"{auc_dir}/{args.scorefunction}_vcds={args.vocoders}_pert={perturbation}_param={args.filter_param}_nfft={args.nfft}_hoplen={args.hop_len}_nmels={args.nmels}_trend={args.trend_correction}_cutoff={args.cutoff}_ntrain={args.num_train}_ntest={args.num_test}.xlsx"
    elif args.transformation == "Avg_MFCC":
        auc_path = f"{auc_dir}/{args.scorefunction}_vcds={args.vocoders}_pert={perturbation}_param={args.filter_param}_nfft={args.nfft}_hoplen={args.hop_len}_nmels={args.nmels}_nmfcc={args.nmfcc}_trend={args.trend_correction}_cutoff={args.cutoff}_ntrain={args.num_train}_ntest={args.num_test}.xlsx"
    
    return auc_path 
    


def correlation_score(fingerprint, input_residual):
    # Calculate the correlation scores using inner product
    # We need to remove the singleton dimensions from the fingerprint and batch elements before using torch.inner
    correlation_scores = torch.inner(input_residual.squeeze(1), fingerprint.squeeze(0))
    # return torch.inner(fingerprint.flatten(), input_residual.flatten())
    return correlation_scores

def mahalanobis_score(fingerprint, batch_residual, invcov):
    scores = []
    for i in range(batch_residual.shape[0]):
        input_residual = batch_residual[i, :, :]
        delta = input_residual.flatten() - fingerprint.flatten()   
        score = torch.sqrt(torch.dot(delta, torch.matmul(invcov, delta)))
        scores.append(-1 * score.item())
    return torch.tensor(scores)


def spec_to_mel_spec(spec):
    return 2595 * np.log10(1 + spec/700)

def mel_spec_to_spec(mel_spec):
    return 700 * (10**(mel_spec/2595) - 1)

def _compute_average_frequency_for_directory(directory: str) -> torch.Tensor:
    print("directory: ", directory)
    
    dataset = AudioDataSet(
        directory,
        target_sample_rate=SAMPLE_RATE,
        train_nrows=TRAINING_SAMPLE,
        device='cuda'
    )

    # print(dataset._paths)
    print("sample rate: ", dataset.target_sample_rate)
    average_per_file = []
    average_spc_per_file = []
    spec_transform = Spectrogram(n_fft=N_FFT).to(dataset.device)
    
    for i, (clip, fs) in enumerate(dataset):
        print(clip.shape)
        # print(fs)
        specgram = spec_transform(clip).squeeze(0)
        print("spectrogram: ", specgram.shape)
        avg = torch.mean(specgram, dim=1)
        # print("avg: ", avg.shape)
        average_spc_per_file.append(avg)
        avg_db = 10. * torch.log(avg + 10e-13)
        average_per_file.append(avg_db)

        if i % 1000 == 0:
            print(f"\rProcessed {i:06} files! \n", end="")
    average_per_file = torch.stack(average_per_file)
    average_per_file = torch.mean(average_per_file, dim=0)

    return average_per_file, average_spc_per_file

def _apply_ax_styling(sr, ax, title, num_freqs, y_min=-150., y_max=40, ylabel="Average Energy (dB)"):
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_ylim(y_min, y_max)

    # convert fftbins to freq.
    freqs = np.fft.fftfreq((num_freqs - 1) * 2, 1 /
                        sr)[:num_freqs-1] / 1_000
    ticks = ax.get_xticks()[1:]
    ticklabels = (np.linspace(
        freqs[0], freqs[-1], len(ticks)) + .5).astype(np.int32)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)

    ax.set_xlabel("Frequency (kHz)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    pass

def plot_barplot(data, title, path):
    fig, ax = plt.subplots()
    num_freqs = data.shape[0]

    ax.bar(x=list(range(num_freqs)), height=data, color="#2D5B68")
    _apply_ax_styling(ax, title, num_freqs)

    fig.tight_layout()
    fig.savefig(path)
    plt.close()

def plot_lineplot_var(data, std, title, path):
    fig, ax = plt.subplots()
    num_freqs = data.shape[0]
    
    ax.plot(list(range(num_freqs)), data, '-', color="crimson")
    # ax.plot(x=list(range(num_freqs)), data, '-')
    ax.fill_between(list(range(num_freqs)), data - std, data + std, alpha=0.2, color="crimson")
    _apply_ax_styling(
                ax, title, num_freqs, y_min=-10, y_max=10, ylabel="")
    fig.tight_layout()
    fig.savefig(path)
    plt.close()

def plot_difference(sr, data, title, ref_data, ref_title, path, absolute: bool = False):
    fig, axis = plt.subplots(1, 3, figsize=(20, 5))

    num_freqs = ref_data.shape[0]
    # plot ref
    ax = axis[0]
    ax.bar(x=list(range(num_freqs)), height=ref_data, color="#2D5B68")
    _apply_ax_styling(sr, ax, ref_title, num_freqs)

    # plot differnce
    ax = axis[1]
    diff = data - ref_data
    # print(data)
    # print(ref_data)
    # print(ref_data.shape[0])
    ax.bar(x=list(range(num_freqs)), height=diff, color="crimson")
    if absolute:
        _apply_ax_styling(
            sr, ax, f"absolute differnce {title} - {ref_title}", num_freqs, y_min=0, y_max=10, ylabel="")
        diff = np.abs(diff)
    else:
        _apply_ax_styling(
            sr, ax, f"Differnce {title} - {ref_title}", num_freqs, y_min=-70, y_max=70, ylabel="")

    # plot data
    ax = axis[2]
    ax.bar(x=list(range(num_freqs)), height=data, color="#2D5B68")
    _apply_ax_styling(sr, ax, title, num_freqs)

    fig.tight_layout()
    fig.savefig(path)
    plt.close()

def plot_finger_freq(sr, ref_data, ref_title, path):
    fig, axis = plt.subplots()

    num_freqs = ref_data.shape[0]
    # print(num_freqs)
    # print("ref_data: ", ref_data)
    # plot ref
    axis.bar(x=list(range(num_freqs)), height=ref_data, color="#2D5B68")

    fig.tight_layout()
    fig.savefig(path)
    plt.close()

def save_audio_files(audio_dL, wrapper, path_audio, prefix):
    cont = 0
    for i in audio_dL:
        if cont > 5:
            break
        audio = i[0]
        name = os.path.splitext(os.path.basename(i[2]))[0]
        filtered_input = wrapper.filter.spect_to_audio(audio)
        torchaudio.save(f"{path_audio}/{prefix}_{name}.wav", filtered_input, wrapper.filter.sample_rate)
        cont += 1 
    pass

def hist_plot(save_plot, ref_corr, ref_label, targ_corr, targ_label, title_metric, x_label):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel(x_label, size=15) # 20
    ax1.set_ylabel('Normalized N° of instances', size=15) # 20
    ax1.hist(ref_corr, color=color, alpha=0.5, label=ref_label, density=True)
    color = 'tab:red'
    ax1.hist(targ_corr, color=color, alpha=0.5, label=targ_label, density=True)
    ax1.tick_params(axis='y', labelsize = 9) # 18
    ax1.tick_params(axis='x', labelsize = 9) # 18
    ax1.legend(loc='upper right', prop={'size': 14})
    # ax1.set_ylim([0, 13])
    
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f"AUROC={title_metric}")
    plt.savefig(save_plot, dpi=300)
    plt.show()
    plt.close()
    pass

class Conv1dKeepLength(torch_nn.Conv1d):
    """ Wrapper for causal convolution
    Input tensor:  (batchsize, length, dim_in)
    Output tensor: (batchsize, length, dim_out)
       
    """
    def __init__(self, input_dim, output_dim, dilation_s, kernel_s, 
                 causal = False, stride = 1, groups=1, bias=True, \
                 tanh = True, pad_mode='constant'):
        super(Conv1dKeepLength, self).__init__(
            input_dim, output_dim, kernel_s, stride=1,
            padding = 0, dilation = dilation_s, groups=groups, bias=bias)

        self.pad_mode = pad_mode
        self.causal = causal
        
        # padding size
        # input & output length will be the same
        if self.causal:
            # left pad to make the convolution causal
            self.pad_le = dilation_s * (kernel_s - 1)
            self.pad_ri = 0
        else:
            # pad on both sizes
            self.pad_le = dilation_s * (kernel_s - 1) // 2
            self.pad_ri = dilation_s * (kernel_s - 1) - self.pad_le
    
        # activation functions
        if tanh:
            self.l_ac = torch_nn.Tanh()
        else:
            self.l_ac = torch_nn.Identity()
        
    def forward(self, data):
        # https://github.com/pytorch/pytorch/issues/1333
        # permute to (batchsize=1, dim, length)
        # add one dimension as (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length+pad_length)
        x = torch_nn_func.pad(data.permute(0, 2, 1).unsqueeze(2), \
                              (self.pad_le, self.pad_ri,0,0), \
                              mode = self.pad_mode).squeeze(2)
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = self.l_ac(super(Conv1dKeepLength, self).forward(x))
        return output.permute(0, 2, 1)

class TimeInvFIRFilter(Conv1dKeepLength):                                    
    """ Wrapper to define a FIR filter
        input tensor  (batchsize, length, feature_dim)
        output tensor (batchsize, length, feature_dim)
        
        Define:
            TimeInvFIRFilter(feature_dim, filter_coef, 
                             causal=True, flag_trainable=False)
        feature_dim: dimension of the feature in each time step
        filter_coef: a 1-D torch.tensor of the filter coefficients
        causal: causal filtering y_i = sum_k=0^K a_k x_i-k
                non-causal: y_i = sum_k=0^K a_k x_i+K/2-k
        flag_trainable: whether update filter coefficients (default False)
    """                                                                   
    def __init__(self, feature_dim, filter_coef, causal=True, 
                 flag_trainable=False):
        # define based on Conv1d with stride=1, tanh=False, bias=False
        # groups = feature_dim make sure that each signal is filtered separated 
        super(TimeInvFIRFilter, self).__init__(                              
            feature_dim, feature_dim, 1, filter_coef.shape[0], causal,              
            groups=feature_dim, bias=False, tanh=False)
        
        if filter_coef.ndim == 1:
            # initialize weight and load filter coefficients
            with torch.no_grad():
                tmp_coef = torch.zeros([feature_dim, 1, filter_coef.shape[0]]).to("cuda")
                tmp_coef[:, 0, :] = filter_coef
                tmp_coef = torch.flip(tmp_coef, dims=[2])
                self.weight = torch.nn.Parameter(tmp_coef, requires_grad = flag_trainable)
        else:
            print("TimeInvFIRFilter expects filter_coef to be 1-D tensor")
            print("Please implement the code in __init__ if necessary")
            sys.exit(1)
                                                                                  
    def forward(self, data):                                              
        return super(TimeInvFIRFilter, self).forward(data)

def preemphasis(waveform, coeff: float = 0.97) -> torch.Tensor:
    r"""Pre-emphasizes a waveform along its last dimension, i.e.
    for each signal :math:`x` in ``waveform``, computes
    output :math:`y` as

    .. math::
        y[i] = x[i] - \text{coeff} \cdot x[i - 1]

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Waveform, with shape `(..., N)`.
        coeff (float, optional): Pre-emphasis coefficient. Typically between 0.0 and 1.0.
            (Default: 0.97)

    Returns:
        torch.Tensor: Pre-emphasized waveform, with shape `(..., N)`.
    """
    waveform = waveform.clone()
    waveform[..., 1:] -= coeff * waveform[..., :-1]
    return waveform

def pad_and_stack(tensor_list, pad_dim=0):
    """
    Pads a list of tensors to the same size along each dimension and stacks them along a new dimension.
    
    Args:
        tensor_list (list of torch.Tensor): List of tensors with varying sizes.
        pad_dim (int): Dimension along which the tensors vary (default is 0).
    
    Returns:
        torch.Tensor: A tensor formed by stacking the padded tensors along a new dimension.
    """
    
    # Step 1: Determine the maximum size along each dimension
    max_sizes = [max(tensor.size(dim) for tensor in tensor_list) for dim in range(len(tensor_list[0].size()))]
    
    # Step 2: Pad each tensor to match the maximum size in all dimensions
    padded_tensors = []
    for tensor in tensor_list:
        padding = []
        for i in range(len(tensor.size()) - 1, -1, -1):
            size_diff = max_sizes[i] - tensor.size(i)
            padding.extend([0, size_diff])
        padded_tensor = torch_nn_func.pad(tensor, padding)
        padded_tensors.append(padded_tensor)
    
    # Step 3: Stack the padded tensors along a new dimension
    stacked_tensor = torch.stack(padded_tensors, dim=pad_dim)
    
    return stacked_tensor

def pad_and_concatenate(tensor_list, concat_dim=0):
    """
    Pads a list of tensors to the same size along each dimension and concatenates them along a specified dimension.
    
    Args:
        tensor_list (list of torch.Tensor): List of tensors with varying sizes.
        concat_dim (int): Dimension along which the tensors vary and will be concatenated.
    
    Returns:
        torch.Tensor: A tensor formed by concatenating the padded tensors along the specified dimension.
    """
    
    # Step 1: Determine the maximum size along each dimension
    max_sizes = [max(tensor.size(dim) for tensor in tensor_list) for dim in range(len(tensor_list[0].size()))]
    
    # Step 2: Pad each tensor to match the maximum size in all dimensions except the concatenation dimension
    padded_tensors = []
    for tensor in tensor_list:
        padding = []
        for i in range(len(tensor.size()) - 1, -1, -1):
            size_diff = max_sizes[i] - tensor.size(i) if i != concat_dim else 0
            padding.extend([0, size_diff])
        padded_tensor = torch_nn_func.pad(tensor, padding)
        padded_tensors.append(padded_tensor)
    
    # Step 3: Concatenate the padded tensors along the specified dimension
    concatenated_tensor = torch.cat(padded_tensors, dim=concat_dim)
    
    return concatenated_tensor

# Function to extract the name up to the first underscore
def extract_name(filepath):
    # Get just the filename from the full path
    filename = filepath.split('/')[-1]
    # Split by _ or . and take the first part
    return filename.split('_')[0].split('.')[0]


def generate_csv_files(
    folder_dir,
    real_audio_path,
    fake_audio_path,
    fake_folders,
    arg_seed):

    csv_files_split_train_dir_path = os.path.join(folder_dir, 'train')
    csv_files_split_test_dir_path = os.path.join(folder_dir, 'test')

    # Ensure the directory exists before saving
    os.makedirs(csv_files_split_train_dir_path, exist_ok=True)
    os.makedirs(csv_files_split_test_dir_path, exist_ok=True)

    # Scenario: Dir with one csv file (Real Audio Dir):
    # Get a list of all wav file paths
    real_wav_files = sorted([os.path.join(real_audio_path, f) for f in os.listdir(real_audio_path) if f.endswith(".wav")])
    print(len(real_wav_files))
    print(asfsaf)
    # Create a DataFrame
    df = pd.DataFrame(real_wav_files, columns=["file_path"])
    # Define CSV save path
    csv_path = os.path.join(folder_dir, "real_audio_files.csv")
    # Save to CSV
    df.to_csv(csv_path, index=False)
    # Split dataset into train and test
    dataset_df = pd.read_csv(csv_path)
    train_df, test_df =  train_test_split(dataset_df, test_size=0.2, train_size=0.8, random_state=arg_seed)
    # Sort train_df by the "file_path" column 
    train_df_sorted = train_df.sort_values(by="file_path")
    test_df_sorted = test_df.sort_values(by="file_path")
    # Save train, and test dataframes to csvs
    train_df_sorted.to_csv(os.path.join(csv_files_split_train_dir_path, 'real_train.csv'), index=False, header=False)
    test_df_sorted.to_csv(os.path.join(csv_files_split_test_dir_path, 'real_test.csv'), index=False, header=False)
    # Delete the master CSV file after saving
    os.remove(csv_path)
    for i in fake_folders:
        # Construct the fake path
        fake_path = os.path.join(fake_audio_path, i)
        fake_wav_files = sorted([os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith(".wav")])
        # Create a DataFrame
        df = pd.DataFrame(fake_wav_files, columns=["file_path"])
        # Define CSV save path
        csv_path = os.path.join(folder_dir, f"{i}_fake_audio_files.csv")
        # Save to CSV
        df.to_csv(csv_path, index=False)
        # Read the dataset CSV
        dataset_df = pd.read_csv(csv_path)
        # Split dataset into train and test
        train_df, test_df =  train_test_split(dataset_df, test_size=0.2, train_size=0.8, random_state=arg_seed)
        # Sort train_df by the "file_path" column 
        train_df_sorted = train_df.sort_values(by="file_path")
        test_df_sorted = test_df.sort_values(by="file_path")
        # Save train, and test dataframes to csvs
        train_df_sorted.to_csv(os.path.join(csv_files_split_train_dir_path, f"{i}_train.csv"), index=False, header=False)
        test_df_sorted.to_csv(os.path.join(csv_files_split_test_dir_path, f"{i}_test.csv"), index=False, header=False)
        # Delete the master CSV file after saving
        os.remove(csv_path)
    pass
    
def generate_csv_files_noise(
    folder_dir,
    real_audio_path,
    fake_audio_path,
    fake_folders,
    arg_seed):

    # Scenario: Dir with one csv file (Real Audio Dir):
    # Get a list of all wav file paths
    real_wav_files = sorted([os.path.join(real_audio_path, f) for f in os.listdir(real_audio_path) if f.endswith(".wav")])
    # Create a DataFrame
    df = pd.DataFrame(real_wav_files, columns=["file_path"])
    test_df_sorted = df.sort_values(by="file_path")
    # Define CSV save path
    csv_path = os.path.join(folder_dir, "real_test.csv")
    # Save to CSV
    test_df_sorted.to_csv(csv_path, index=False, header=False)
    for i in fake_folders:
        # Construct the fake path
        fake_path = os.path.join(fake_audio_path, i)
        fake_wav_files = sorted([os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith(".wav")])
        # Create a DataFrame
        df = pd.DataFrame(fake_wav_files, columns=["file_path"])
        test_df_sorted = df.sort_values(by="file_path")
        # Define CSV save path
        csv_path = os.path.join(folder_dir, f"{i}_test.csv")
        # Save to CSV
        test_df_sorted.to_csv(csv_path, index=False, header=False)
    pass

def get_csv_dict(folder_dir, pert_val):
    """
    Reads the names of CSV files in a directory and returns them as a list.
    
    :param dir_data_files: Path to the directory containing CSV files.
    :return: List of CSV file names (e.g., ['a.csv', 'b.csv', 'c.csv']).
    """
    if pert_val == "noise":
        new_path = os.path.join(*folder_dir.strip("/").split("/")[:-2])
        train_dir_path = os.path.join(new_path, 'train')
        test_dir_path = folder_dir
    else:
        train_dir_path = os.path.join(folder_dir, 'train')
        test_dir_path = os.path.join(folder_dir, 'test')    
    train_dir_path_sorted = sorted([f for f in os.listdir(train_dir_path) if f.endswith(".csv")])
    test_dir_path_sorted = sorted([f for f in os.listdir(test_dir_path) if f.endswith(".csv")])
    return train_dir_path_sorted, test_dir_path_sorted
    