import sys
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torchaudio
import logging


# Setup module-level logger
logger = logging.getLogger(__name__)

class OracleFilter:
    name = "Oracle"
    def __init__(self, path_real_dir=None, device="cuda"):
        self.path_real_dir = path_real_dir  
        self.device = device 

class filter_fn:
    # name = "low_pass_filter"
    """
    Initializes the filter function.

    :param signal_dim: Dimension of the input signal, defining the expected input feature size.
    :param coef: Coefficients for the filter, determining the characteristics of the filtering operation.
    :param name: Name identifier for the filter, often used to specify filter type or instance.
    """
    def __init__(self, signal_dim, coef, name):
        self.signal_dim = signal_dim
        self.coef = coef
        self.filter_layer = TimeInvFIRFilter(self.signal_dim, self.coef)
        self.name = name
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Rearrange the dimensions
        batch = batch.permute(0, 2, 1)
        batch = self.filter_layer(batch)
        return batch.permute(0, 2, 1)

class EncodecFilter:
    name = "EncodecFilter"
    def __init__(self, model, bandwidth, computations_samplewise=False, device="cuda"):
        self.model = model.to(device)
        self.bandwidth = bandwidth
        self.device = device
        self.model.set_target_bandwidth(bandwidth)   
        self.computations_samplewise = computations_samplewise     
    
    def forward(self, batch: torch.Tensor, batch_sample_rate) -> torch.Tensor:
        with torch.no_grad():
            # Resample input to Encodec model sample rate
            sample_transform = torchaudio.transforms.Resample(batch_sample_rate, self.model.sample_rate).to(self.device)
            batch_converted = sample_transform(batch)
            if self.computations_samplewise:
                output = []
                for i in range(batch_converted.size(0)):
                    output.append(self.model(batch_converted[i][None])[0])
                output = torch.stack(output)
            else:
                output = self.model(batch_converted)
            
            # Resample back to original sample rate
            resample_transform = torchaudio.transforms.Resample(self.model.sample_rate, batch_sample_rate).to(self.device)
            output = resample_transform(output) 
        return output

class Conv1dKeepLength(torch_nn.Conv1d):
    """
    Copyright (c) 2022, Xin Wang, National Institute of Informatics
    All rights reserved.
    https://colab.research.google.com/drive/1EO-ggi1U9f2zXwTiqg7AEljVx11JKta7?usp=sharing

    Wrapper for causal convolution
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
    """ 
        Copyright (c) 2022, Xin Wang, National Institute of Informatics
        All rights reserved.
        https://colab.research.google.com/drive/1EO-ggi1U9f2zXwTiqg7AEljVx11JKta7?usp=sharing
    
        Wrapper to define a FIR filter
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
        if not isinstance(filter_coef, torch.Tensor):
            logger.error("Filter coefficients must be a torch.Tensor.")
            raise TypeError("filter_coef must be a torch.Tensor.")

        if filter_coef.ndim != 1:
            logger.error("Filter coefficients must be a 1D tensor.")
            raise ValueError("filter_coef must be a 1-D tensor.")

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