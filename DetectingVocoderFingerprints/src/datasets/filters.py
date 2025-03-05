from torch.nn import Tanh, Identity, Conv1d
import torch
import sys


class Conv1dKeepLength(Conv1d):
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
            self.l_ac = Tanh()
        else:
            self.l_ac = Identity()


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
    def __init__(self, feature_dim, filter_coef, dev, causal=True, 
                 flag_trainable=False):
        self.dev = dev
        # define based on Conv1d with stride=1, tanh=False, bias=False
        # groups = feature_dim make sure that each signal is filtered separated 
        super(TimeInvFIRFilter, self).__init__(                              
            feature_dim, feature_dim, 1, filter_coef.shape[0], causal,              
            groups=feature_dim, bias=False, tanh=False)
        
        if filter_coef.ndim == 1:
            # initialize weight and load filter coefficients
            with torch.no_grad():
                tmp_coef = torch.zeros([feature_dim, 1, filter_coef.shape[0]]).to(self.dev)
                tmp_coef[:, 0, :] = filter_coef
                tmp_coef = torch.flip(tmp_coef, dims=[2])
                self.weight = torch.nn.Parameter(tmp_coef, requires_grad = flag_trainable)
        else:
            print("TimeInvFIRFilter expects filter_coef to be 1-D tensor")
            print("Please implement the code in __init__ if necessary")
            sys.exit(1)
                                                                                  
    def forward(self, data):                                              
        return super(TimeInvFIRFilter, self).forward(data)
    

class filter_fn:
    # name = "low_pass_filter"
    def __init__(self, signal_dim, coef, dev):
        self.signal_dim = signal_dim
        self.coef = coef
        self.dev = dev
        self.filter_layer = TimeInvFIRFilter(self.signal_dim, self.coef, self.dev)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch = batch.unsqueeze(-1)
        # Rearrange the dimensions
        #batch = batch.permute(0, 2, 1)
        self.filter_layer.weight = self.filter_layer.weight.to(self.dev)
        batch = self.filter_layer(batch)
        return batch
        #return batch.permute(0, 2, 1)
