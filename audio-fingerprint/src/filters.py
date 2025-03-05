import torch 
from .utils import TimeInvFIRFilter

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
        # batch = batch.unsqueeze(-1)
        # Rearrange the dimensions
        batch = batch.permute(0, 2, 1)
        # self.filter_layer.weight = self.filter_layer.weight.to("cuda")
        batch = self.filter_layer(batch)
        return batch.permute(0, 2, 1)