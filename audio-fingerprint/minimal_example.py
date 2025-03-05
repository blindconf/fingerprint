import torch
from src.fingerprinting import FingerprintingWrapper, HighPassFilter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


if __name__=="__main__":
    # generate random data from 2 sources:
    # for a minimal example, the two souces are just two differnet gaussian distributions
    mean_a = torch.randn((64, 64))
    mean_b = torch.randn((64, 64))

    n_train = 1000
    n_test = 100
    num_frequency_bins = 64
    num_time_bins = 64

    # generate train data from generator a
    train_data = torch.randn((n_train, num_time_bins, num_frequency_bins)) + mean_a

    # generate test data from generator a and b
    test_data_a = torch.randn((n_test, num_time_bins, num_frequency_bins)) + mean_a
    test_data_b = torch.randn((n_test, num_time_bins, num_frequency_bins)) + mean_b
    test_data = torch.concat([test_data_a, test_data_b], dim=0)


    wrapper = FingerprintingWrapper(filter=HighPassFilter(num_frequency_bins=64))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    wrapper.train(train_loader)

    output_a = wrapper.forward(test_data_a)
    output_b = wrapper.forward(test_data_b)

    plt.hist(output_a, label="source a", color="orange")
    plt.hist(output_b, label="source b", color="blue")
    plt.legend()
    plt.savefig("minimal_example.png")