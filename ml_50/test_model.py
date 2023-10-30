import torch
import pickle
import numpy as np
import pandas as pd
from model import hbiascorrect
from main import seq_len, seq_dim, g_embed_dim, f_embed_dim, output_dim, num_layers, num_heads, hidden_size, dropout_rate, split_size, train_epoch

import matplotlib.pyplot as plt
from matplotlib import rcParams

def load_data_from_files(split_size):
    with open(f"test_tensor_{int((1-split_size)*100)}_{int(split_size*100)}.pkl", 'rb') as f:
        test_tensor = pickle.load(f)

    with open(f"test_real_tensor_{int((1-split_size)*100)}_{int(split_size*100)}.pkl", 'rb') as f:
        test_real_tensor = pickle.load(f)

    with open('scaler_target.pkl', 'rb') as f:
        scaler_target = pickle.load(f)

    return test_tensor, test_real_tensor, scaler_target


def test_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test data
    
    test_tensor, test_real_tensor, scaler_target = load_data_from_files(split_size)

    # Instantiate the model and load the parameters
    # Load the model parameters
    
    transformer = hbiascorrect(seq_len, seq_dim, g_embed_dim, f_embed_dim, output_dim, num_layers, num_heads, hidden_size, dropout_rate)

    with open(f"final_model_parameters_{int((1-split_size)*100)}_{int(split_size*100)}_{train_epoch}.pkl", 'rb') as f:
        model_parameters = pickle.load(f)
    transformer.load_state_dict(model_parameters)

    transformer.to(device)
    transformer.eval()

    predictions = []
    targets = []
    
    

    with torch.no_grad():
        for tensor, real_tensor in zip(test_tensor, test_real_tensor):
            tensor = tensor.unsqueeze(dim=0).to(device)
            val, _, _ = transformer.forward(tensor)

            numpy_array = val.cpu().detach().numpy()
            numpy_array = np.reshape(numpy_array, (-1, seq_len))  
            predicted_scores = scaler_target.inverse_transform(numpy_array)
            predicted_scores = torch.Tensor(predicted_scores)

            # Append predicted scores and target scores to lists
            predictions.append(predicted_scores.cpu().numpy().flatten())
            targets.append(real_tensor.t().cpu().numpy().flatten())

    # Convert the lists to numpy arrays before creating tensors
    predictions = torch.tensor(np.array(predictions)).to(device)
    targets = torch.tensor(np.array(targets)).to(device)

    abs_err = torch.abs(predictions - targets)

    print(f"predictions.size(): {predictions.size()}")
    print(f"targets.size(): {targets.size()}")



    # Calculate the mean and standard deviation for each item in seq_len
    mean_errors = torch.mean(abs_err, dim=0).cpu().numpy()
    std_errors = torch.std(abs_err, dim=0).cpu().numpy()

    # Plot the mean and standard deviation for each item using error bars
    items = np.arange(1, seq_len + 1)
    x_pos = np.arange(len(items))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x_pos, mean_errors, yerr=std_errors, fmt='o', ecolor='r', capsize=5, elinewidth=1, markeredgewidth=1, markerfacecolor='blue', markeredgecolor='blue')
    ax.set_xlabel('Items', fontsize=14)
    ax.set_ylabel('Mean Absolute Error', fontsize=14)
    ax.set_title(f"Mean Absolute Error for movielens 1M dataset", fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(items)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    test_model()
