from model import hbiascorrect
# from features import preprocess_data
import torch.optim as optim
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time
import pickle
from torch.utils.tensorboard import SummaryWriter


def train_model(transformer, dataloader, max_optim_epoch, min_optim_epoch, train_epoch, optimizer, split_size):
    
    criterion = torch.nn.MSELoss()
    train_loss = torch.zeros(train_epoch)

    print(f"train_epoch: {train_epoch}")
    total_batches = len(dataloader)
    total_training_time= 0

    # Define the file path for logging TensorBoard data
    log_dir = f"./logs/{int((1-split_size)*100)}_{int(split_size*100)}_{train_epoch}"

    # Create a SummaryWriter instance with the specified file path
    writer = SummaryWriter(log_dir=log_dir)


    for i in range(train_epoch):
        print(f"Epoch: {i + 1}/{train_epoch}")
        start_time = time.time()  # Time at the start of the epoch


        for batch_idx, (input_batch, y_label_batch, y_star_batch) in enumerate(dataloader, 1):
            print(f"\tBatch: {batch_idx}/{total_batches}")

            for param in transformer.parameters():
                param.requires_grad = False

            for param in transformer.matlayer.parameters():
                param.requires_grad = True

            for max_count in range(max_optim_epoch):
                yhat, y_pred, y_gtx = transformer.forward(input_batch)

                optimizer.zero_grad()

                g_pred_loss = criterion(y_gtx, y_label_batch)
                f_pred_loss = criterion(y_star_batch, y_pred)
                f_pred_loss_wbias = criterion(y_star_batch, yhat)
                f_perm_loss = criterion(y_pred, yhat)
                loss = g_pred_loss + f_pred_loss + f_pred_loss_wbias + f_perm_loss

                if torch.isnan(loss):
                    print("encountered Nan")

                print(f"\t\tMax Optim Loop: {max_count + 1}/{max_optim_epoch} Loss: {loss.item()}")

                loss.backward()

                transformer.matlayer.weight.grad = -(transformer.matlayer.weight.grad)
                optimizer.step()

            for param in transformer.parameters():
                param.requires_grad = True

            for param in transformer.matlayer.parameters():
                param.requires_grad = False
                
            print('\n')

            for min_count in range(min_optim_epoch):
                yhat, y_pred, y_gtx = transformer.forward(input_batch)

                optimizer.zero_grad()

                g_pred_loss = criterion(y_gtx, y_label_batch)
                f_pred_loss = criterion(y_star_batch, y_pred)
                f_pred_loss_wbias = criterion(y_star_batch, yhat)
                f_perm_loss = criterion(y_pred, yhat)
                loss = g_pred_loss + f_pred_loss + f_pred_loss_wbias + f_perm_loss

                if torch.isnan(loss):
                    print("encountered Nan")

                print(f"\t\tMin Optim Loop: {min_count + 1}/{min_optim_epoch} Loss: {loss.item()}")

                loss.backward()
                optimizer.step()

            train_loss[i] = loss
            
            
        end_time = time.time()  # Time at the end of the epoch
        epoch_training_time = end_time - start_time  # Time taken for the current epoch
        total_training_time += epoch_training_time  # Accumulating the training time for each epoch

        # Logging the training loss to TensorBoard
        writer.add_scalar('Loss/Train', train_loss[i], i + 1)

        print(f"\ntrain_epoch: {i+1}; train_loss: {train_loss[i]}\n")
        print(f"\nTime taken for epoch {i + 1}: {epoch_training_time/60:.2f} minutes\n")

    print(f"Total training time: {total_training_time/60:.2f} minutes")

    # Logging the total training time
    writer.add_scalar('Total Training Time', total_training_time, 0)

    # Closing the writer
    writer.close()

    return train_loss

    
def plot(train_loss):
    epochs = range(len(train_loss))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss.detach().numpy(), marker='o', markersize=5, linestyle='--', color='b', label='Train Loss')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Train Loss', fontsize=14)
    plt.title('Training Loss over Epochs', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.savefig(f"training loss_{int((1-split_size)*100)}_{int(split_size*100)}_{train_epoch}.png")
    plt.show()



def load_main_data_from_files(split_size):

    with open(f"train_tensor_{int((1-split_size)*100)}_{int(split_size*100)}.pkl", 'rb') as f:
        train_tensor = pickle.load(f)

    with open(f"train_score_tensor_{int((1-split_size)*100)}_{int(split_size*100)}.pkl", 'rb') as f:
        train_score_tensor = pickle.load(f)

    with open(f"train_real_tensor_{int((1-split_size)*100)}_{int(split_size*100)}.pkl", 'rb') as f:
        train_real_tensor = pickle.load(f)

    with open('scaler_target.pkl', 'rb') as f:
        scaler_target = pickle.load(f)

    return train_tensor, train_score_tensor,train_real_tensor, scaler_target


split_size= 0.4

input_seq, y_label,y_star, scaler_target = load_main_data_from_files(split_size)

seq_len= input_seq.size()[1]
seq_dim= input_seq.size()[-1]
output_dim= y_label.size()[-1]

hidden_size = 16
g_embed_dim= 18
f_embed_dim= 13
num_layers = 2
num_heads = 4

dropout_rate = 0.1

max_optim_epoch = 5
min_optim_epoch = 5
train_epoch = 10

batch_size = 32 

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the data to the GPU
input_seq = input_seq.to(device)
y_star = y_star.to(device)
y_label = y_label.to(device)


def main():

    

    transformer = hbiascorrect(seq_len, seq_dim, g_embed_dim, f_embed_dim, output_dim,num_layers, num_heads, hidden_size, dropout_rate)

    
    dataset = TensorDataset(input_seq, y_label, y_star) 

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  

    
    # with open(f"final_model_parameters_{int((1-split_size)*100)}_{int(split_size*100)}.pkl", 'rb') as f:
    #     model_parameters = pickle.load(f)
    # transformer.load_state_dict(model_parameters)

    # Move the model to the GPU
    
    transformer.to(device)

    optimizer = optim.Adam(transformer.parameters(), lr=0.001)
    transformer.train()

    # Save the initial model parameters
    with open(f"initial_model_parameters_{int((1-split_size)*100)}_{int(split_size*100)}_{train_epoch}.pkl", 'wb') as f:
        pickle.dump(transformer.state_dict(), f)


        # Check if the tensors and the model are on the same device
    print(f"Model device: {next(transformer.parameters()).device}")
    print(f"input_seq device: {input_seq.device}")
    print(f"y_star device: {y_star.device}")
    print(f"y_label device: {y_label.device}")


    train_loss = train_model(transformer, dataloader, max_optim_epoch, min_optim_epoch, train_epoch, optimizer, split_size)

    # Save the final model parameters
    with open(f"final_model_parameters_{int((1-split_size)*100)}_{int(split_size*100)}_{train_epoch}.pkl", 'wb') as f:
        pickle.dump(transformer.state_dict(), f)

    # Save the training loss values
    with open(f"train_loss_values_{int((1-split_size)*100)}_{int(split_size*100)}_{train_epoch}.pkl", 'wb') as f:
        pickle.dump(train_loss, f)
    
    plot(train_loss)

    # test_model(transformer , test_tensor , test_real_tensor, scaler_target)

if __name__ == "__main__":
    main()

    
