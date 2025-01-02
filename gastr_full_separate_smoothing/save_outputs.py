import torch
import numpy as np
import scanpy as sc
import sys, os
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from model import NETWORK  # Ensure that model.py is saved in the same directory
from dataloaders import * # Ensure that dataloaders.py is saved in the same directory
from utils import *

# Setup configuration
latent_dim = 64  # Latent dimension size, can be adjusted
hidden_dim = 512  # Hidden dimension size for the encoder and decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

n_components = 100
n_knn_search = 10
dataset_name = "gastrulation_erythroid"
cell_type_key = "celltype"
model_name = "VeloFormer"

num_genes = 2000
nhead = 1 #original: 1
embedding_dim = 128*nhead# original: 128
num_encoder_layers = 1 #original: 1
num_bins = 50
batch_size = 128  # Batch size for training
epochs = 10  # Number of epochs for training
learning_rate = 1e-4  # Learning rate for the optimizer
lambda1 = 1e-1  # Weight for heuristic loss
lambda2 = 1 # Weight for discrepancy loss
K = 11  # Number of neighbors for heuristic loss

# Load data
adata = sc.read_h5ad("gastrulation_processed.h5ad")


# Initialize model, optimizer, and loss function
model = NETWORK(input_dim=adata.shape[1]*2, latent_dim=latent_dim, 
                hidden_dim=hidden_dim, emb_dim = embedding_dim,
                nhead=nhead, num_encoder_layers=num_encoder_layers,
                num_genes=num_genes, num_bins=num_bins).to(device)
                
model.load_state_dict(torch.load('model.pth'))

# Ensure to call model.eval() if you're loading the model for inference to set the dropout and batch normalization layers to evaluation mode
model.eval()

_, _, full_data_loader = setup_dataloaders_binning(adata, 
                                                    batch_size=batch_size, 
                                                    num_genes=num_genes, 
                                                    num_bins=num_bins)

# Initialize empty layers in adata for storing results
adata.layers["velocity_u"] = np.zeros_like(adata.layers["Mu"], dtype=np.float32)
adata.layers["velocity"] = np.zeros_like(adata.layers["Ms"], dtype=np.float32)
adata.obsm["pred"] = np.zeros((adata.shape[0], adata.shape[1] * 2), dtype=np.float32)
adata.obsm["cell_embeddings"] = np.zeros((adata.shape[0], adata.shape[1] * 2), dtype=np.float32)
adata.layers["pp"] = np.zeros_like(adata.layers["Mu"])  # Same shape as Mu
adata.layers["nn"] = np.zeros_like(adata.layers["Mu"])  # Same shape as Mu
adata.layers["pn"] = np.zeros_like(adata.layers["Mu"])  # Same shape as Mu
adata.layers["np"] = np.zeros_like(adata.layers["Mu"])  # Same shape as Mu
model.eval()
with torch.no_grad():
    for batch_idx, (tokens, data, batch_indices) in enumerate(full_data_loader):
        print(f"Batch {batch_idx+1}/{len(full_data_loader)}")
        tokens = tokens.to(device)
        data = data.to(device)
        out_dic = model(tokens, data)

        # Store results and convert to numpy inside the loop to reduce peak memory usage
        adata.layers["velocity_u"][batch_indices] = out_dic["v_u"].detach().cpu().numpy()
        adata.layers["velocity"][batch_indices] = out_dic["v_s"].detach().cpu().numpy()
        adata.obsm["pred"][batch_indices] = out_dic["pred"].detach().cpu().numpy()
        adata.obsm["cell_embeddings"][batch_indices] = out_dic["cell_embeddings"].detach().cpu().numpy()
        adata.layers["pp"][batch_indices] = out_dic["pp"].cpu().numpy()
        adata.layers["nn"][batch_indices] = out_dic["nn"].cpu().numpy()
        adata.layers["pn"][batch_indices] = out_dic["pn"].cpu().numpy()
        adata.layers["np"][batch_indices] = out_dic["np"].cpu().numpy()

        # Explicit memory cleanup
        del tokens, data, out_dic
        gc.collect()
        torch.cuda.empty_cache()  # If using CUDA
    

adata.write("gastrulation_velocity.h5ad")