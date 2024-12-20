import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import scanpy as sc
import math

class CustomDataset(Dataset):
    def __init__(self, adata):
        unspliced = torch.tensor(adata.layers["Mu"], dtype=torch.float32)
        spliced = torch.tensor(adata.layers["Ms"], dtype=torch.float32)
        self.x = torch.cat([unspliced, spliced], dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], idx
    
class RankedExpressionDataset(Dataset):
    def __init__(self, adata, num_genes, embedding_dim):
        self.data = adata
        self.num_genes = num_genes
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(2*num_genes, embedding_dim)  # 2*num_genes for unspliced and spliced
        # Creating fixed positional embeddings based on rank
        self.pos_embeddings = self.generate_positional_embeddings(2*num_genes, embedding_dim)

    def generate_positional_embeddings(self, num_positions, dim):
        pos_embedding = torch.zeros(num_positions, dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_embedding

    def __len__(self):
        return self.data.n_obs

    def __getitem__(self, idx):
        unspliced = torch.tensor(self.data.layers["Mu"][idx], dtype=torch.float32)
        spliced = torch.tensor(self.data.layers["Ms"][idx], dtype=torch.float32)
        combined = torch.cat([unspliced, spliced])
        ranked_indices = torch.argsort(combined, descending=True)
        tokens = self.embeddings(ranked_indices)
        # Adding rank-based positional embeddings
        tokens += self.pos_embeddings[:len(ranked_indices)]
        return tokens, combined, idx


class BinnedExpressionDataset(Dataset):
    def __init__(self, adata, num_genes, embedding_dim, num_bins):
        self.data = adata
        self.num_genes = num_genes
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins
        self.embeddings = nn.Embedding(2 * num_genes, embedding_dim)  # 2*num_genes for unspliced and spliced

    def bin_expression_values(self, expression_values):
        """
        Bin gene expression values into discrete bins.
        """
        binned_values = torch.floor(expression_values / (torch.max(expression_values) / self.num_bins)).clamp(0, self.num_bins - 1)
        return binned_values.to(dtype=torch.long)

    def __len__(self):
        return self.data.n_obs

    def __getitem__(self, idx):
        unspliced = torch.tensor(self.data.layers["Mu"][idx], dtype=torch.float32)
        spliced = torch.tensor(self.data.layers["Ms"][idx], dtype=torch.float32)
        combined = torch.cat([unspliced, spliced])
        
        # Bin the expression values
        binned_indices = self.bin_expression_values(combined)
        
        # Generate token embeddings based on binned indices
        tokens = self.embeddings(binned_indices)
        
        return tokens, combined, idx
    

def setup_dataloaders_ranked(adata, batch_size, train_size=0.8, split_data=True,
                      num_genes=2000, embedding_dim=128):
    custom_dataset = RankedExpressionDataset(adata, num_genes, embedding_dim)
    if split_data:
        num_samples = len(custom_dataset)
        indices = np.random.permutation(num_samples)
        split = int(train_size * num_samples)
        train_indices, test_indices = indices[:split], indices[split:]

        train_subset = Subset(custom_dataset, train_indices)
        test_subset = Subset(custom_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    else:
        train_subset = custom_dataset
        test_loader = None

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    full_data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)  # Simplified DataLoader

    return train_loader, test_loader, full_data_loader

def setup_dataloaders_binning(adata, batch_size, train_size=0.8, split_data=True,
                      num_genes=2000, embedding_dim=128, num_bins=10):
    custom_dataset = BinnedExpressionDataset(adata, num_genes, embedding_dim, num_bins)
    if split_data:
        num_samples = len(custom_dataset)
        indices = np.random.permutation(num_samples)
        split = int(train_size * num_samples)
        train_indices, test_indices = indices[:split], indices[split:]

        train_subset = Subset(custom_dataset, train_indices)
        test_subset = Subset(custom_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    else:
        train_subset = custom_dataset
        test_loader = None

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    full_data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)  # Simplified DataLoader

    return train_loader, test_loader, full_data_loader


def setup_dataloaders(adata, batch_size, train_size=0.8, split_data=True):
    custom_dataset = CustomDataset(adata)
    if split_data:
        num_samples = len(custom_dataset)
        indices = np.random.permutation(num_samples)
        split = int(train_size * num_samples)
        train_indices, test_indices = indices[:split], indices[split:]

        train_subset = Subset(custom_dataset, train_indices)
        test_subset = Subset(custom_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    else:
        train_subset = custom_dataset
        test_loader = None

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    full_data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)  # Simplified DataLoader

    return train_loader, test_loader, full_data_loader


