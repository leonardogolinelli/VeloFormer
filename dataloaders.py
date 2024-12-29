import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

class BinnedExpressionDataset(Dataset):
    def __init__(self, adata, num_genes, num_bins):
        self.data = adata
        self.num_genes = num_genes
        self.num_bins = num_bins

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
        
        return binned_indices, combined, idx

def setup_dataloaders_binning(adata, batch_size, train_size=0.8, split_data=True,
                      num_genes=2000, num_bins=10):
    custom_dataset = BinnedExpressionDataset(adata, num_genes, num_bins)
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
