import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import torch.nn.functional as F

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

class BinnedExpressionDatasetMinMax(Dataset):
    def __init__(self, adata, num_genes, num_bins):
        self.data = adata
        self.num_genes = num_genes
        self.num_bins = num_bins
        self.min_values = np.min(adata.layers["Mu"], axis=0)
        self.max_values = np.max(adata.layers["Mu"], axis=0)

    def bin_expression_values(self, expression_values):
        """
        Bin gene expression values into discrete bins.
        """
        binned_values = torch.floor(expression_values / (torch.max(expression_values) / self.num_bins)).clamp(0, self.num_bins - 1)
        return binned_values.to(dtype=torch.long)

    def normalize(self, values, min_values, max_values):
        """
        Normalize gene expression values using min-max normalization.
        """
        return (values - min_values) / (max_values - min_values)

    def __len__(self):
        return self.data.n_obs

    def __getitem__(self, idx):
        unspliced = torch.tensor(self.data.layers["Mu"][idx], dtype=torch.float32)
        spliced = torch.tensor(self.data.layers["Ms"][idx], dtype=torch.float32)
        
        # Normalize the expression values
        unspliced_minmax = self.normalize(unspliced, self.min_values, self.max_values)
        spliced_minmax = self.normalize(spliced, self.min_values, self.max_values)
        
        combined_minmax = torch.cat([unspliced_minmax, spliced_minmax])
        combined = torch.cat([unspliced, spliced])

        # Bin the expression values
        binned_indices = self.bin_expression_values(combined_minmax)
        
        return binned_indices, combined, idx


def setup_dataloaders_binning_MinMax(adata, batch_size, train_size=0.8, split_data=True,
                      num_genes=2000, num_bins=10):
    custom_dataset = BinnedExpressionDatasetMinMax(adata, num_genes, num_bins)
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



class BinnedExpressionDatasetPancreas(Dataset):
    def __init__(self, adata):
        self.data = adata
        self.gastrulation_var_names = pd.Index(adata.uns["gastrulation_var_names"])
        self.num_genes = 2000  # Total genes in gastrulation (expected by the model)

        # Initialize placeholders for bin ranges
        self.precomputed_bin_ranges = torch.zeros(self.num_genes * 2, dtype=torch.float32)

        # Fetch the gastrulation gene names and original bin ranges
        original_bin_ranges = torch.tensor(adata.uns["bin_ranges"], dtype=torch.float32)

        # Map gastrulation gene names to indices in the pancreas dataset
        pancreas_gene_index = {gene: idx for idx, gene in enumerate(adata.var_names)}
        for i, gene in enumerate(self.gastrulation_var_names):
            if gene in pancreas_gene_index:
                idx = pancreas_gene_index[gene]
                # Set the bin ranges for both unspliced and spliced
                self.precomputed_bin_ranges[2 * idx] = original_bin_ranges[i]
                self.precomputed_bin_ranges[2 * idx + 1] = original_bin_ranges[i]

    def bin_expression_values(self, expression_values):
        # Pad expression values if necessary to match the model's expected input size
        if expression_values.size(0) < 4000:
            expression_values = F.pad(expression_values, (0, 4000 - expression_values.size(0)), "constant", 0)
        # Compute binned values
        binned_values = (expression_values / self.precomputed_bin_ranges).clamp(0, 49).floor()
        return binned_values.to(dtype=torch.long)

    def __getitem__(self, idx):
        unspliced = torch.tensor(self.data.layers["Mu"][idx], dtype=torch.float32)
        spliced = torch.tensor(self.data.layers["Ms"][idx], dtype=torch.float32)
        combined = torch.cat([unspliced, spliced])

        # Pad the combined tensor to the full length expected by the model
        if combined.size(0) < 4000:
            combined = F.pad(combined, (0, 4000 - combined.size(0)), "constant", 0)

        binned_indices = self.bin_expression_values(combined)
        return binned_indices, combined, idx

    def __len__(self):
        return self.data.n_obs


def setup_dataloaders_binning_pancreas(adata, batch_size, train_size=0.8, split_data=True,
                      num_genes=2000, num_bins=10):
    custom_dataset = BinnedExpressionDatasetPancreas(adata)
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



class BinnedExpressionDatasetSimpler(Dataset):
    def __init__(self, adata, num_bins=50):
        """
        Simplified Dataset for aligned AnnData objects.

        Args:
            adata: Aligned AnnData object with consistent var_names.
            bin_ranges: Precomputed bin ranges for gene expression values.
            num_bins: Number of bins to discretize expression values.
        """
        self.data = adata
        self.num_genes = len(adata.var_names)  # Total genes (aligned)
        self.bin_ranges = torch.tensor(adata.uns["bin_ranges"], dtype=torch.float32)
        self.precomputed_bin_ranges = torch.tensor(self.bin_ranges, dtype=torch.float32)
        self.num_bins = num_bins

    def bin_expression_values(self, expression_values):
        """
        Bin gene expression values into discrete bins using precomputed ranges.
        """
        binned_values = (expression_values / self.precomputed_bin_ranges).clamp(0, self.num_bins - 1).floor()
        return binned_values.to(dtype=torch.long)

    def __getitem__(self, idx):
        """
        Fetch and bin expression values for the specified index.
        """
        # Fetch unspliced and spliced values
        unspliced = torch.tensor(self.data.layers["Mu"][idx], dtype=torch.float32)
        spliced = torch.tensor(self.data.layers["Ms"][idx], dtype=torch.float32)
        combined = torch.cat([unspliced, spliced])  # Combined unspliced + spliced

        # Compute binned indices
        binned_indices = self.bin_expression_values(combined)
        return binned_indices, combined, idx

    def __len__(self):
        return self.data.n_obs


def setup_dataloaders_binning_simpler(adata, batch_size, train_size=0.8, split_data=True,
                      num_genes=2000, num_bins=10):
    custom_dataset = BinnedExpressionDatasetSimpler(adata, num_bins)
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