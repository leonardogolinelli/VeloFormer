import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import numpy as np

adata = sc.read_h5ad("final_adata_10epochs.h5ad")

# Assuming `adata` is your original AnnData object
n_obs = adata.n_obs  # Total number of observations (cells)

# Calculate the number of observations to sample (half the total)
n_sample = n_obs // 2

# Randomly sample indices without replacement
random_indices = np.random.choice(n_obs, n_sample, replace=False)

# Subset the AnnData object
adata_half = adata[random_indices].copy()


print("Computing neighbors")
sc.pp.neighbors(adata)
sc.tl.umap(adata)
print("Reversing velocity")
adata.layers["velocity_u"] *= -1
adata.layers["velocity"] *= -1
print("Computing velocity graph")
scv.tl.velocity_graph(adata, n_jobs=20, mode_neighbors='distances', backend="threading")
print("Computing velocity confidence")
scv.tl.velocity_confidence(adata)
keys = ["celltype", "velocity_confidence", "velocity_length"]
for key in keys:
    print(f"Plotting {key}")
    sc.pl.umap(adata, color=key)
    #plt.savefig(f"plots/gastrulation_velocity_{key}.png", bbox_inches="tight")
    scv.pl.velocity_embedding_stream(adata, f"plots/stream_gastrulation_velocity_{key}.png", color=key)
    plt.savefig(f"plots/stream_gastrulation_velocity_{key}.png", bbox_inches="tight")

print("Saving anndata with velocity graph")
adata.write("gastrulation_velocity_10epochs_compute_umap.h5ad")