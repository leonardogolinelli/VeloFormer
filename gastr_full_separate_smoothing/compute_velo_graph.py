import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt

adata = sc.read_h5ad("gastrulation_velocity.h5ad")

print("Computing neighbors")
sc.pp.neighbors(adata)
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
    plt.savefig(f"plots/gastrulation_velocity_{key}.png", bbox_inches="tight")
print("Saving anndata with velocity graph")
adata.write("gastrulation_velocity.h5ad")