import scvelo as scv

adata = scv.datasets.gastrulation()
adata.write_h5ad("gastrulation_unprocessed.h5ad")