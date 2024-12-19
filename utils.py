

import os
import scanpy as sc
import torch
import numpy as np
from model import NETWORK
import pickle



def manifold_and_neighbors(adata, n_components, n_knn_search, dataset_name, K, knn_rep, best_key, ve_layer):
    from sklearn.manifold import Isomap
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors

    MuMs = adata.obsm["MuMs"]

    print("computing isomap 1...")
    isomap = Isomap(n_components=n_components, n_neighbors=n_knn_search).fit_transform(MuMs)
    print("computing isomap 2..")
    isomap_unique = Isomap(n_components=1, n_neighbors=n_knn_search).fit_transform(MuMs)
    pca_runner = PCA(n_components=n_components)
    pca = pca_runner.fit_transform(MuMs)
    pca_unique = PCA(n_components=1).fit_transform(MuMs)
    adata.uns["PCA_weights"] = pca_runner.components_
    ve_path = f"/mnt/data2/home/leonardo/git/dim_reduction/12_july/embeddings/6layer_{dataset_name}_smooth_K_{ve_layer}.npy"
    #ve = np.load(f"../dim_reduction/outputs/saved_z_matrices/{dataset_name}_z{ve_layer[0]}.npy")
    ve = np.load(ve_path)
    print(f"ve shape: {ve.shape}")
    print(f"adata shape: {adata.shape}")
    for rep, name in zip([isomap, isomap_unique, pca, pca_unique, ve], ["isomap", "isomap_unique", "pca", "pca_unique", "ve"]):
        adata.obsm[name] = rep
        base_path = f"outputs/{dataset_name}/K{K}/embeddings/time_umaps/"
        os.makedirs(base_path, exist_ok=True)
        if name in ["isomap", "pca"]:
            fname = f"{name}_1"
            adata.obs[fname] = rep[:,0]
            #sc.pl.umap(adata, color=fname)
            #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

            if n_components > 1:
                fname = f"{name}_2"
                adata.obs[fname] = rep[:,1]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

                fname = f"{name}_3"
                adata.obs[fname] = rep[:,2]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

                fname = f"{name}_1+2"
                adata.obs[fname] = rep[:,0] + rep[:,1]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

                fname = f"{name}_1+3"
                adata.obs[fname] = rep[:,0] + rep[:,2]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

                fname = f"{name}_2+3"
                adata.obs[fname] = rep[:,1] + rep[:,2]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

    print(f"n_components: {n_components}")
    print(f"n_neighbors: {n_knn_search}")
    print(f"knn rep used: {knn_rep}")

    if knn_rep == "isomap":
        print("isomap key used")
        embedding = isomap

        if best_key:
            print("best key used")
            embedding = np.array(adata.obsm[best_key]).reshape(-1,1)

    elif knn_rep == "isomap_unique":
        print("isomap unique key used")
        embedding = isomap_unique
        
    elif knn_rep == "ve":
        print("ve key used")
        embedding = ve

    elif knn_rep == "pca":
        print("pca key used")
        embedding = pca

        if best_key:
            print("best key used")
            embedding = np.array(adata.obsm[best_key]).reshape(-1,1)
        
    nbrs = NearestNeighbors(n_neighbors=adata.shape[0], metric='euclidean')
    nbrs.fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    adata.uns["distances"] = distances
    adata.uns["indices"] = indices
