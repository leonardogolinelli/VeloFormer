

import os
import scanpy as sc
import torch
import numpy as np
from model import NETWORK
import pickle
import pandas as pd
import seaborn as sns



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


import matplotlib.pyplot as plt

def plot_phase_plane(adata, gene_name, dataset, K, u_scale=.01, s_scale=0.01, alpha=0.5, head_width=0.02, head_length=0.03, length_includes_head=False, log=False,
                        norm_velocity=True, filter_cells=False, smooth_expr=True, show_plot=True, save_plot=True, save_path=".",
                        cell_type_key="clusters"):

    if smooth_expr:
        unspliced_expression = adata.layers["Mu"][:, adata.var_names.get_loc(gene_name)].flatten() 
        spliced_expression = adata.layers["Ms"][:, adata.var_names.get_loc(gene_name)].flatten() 
    else:
        unspliced_expression = adata.layers["unspliced"][:, adata.var_names.get_loc(gene_name)].flatten()
        spliced_expression = adata.layers["spliced"][:, adata.var_names.get_loc(gene_name)].flatten()

    # Normalize the expression data
    unspliced_expression_min, unspliced_expression_max = np.min(unspliced_expression), np.max(unspliced_expression)
    spliced_expression_min, spliced_expression_max = np.min(spliced_expression), np.max(spliced_expression)

    # Min-Max normalization
    unspliced_expression = (unspliced_expression - unspliced_expression_min) / (unspliced_expression_max - unspliced_expression_min)
    spliced_expression = (spliced_expression - spliced_expression_min) / (spliced_expression_max - spliced_expression_min)

    # Extract the velocity data
    unspliced_velocity = adata.layers['velocity_u'][:, adata.var_names.get_loc(gene_name)].flatten()
    spliced_velocity = adata.layers['velocity'][:, adata.var_names.get_loc(gene_name)].flatten()

    def custom_scale(data):
        max_abs_value = np.max(np.abs(data))  # Find the maximum absolute value
        scaled_data = data / max_abs_value  # Scale by the maximum absolute value
        return scaled_data

    if norm_velocity:
        unspliced_velocity = custom_scale(unspliced_velocity)
        spliced_velocity = custom_scale(spliced_velocity)


    # Apply any desired transformations (e.g., log) here
    if log:
        # Apply log transformation safely, ensuring no log(0)
        unspliced_velocity = np.log1p(unspliced_velocity)
        spliced_velocity = np.log1p(spliced_velocity)

    # Generate boolean masks for conditions and apply them
    if filter_cells:
        valid_idx = (unspliced_expression > 0) & (spliced_expression > 0)
    else:
        valid_idx = (unspliced_expression >= 0) & (spliced_expression >= 0)

    # Filter data based on valid_idx
    unspliced_expression_filtered = unspliced_expression[valid_idx]
    spliced_expression_filtered = spliced_expression[valid_idx]
    unspliced_velocity_filtered = unspliced_velocity[valid_idx]
    spliced_velocity_filtered = spliced_velocity[valid_idx]

    # Also filter cell type information to match the filtered expressions
    # First, get unique cell types and their corresponding colors
    unique_cell_types = adata.obs[cell_type_key].cat.categories
    celltype_colors = adata.uns[f"{cell_type_key}_colors"]
    
    # Create a mapping of cell type to its color
    celltype_to_color = dict(zip(unique_cell_types, celltype_colors))

    # Filter cell types from the data to get a list of colors for the filtered data points
    cell_types_filtered = adata.obs[cell_type_key][valid_idx]
    colors = cell_types_filtered.map(celltype_to_color).to_numpy()
    plt.figure(figsize=(9, 6.5), dpi=100)
  # Lower dpi here if the file is still too large    scatter = plt.scatter(unspliced_expression_filtered, spliced_expression_filtered, c=colors, alpha=0.6)

    """# Plot velocity vectors
    for i in range(len(unspliced_expression_filtered)):
        cell_type_index = np.where(unique_cell_types == cell_types_filtered[i])[0][0]
        arrow_color = celltype_to_color[cell_types_filtered[i]]  # Use the color corresponding to the cell type
        plt.arrow(
            unspliced_expression_filtered[i], spliced_expression_filtered[i], 
            unspliced_velocity_filtered[i] * u_scale, spliced_velocity_filtered[i] * s_scale, 
            color=arrow_color, alpha=alpha, head_width=head_width, head_length=head_length, length_includes_head=length_includes_head
        )"""

    # Plot velocity vectors
    for i in range(len(unspliced_expression_filtered)):
        cell_type_index = np.where(unique_cell_types == cell_types_filtered[i])[0][0]
        arrow_color = celltype_to_color[cell_types_filtered[i]]  # Use the color corresponding to the cell type
        plt.arrow(
            spliced_expression_filtered[i], unspliced_expression_filtered[i], 
            spliced_velocity_filtered[i] * s_scale, unspliced_velocity_filtered[i] * u_scale, 
            color=arrow_color, alpha=alpha, head_width=head_width, head_length=head_length, length_includes_head=length_includes_head
        )

    plt.ylabel(f'Normalized Unspliced Expression of {gene_name}')
    plt.xlabel(f'Normalized Spliced Expression of {gene_name}')
    plt.title(f'Expression and Velocity of {gene_name} by Cell Type')

    # Create a legend
    patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=celltype_to_color[celltype], markersize=10, label=celltype) 
            for celltype in unique_cell_types]
    plt.legend(handles=patches, title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')

    

    if save_plot:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Check if show_plot is True, then display the plot
    if show_plot:
        plt.show()
    else:
        plt.close()

    plt.show()

def color_keys(adata, cell_type_key):
    adata.obs[cell_type_key] = [str(cat) for cat in list(adata.obs[cell_type_key])]
    adata.obs[cell_type_key] = pd.Series(adata.obs[cell_type_key], dtype="category")
    unique_categories = adata.obs[cell_type_key].cat.categories
    rgb_colors = sns.color_palette("tab20", len(unique_categories))
    hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b in rgb_colors]
    adata.uns[f"{cell_type_key}_colors"] = hex_colors
    adata.layers['counts_unspliced'] = adata.layers["unspliced"].copy()
    adata.layers['counts_spliced'] = adata.layers["spliced"].copy()
    return adata