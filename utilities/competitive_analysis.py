import numpy as np
import pandas as pd
import igraph as ig
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm
from .pathway_analysis import normalize, gather_pathways_between, get_genes_in_pathway
from .vae_model import VariationalAutoencoder, vae_loss_function, calculate_reconstruction_error
from .embeddings import generate_embedding


def embedding_recon_competitive(
    G, 
    categorized_pathways, 
    pathway_genes, 
    num_walks, 
    max_walk_length, 
    null_dist_size,
    neg_subgraphs_pathway_s=None, 
    random_subgraphs_pathway_s=None,
    plot=False,
    pathway_name="Pathway"
):
    """
    Perform embedding reconstruction for a given pathway.
    
    Parameters:
    - G: The graph.
    - categorized_pathways: Categorized pathways.
    - pathway_genes: Genes in the pathway.
    - num_walks: Number of walks (unused in current context).
    - max_walk_length: Maximum walk length (unused in current context).
    - null_dist_size: Number of null distributions.
    - neg_subgraphs_pathway_s: Pre-generated list of negative subgraphs (optional).
    - random_subgraphs_pathway_s: Pre-generated list of random subgraphs (optional).
    - plot: Whether to plot the reconstruction error distributions.
    - pathway_name: Name of the pathway (for labeling plots).
    
    Returns:
    - Dictionary containing reconstruction errors, p-value, z-score, and subgraphs.
    """

    node_names = set(v["name"] for v in G.vs)
    found_genes = [gene for gene in pathway_genes if gene in node_names]

    fraction = len(found_genes) / max(1, len(pathway_genes))

    pathway_subgraph_s = G.subgraph([v.index for v in G.vs if v["name"] in found_genes])

    if neg_subgraphs_pathway_s is None:
        neg_subgraphs_pathway_s = []
        all_nodes = set(v["name"] for v in G.vs)
        num_nodes = pathway_subgraph_s.vcount()

        for _ in range(null_dist_size):
            vertices = random.sample(all_nodes, k=num_nodes)
            neg_subgraph = G.subgraph([v.index for v in G.vs if v["name"] in vertices])
            neg_subgraphs_pathway_s.append(neg_subgraph)

    if random_subgraphs_pathway_s is None:
        random_subgraphs_pathway_s = []
        all_nodes = set(v["name"] for v in G.vs)
        num_nodes = pathway_subgraph_s.vcount()

        for _ in range(null_dist_size):
            vertices = random.sample(all_nodes, k=num_nodes)
            random_subgraph = G.subgraph([v.index for v in G.vs if v["name"] in vertices])
            random_subgraphs_pathway_s.append(random_subgraph)

    embeddings_neg_pathway_s = [generate_embedding(subgraph) for subgraph in neg_subgraphs_pathway_s]
    embeddings_random_pathway_s = [generate_embedding(subgraph) for subgraph in random_subgraphs_pathway_s]
    embedding_pathway_s = generate_embedding(pathway_subgraph_s)

    df_neg = pd.DataFrame(embeddings_neg_pathway_s)
    df_random = pd.DataFrame(embeddings_random_pathway_s)
    df_pathway = pd.DataFrame([embedding_pathway_s])

    df_all = pd.concat([df_neg, df_random, df_pathway], axis=0).reset_index(drop=True)
    df_normalized = df_all.apply(normalize, axis=0)

    embeddings_neg_pathway_s = df_normalized.iloc[:null_dist_size].values.tolist()
    embeddings_random_pathway_s = df_normalized.iloc[null_dist_size:2*null_dist_size].values.tolist()
    embedding_pathway_s = df_normalized.iloc[-1].values.tolist()

    num_epochs = 1000
    embedding_dim = len(embedding_pathway_s)
    latent_dim = 16
    model = VariationalAutoencoder(embedding_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    embeddings_tensor = torch.Tensor(embeddings_neg_pathway_s)

    for epoch in range(num_epochs):
        model.zero_grad()
        recon_batch, mu, logvar = model(embeddings_tensor)
        loss = vae_loss_function(recon_batch, embeddings_tensor, mu, logvar)
        loss.backward()
        optimizer.step()

    reconstruction_errors_pathway_s = calculate_reconstruction_error([embedding_pathway_s], model, vae_loss_function)
    reconstruction_errors_neg_pathway_s = calculate_reconstruction_error(embeddings_neg_pathway_s, model, vae_loss_function)
    reconstruction_errors_random_pathway_s = calculate_reconstruction_error(embeddings_random_pathway_s, model, vae_loss_function)

    error_random = np.array(reconstruction_errors_random_pathway_s)
    error_pathway = np.array(reconstruction_errors_pathway_s)

    # Add a pseudocount by including the pathway error in the random errors array
    error_random_with_pseudocount = np.append(error_random, error_pathway)

    p_value = np.mean(error_random_with_pseudocount >= error_pathway)

    # Calculate Z-Score
    mean_random = np.mean(error_random)
    std_random = np.std(error_random)
    if std_random > 0:
        z_score = (error_pathway[0] - mean_random) / std_random
    else:
        z_score = np.nan  # Handle the case where standard deviation is zero
        
    # Optional Plotting
    if plot:
        plt.figure(figsize=(10, 6))
        
        # Define a common bin range based on all reconstruction errors
        max_error = max(
            np.max(error_random) if len(error_random) > 0 else 0,
            np.max(reconstruction_errors_neg_pathway_s) if len(reconstruction_errors_neg_pathway_s) > 0 else 0,
            error_pathway[0] if len(error_pathway) > 0 else 0
        )
        bins = np.arange(0, max_error + 0.05, 0.001)

        # Plot Negative Subgraphs Reconstruction Errors
        plt.hist(reconstruction_errors_neg_pathway_s, bins=bins, color='red', label="Negative Subgraphs", alpha=0.5)

        # Plot Random Subgraphs Reconstruction Errors
        plt.hist(reconstruction_errors_random_pathway_s, bins=bins, color='blue', label="Random Subgraphs", alpha=0.5)

        # Plot Pathway Reconstruction Error
        plt.hist([error_pathway[0]], bins=bins, color='green', label=f"{pathway_name} Pathway", alpha=0.7, edgecolor='black')

        plt.annotate(f'{pathway_name} Pathway', 
                     xy=(error_pathway[0], 1), 
                     xytext=(error_pathway[0] + 0.05, plt.ylim()[1]*0.8),
                     arrowprops=dict(facecolor='green', shrink=0.05))

        plt.title(f"Reconstruction Errors for {pathway_name}")
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "reconstruction_errors_random": reconstruction_errors_random_pathway_s,
        "reconstruction_errors_pathway": reconstruction_errors_pathway_s,
        "p_value": p_value,
        "z_score": z_score,
        "neg_subgraphs_pathway_s": neg_subgraphs_pathway_s,
        "random_subgraphs_pathway_s": random_subgraphs_pathway_s
    } 

def competitive_pathway_analysis(
    G, 
    categorized_pathways, 
    pathway1, 
    pathway2, 
    num_walks=200, 
    max_walk_length=200, 
    null_dist_size=200,
    plot=False
):
    """
    Perform competitive pathway analysis between two pathways using T-statistic based p-values.
    
    Parameters:
    - G: The graph (igraph.Graph object).
    - categorized_pathways: Dictionary categorizing pathways to their respective genes.
    - pathway1: Name of the first pathway.
    - pathway2: Name of the second pathway.
    - num_walks: Number of walks for embedding reconstruction.
    - max_walk_length: Maximum walk length for embedding reconstruction.
    - null_dist_size: Number of null distributions.
    - plot: Whether to plot the reconstruction error distributions.
    
    Returns:
    - DataFrame containing the competitive analysis results including competitive p-values based on T-statistics.
    """
    
    results = []
    
    # Gather pathways and get genes for pathway1
    pathway_list1 = gather_pathways_between(pathway1, pathway1, categorized_pathways)
    genes1 = get_genes_in_pathway(pathway_list1)
    
    # Perform embedding reconstruction for pathway1
    recon1 = embedding_recon_competitive(
        G, categorized_pathways, genes1, num_walks, max_walk_length, null_dist_size, plot=plot, pathway_name=pathway1
    )
    errors_random1 = np.array(recon1["reconstruction_errors_random"])
    error_pathway1 = recon1["reconstruction_errors_pathway"][0]
    
    # Store pathway1's negative and random subgraphs
    neg_subgraphs_p1 = recon1["neg_subgraphs_pathway_s"]
    random_subgraphs_p1 = recon1["random_subgraphs_pathway_s"]
    
    # Gather pathways and get genes for pathway2
    pathway_list2 = gather_pathways_between(pathway2, pathway2, categorized_pathways)
    genes2 = get_genes_in_pathway(pathway_list2)
    
    # Create the subgraph for pathway2
    node_names = set(v["name"] for v in G.vs)
    found_genes2 = [gene for gene in genes2 if gene in node_names]
    pathway_subgraph_s2 = G.subgraph([v.index for v in G.vs if v["name"] in found_genes2])
    num_nodes_p2 = pathway_subgraph_s2.vcount()
    
    # Get number of nodes in pathway1's subgraphs (assume all are the same; use the first)
    num_nodes_p1 = len(neg_subgraphs_p1[0].vs) if neg_subgraphs_p1 else 0
    
    if num_nodes_p1 == 0:
        raise ValueError(f"Pathway '{pathway1}' has no genes present in the graph.")
    
    # Adjust subgraphs for pathway2 to have the same number of nodes as pathway1
    if num_nodes_p2 > num_nodes_p1:
        diff = num_nodes_p2 - num_nodes_p1
        adjusted_neg_subgraphs_p2 = []
        adjusted_random_subgraphs_p2 = []
        all_nodes = set(v["name"] for v in G.vs)
        for idx, neg_sub in enumerate(neg_subgraphs_p1):
            available_nodes_neg = list(all_nodes - set(neg_sub.vs["name"]))
            if len(available_nodes_neg) < diff:
                raise ValueError(f"Not enough available nodes to add to negative subgraph {idx} for pathway2. Required: {diff}, Available: {len(available_nodes_neg)}")
            nodes_to_add_neg = random.sample(available_nodes_neg, k=diff)
            new_nodes_neg = neg_sub.vs["name"] + nodes_to_add_neg
            adjusted_neg_sub = G.subgraph([v.index for v in G.vs if v["name"] in new_nodes_neg])
            assert adjusted_neg_sub.vcount() == num_nodes_p2, f"Adjusted negative subgraph {idx} size {adjusted_neg_sub.vcount()} does not match expected {num_nodes_p2}."
            adjusted_neg_subgraphs_p2.append(adjusted_neg_sub)
        for idx, rand_sub in enumerate(random_subgraphs_p1):
            available_nodes_rand = list(all_nodes - set(rand_sub.vs["name"]))
            if len(available_nodes_rand) < diff:
                raise ValueError(f"Not enough available nodes to add to random subgraph {idx} for pathway2. Required: {diff}, Available: {len(available_nodes_rand)}")
            nodes_to_add_rand = random.sample(available_nodes_rand, k=diff)
            new_nodes_rand = rand_sub.vs["name"] + nodes_to_add_rand
            adjusted_rand_sub = G.subgraph([v.index for v in G.vs if v["name"] in new_nodes_rand])
            assert adjusted_rand_sub.vcount() == num_nodes_p2, f"Adjusted random subgraph {idx} size {adjusted_rand_sub.vcount()} does not match expected {num_nodes_p2}."
            adjusted_random_subgraphs_p2.append(adjusted_rand_sub)
    elif num_nodes_p2 < num_nodes_p1:
        diff = num_nodes_p1 - num_nodes_p2
        adjusted_neg_subgraphs_p2 = []
        adjusted_random_subgraphs_p2 = []
        for idx, neg_sub in enumerate(neg_subgraphs_p1):
            if len(neg_sub.vs) < diff:
                raise ValueError(f"Cannot remove {diff} nodes from negative subgraph {idx} with {len(neg_sub.vs)} nodes.")
            nodes_to_remove_neg = random.sample(neg_sub.vs["name"], k=diff)
            new_nodes_neg = [name for name in neg_sub.vs["name"] if name not in nodes_to_remove_neg]
            adjusted_neg_sub = G.subgraph([v.index for v in G.vs if v["name"] in new_nodes_neg])
            assert adjusted_neg_sub.vcount() == num_nodes_p2, f"Adjusted negative subgraph {idx} size {adjusted_neg_sub.vcount()} does not match expected {num_nodes_p2}."
            adjusted_neg_subgraphs_p2.append(adjusted_neg_sub)
        for idx, rand_sub in enumerate(random_subgraphs_p1):
            if len(rand_sub.vs) < diff:
                raise ValueError(f"Cannot remove {diff} nodes from random subgraph {idx} with {len(rand_sub.vs)} nodes.")
            nodes_to_remove_rand = random.sample(rand_sub.vs["name"], k=diff)
            new_nodes_rand = [name for name in rand_sub.vs["name"] if name not in nodes_to_remove_rand]
            adjusted_rand_sub = G.subgraph([v.index for v in G.vs if v["name"] in new_nodes_rand])
            assert adjusted_rand_sub.vcount() == num_nodes_p2, f"Adjusted random subgraph {idx} size {adjusted_rand_sub.vcount()} does not match expected {num_nodes_p2}."
            adjusted_random_subgraphs_p2.append(adjusted_rand_sub)
    else:
        adjusted_neg_subgraphs_p2 = neg_subgraphs_p1.copy()
        adjusted_random_subgraphs_p2 = random_subgraphs_p1.copy()
    
    # Perform embedding reconstruction for pathway2 using the adjusted subgraphs
    recon2 = embedding_recon_competitive(
        G, 
        categorized_pathways, 
        genes2, 
        num_walks, 
        max_walk_length, 
        null_dist_size,
        neg_subgraphs_pathway_s=adjusted_neg_subgraphs_p2, 
        random_subgraphs_pathway_s=adjusted_random_subgraphs_p2, 
        plot= plot,
        pathway_name=pathway2
    )
    errors_random2 = np.array(recon2["reconstruction_errors_random"])
    error_pathway2 = recon2["reconstruction_errors_pathway"][0]
    
    # Compute variances for the random reconstruction errors (using ddof=1)
    var1 = np.var(errors_random1, ddof=1)
    var2 = np.var(errors_random2, ddof=1)
    
    # Compute T statistics for competitive analysis
    T1 = (error_pathway1 - error_pathway2) / np.sqrt(var1 + var2)
    T2 = (error_pathway2 - error_pathway1) / np.sqrt(var1 + var2)
    
    # Compute competitive p-values from the T statistics
    competitive_p_value1 = 1 - norm.cdf(T1)
    competitive_p_value2 = 1 - norm.cdf(T2)
    
    # Prepare the results with the competitive p-values
    result_entry = {
        "Pathway 1": pathway1,
        "reconstruction_error1": error_pathway1,
        f"{pathway1} Competitive P-value": competitive_p_value1,
        "Pathway 2": pathway2,
        "reconstruction_error2": error_pathway2,
        f"{pathway2} Competitive P-value": competitive_p_value2,
    }
    
    results.append(result_entry)
    results_df = pd.DataFrame(results)
    return results_df

