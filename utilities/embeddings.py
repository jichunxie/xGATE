import numpy as np
import pandas as pd
import igraph as ig
import random
import torch
import torch.nn as nn
from scipy.linalg import expm
from .pathway_analysis import normalize
from .vae_model import VariationalAutoencoder, vae_loss_function, calculate_reconstruction_error

def longest_random_walk(G, start_node_index, num_walks = 200, max_walk_length = 200):
    max_length = 0
    for _ in range(num_walks):
        visited = set()
        current_node = start_node_index
        previous_node = None
        length = 0

        while length < max_walk_length:
            if current_node in visited:
                break

            visited.add(current_node)
            neighbors = [n for n in G.neighbors(current_node) if n != previous_node]

            if not neighbors and G.degree(current_node) == 1:
                neighbors = [previous_node] if previous_node is not None else []

            if not neighbors:
                break

            next_node = np.random.choice(neighbors)
            previous_node = current_node
            current_node = next_node
            length += 1

        max_length = max(max_length, length)

    return max_length

def subgraph_centrality(G):
    A = np.array(G.get_adjacency().data, dtype=float)
    expA = expm(A)
    return np.diag(expA)


def generate_embedding(G):
    harmonic_centralities = G.harmonic_centrality()
    nodes_sorted_by_centrality = sorted(G.vs, key=lambda node: harmonic_centralities[node.index], reverse=True)
    embedding = []

    for node in nodes_sorted_by_centrality:
        longest_walk = longest_random_walk(G, node.index)
        strength = sum(G.es[G.incident(node.index)]['weight'])
        eccentricity = G.eccentricity()[node.index]
        embedding.extend([longest_walk, strength, eccentricity])

    return embedding


def embedding_recon(G, categorized_pathways, pathway_genes, num_walks, max_walk_length, null_dist_size):
    node_names = set(v["name"] for v in G.vs)
    found_genes = [gene for gene in pathway_genes if gene in node_names]

    fraction = len(found_genes) / max(1, len(pathway_genes))

    pathway_subgraph_s = G.subgraph([v.index for v in G.vs if v["name"] in found_genes])

    neg_subgraphs_pathway_s = []
    all_nodes = set(v["name"] for v in G.vs)
    num_nodes = pathway_subgraph_s.vcount()

    for _ in range(null_dist_size):
        vertices = random.sample(all_nodes, k=num_nodes)
        neg_subgraph = G.subgraph([v.index for v in G.vs if v["name"] in vertices])
        neg_subgraphs_pathway_s.append(neg_subgraph)

    random_subgraphs_pathway_s = []

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

    return p_value, z_score




