import numpy as np
import pandas as pd
import mygene
import networkx as nx
import igraph as ig
import warnings
from scipy.sparse import issparse
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import norm
from Bio.KEGG import REST
from Bio.KEGG.KGML import KGML_parser
from collections import defaultdict
from .data_processing import EstNull, norm_FDR_SQAUC, create_sifinet_object, filter_lowexp

def cal_coexp(so, X, X_full):
    p = X.shape[1]  # number of columns (genes)
    n = X.shape[0]  # number of rows (cells)
    
    # Initialize q as a vector of length p
    q = np.zeros(p)
    for i in range(p):
        q[i] = np.mean(X_full[:, i])  # mean of the i-th column
    
    mq = 1 - q
    
    # Calculate c
    c = np.dot(X.T, X) - np.outer(q, q) * n
    
    # Calculate d
    d = np.sqrt(n * np.outer(q, q) * np.outer(mq, mq))


    # Calculate the coexpression matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        coexp_matrix = c/d
        coexp_matrix[~np.isfinite(coexp_matrix)] = 0  # This will set inf and nan values to 0

    
    so.set_coexp(coexp_matrix)    
    return so

def cal_coexp_df(so, X, X_full):
    p = X.shape[1]  # number of columns (genes)
    n = X.shape[0]  # number of rows (cells)
    
    # Initialize q as a vector of length p
    q = np.zeros(p)
    for i in range(p):
        q[i] = np.mean(X_full.iloc[:, i])  # mean of the i-th column
    
    mq = 1 - q
    
    # Calculate c
    c = np.dot(X.T, X) - np.outer(q, q) * n
    
    # Calculate d
    d = np.sqrt(n * np.outer(q, q) * np.outer(mq, mq))
    
    # Calculate the coexpression matrix
    coexp_matrix = c / d
    
    so.set_coexp(coexp_matrix)    
    return so
    
def cal_coexp_sp(so, X, X_full):
    #X = pd.DataFrame.transpose(X)
    #X_full = pd.DataFrame.transpose(X_full)
    p = X.shape[1]
    n = X.shape[0]
    q = np.array(X_full.mean(axis=0)).flatten()  # Mean of each column
    mq = 1 - q
    c = X.T @ X - (q[:, None] * q[None, :]) * n  # Adjusted outer product and scaling by n
    d = np.sqrt(n * (q[:, None] * q[None, :]) * (mq[:, None] * mq[None, :]))
    
    # To prevent division by zero issues, ensure no zero elements in d
    d[d == 0] = np.finfo(float).eps  # Replace zeros with a very small number
    
    coexp_matrix = c / d  # Element-wise division
    return coexp_matrix

def create_network(so, alpha=0.05, manual=False, least_edge_prop=0.01):
    if so.coexp is None:
        raise ValueError("Coexpression matrix is not set in the object.")

    # Flatten the upper triangle of the coexpression matrix excluding the diagonal
    coex_vec = so.coexp.values[np.triu_indices_from(so.coexp, k=1)]
    
    # Estimate the null distribution mean and standard deviation
    est_ms = EstNull(coex_vec)
    
    # Update so object with estimated mean and std
    so.est_ms = est_ms
    
    # Determine the threshold for network edges based on FDR control
    thres = norm_FDR_SQAUC(coex_vec, est_ms['mean'], est_ms['std'], alpha, so.coexp.shape[0], so.coexp.shape[1])
    
    # Allow manual adjustment of the threshold if required
    if manual:
        adjusted_threshold = np.quantile(np.abs(coex_vec - est_ms['mean']), 1 - least_edge_prop)
        thres = min(thres, adjusted_threshold)
    
    # Update so object with the new threshold
    so.thres = thres
    
    return so

def gene_coexpression_network(counts, counts_subcohort=None, gene_name=None, meta_data=None, data_name=None, sparse=False, rowfeature=True):
    if counts_subcohort is None:
        counts_subcohort = counts

    # Create SiFiNet object
    so = create_sifinet_object(counts, gene_name, meta_data, data_name, sparse, rowfeature)    
    counts = pd.DataFrame.transpose(counts)
    counts_subcohort = pd.DataFrame.transpose(counts_subcohort)
    n, p = counts.shape  # dimensions from full matrix
    Z = np.mean(counts, axis=1)  # Use rowMeans from the full matrix

    dt = np.zeros((n, p))  # dense matrix initialization for the full data

    if not so.sparse:
        dt = np.zeros((n, p))
        for j in range(p):
            temp = counts.iloc[:, j].values
            #print("temp:", temp)
            #print("Z:", Z.flatten())
            v5 = np.quantile(temp, 0.5)
            #print("v5:", v5)
            if (v5 == 0) or (v5 >= np.quantile(temp, 0.99)):
                dt[:, j] = (temp > 0).astype(int)
            
            else:
                quant = np.sum(temp <= v5) / n
                model = QuantReg(temp, Z)
                fit = model.fit(q=quant)
                Q = fit.predict(Z.values.reshape(-1, 1))  # Ensure Z is 2D for statsmodels
                dt[:, j] = (temp > Q).astype(int)


    # Filter dt to only include rows that are common with M_subcohort
    common_indices = counts.index.isin(counts_subcohort.index)
    filtered_dt = dt[common_indices, :]
    
    so = cal_coexp_df(so, X = pd.DataFrame(filtered_dt), X_full = counts)
    
    so = create_network(so, alpha=0.05, manual=False, least_edge_prop=0.01)
    
    so = filter_lowexp(so, t1=10, t2=0.9, t3=0.9)

    return so

def normalize(column):
    mean = column.mean()
    std = column.std()
    if std == 0:
        std = 1
    return (column - mean) / std

def get_entrez_mapping(genes, scopes):
    mg = mygene.MyGeneInfo()
    result = mg.querymany(genes, scopes=scopes, fields='entrezgene', species='human')
    # create a mapping from gene id to entrez id, keeping the original gene id if no mapping is found
    mapping = {item['query']: str(item.get('entrezgene', item['query'])) for item in result}
    return [mapping[gene] for gene in genes]

def convert_gene_ids(adj_matrix, gene_type):
    """
    Convert gene IDs to Entrez IDs in the given adjacency matrix.

    Parameters:
    adj_matrix (pd.DataFrame): DataFrame with gene IDs as both index and columns.
    gene_type (str): Type of gene IDs used in the adjacency matrix ('ensembl' or 'symbol').

    Returns:
    pd.DataFrame: Modified DataFrame with Entrez IDs as both index and columns.
    
    Raises:
    ValueError: If gene_type is not 'ensembl' or 'symbol'.
    """
    if gene_type not in ['ensembl', 'symbol']:
        raise ValueError("gene_type must be either 'ensembl' or 'symbol'")

    # Convert row names
    row_genes = adj_matrix.index.tolist()
    row_scopes = 'ensembl.gene' if gene_type == 'ensembl' else 'symbol'
    row_converted = get_entrez_mapping(row_genes, row_scopes)
    adj_matrix.index = row_converted

    # Convert column names
    col_genes = adj_matrix.columns.tolist()
    col_scopes = 'ensembl.gene' if gene_type == 'ensembl' else 'symbol'
    col_converted = get_entrez_mapping(col_genes, col_scopes)
    adj_matrix.columns = col_converted

    # Remove duplicate rows and columns
    adj_matrix = adj_matrix.loc[~adj_matrix.index.duplicated(keep='first')]
    adj_matrix = adj_matrix.loc[:, ~adj_matrix.columns.duplicated(keep='first')]

    # Add "hsa:" to each row and column name
    adj_matrix.index = ["hsa:" + gene for gene in adj_matrix.index]
    adj_matrix.columns = ["hsa:" + gene for gene in adj_matrix.columns]

    return adj_matrix

def convert_gene_ids_list(gene_list, gene_type):
    """
    Convert gene IDs to Entrez IDs in the given gene list.

    Parameters:
    gene_list (list): List of gene IDs.
    gene_type (str): Type of gene IDs used in the gene list ('ensembl' or 'symbol').

    Returns:
    list: Modified list with Entrez IDs, prefixed with "hsa:".
    
    Raises:
    ValueError: If gene_type is not 'ensembl' or 'symbol'.
    """
    if gene_type not in ['ensembl', 'symbol']:
        raise ValueError("gene_type must be either 'ensembl' or 'symbol'")
    
    # Convert gene names
    gene_scopes = 'ensembl.gene' if gene_type == 'ensembl' else 'symbol'
    converted_genes = get_entrez_mapping(gene_list, gene_scopes)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_genes = []
    for gene in converted_genes:
        if gene not in seen:
            seen.add(gene)
            unique_genes.append(gene)
    
    # Add "hsa:" prefix to each gene ID
    prefixed_genes = [f"hsa:{gene}" for gene in unique_genes if gene]  # Ensures gene is not None or empty
    
    return prefixed_genes


# Function to convert NetworkX graph to iGraph graph
def create_network_from_adj_matrix(adj_matrix):
    """
    Create a network from an adjacency matrix, convert it to an iGraph graph.

    Parameters:
    adj_matrix (pd.DataFrame): Adjacency matrix with gene IDs as both index and columns.

    Returns:
    ig.Graph: iGraph graph with the same structure as the NetworkX graph.
    """
    # Convert the adjacency matrix to a NetworkX graph
    G_nx = nx.from_numpy_array(adj_matrix.values, create_using=nx.Graph)

    # Update the node names using the row names of the adjacency matrix
    G_nx = nx.relabel_nodes(G_nx, {i: name for i, name in enumerate(adj_matrix.index)})

    # Create a mapping from node names to indices
    mapping = {node: index for index, node in enumerate(G_nx.nodes())}

    # Initialize an igraph Graph with the correct number of vertices
    G_ig = ig.Graph(directed=False)
    # Add nodes with their names as attributes
    for node in G_nx.nodes():
        G_ig.add_vertex(name=node)

    # Prepare a list of edges with the corresponding weights
    edges = [(mapping[edge[0]], mapping[edge[1]]) for edge in G_nx.edges()]
    weights = [G_nx[edge[0]][edge[1]].get('weight', 1) for edge in G_nx.edges()]  # Adjust if using multigraph

    # Add edges and weights to the igraph Graph
    G_ig.add_edges(edges)
    G_ig.es['weight'] = weights

    return G_ig

def get_categorized_pathways():
    pathways = REST.kegg_list("pathway", "hsa").read().splitlines()
    categorized_pathways = defaultdict(list)

    for line in pathways:
        entry, description = line.split("\t")
        pathway_id = entry
        pathway_name = description
        category = pathway_name.split(" - ")[0]
        categorized_pathways[category].append(pathway_id)

    return categorized_pathways

def gather_pathways_between(start_pathway, end_pathway, categorized_pathways):
    start_found = False
    selected_pathways = []

    for category in categorized_pathways:
        if category == start_pathway:
            start_found = True
        if start_found:
            selected_pathways.extend(categorized_pathways[category])
        if category == end_pathway:
            break

    return selected_pathways

def get_genes_in_pathway(pathway_id):
    pathway_file = REST.kegg_get(pathway_id, "kgml").read()
    pathway = KGML_parser.read(pathway_file)
    genes = [gene_id for element in pathway.genes for gene_id in element.name.split()]
    return genes

def analyze_pathways(G, test_pathways=None, categorized_pathways=None, pathway_genes=None, num_walks=200, max_walk_length=200, null_dist_size=200):
    results = []

    if test_pathways and categorized_pathways:
        for pathway in test_pathways:
            start_pathway = pathway
            end_pathway = pathway

            # Gather pathways and get genes
            pathway_list = gather_pathways_between(start_pathway, end_pathway, categorized_pathways)
            current_pathway_genes = get_genes_in_pathway(pathway_list)
            
            # Perform embedding reconstruction
            p_value, z_score = embedding_recon(G, categorized_pathways, current_pathway_genes, num_walks, max_walk_length, null_dist_size)
            
            # Print and store results
            print(f"Pathway: {start_pathway}")
            print(f"p-value: {p_value}")
            print(f"Z-Score: {z_score}")
            print()
            results.append({
                "pathway": start_pathway, 
                "p-value": p_value, 
                "z-score": z_score
            })

    # Handle the case where pathway_genes is directly provided
    if pathway_genes and not test_pathways:
        p_value, z_score = embedding_recon(G, categorized_pathways, pathway_genes, num_walks, max_walk_length, null_dist_size)
        print("Custom Pathway Genes:", pathway_genes)
        print(f"p-value: {p_value}")
        print(f"Z-Score: {z_score}")
        print()
        results.append({
            "pathway": "Custom pathway", 
            "p-value": p_value, 
            "z-score": z_score
        })

    results_df = pd.DataFrame(results)
    return results_df
