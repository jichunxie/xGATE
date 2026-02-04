# xGATE

**xGATE** is a novel knowledge-base-informed method designed to infer pathway activity by capturing chain-like topology structures in gene co-expression graphs. It integrates graph theory, deep learning, and hypothesis testing to provide a high-accuracy, computationally efficient solution for single-cell RNA-sequencing (scRNA-seq) and spatial transcriptomics pathway analysis.

## Key Features

- **Biologically-Informed and Flexible**: xGATE integrates existing pathway databases, such as KEGG, Reactome, and WikiPathways, to analyze and identify active pathways. However, xGATE also allows users to define and test any custom gene set where the genes are believed to work together to perform a functional role.
- **High Accuracy**: Demonstrated superior performance in identifying active pathways across multiple cell types.
- **Resilient to Batch Effects**: Mitigates batch effects by leveraging co-expression relationships rather than marginal expression levels.

## How xGATE Works

- **User input**: Gene set/pathway → **xGATE Output**: Gene set/pathway p-value and effect size.

1. Constructs a gene co-expression graph from scRNA-seq data.
2. Extracts pathway subgraphs and generates random subgraphs for null comparisons.
3. Derives graph embeddings to capture relevant topological features.
4. Uses a graph-based **variational autoencoder (VAE)** to establish a null distribution for pathway activeness.
5. Performs statistical hypothesis testing to determine pathway activity and computes effect sizes.


![xGATE Overview](images/xGATE_Flowchart.jpeg)

## Installation

To install xGATE, use the following steps:

```bash
# Clone the repository
git clone https://github.com/Cafecito95/xGATE.git

# Navigate to the directory
cd xGATE

# Install dependencies
pip install -r requirements.txt
