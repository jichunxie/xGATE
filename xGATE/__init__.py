"""xGATE — pathway analysis for single-cell RNA-seq.

The method implementation lives in the :mod:`xGATE.utilities` subpackage; import
what you need from there, e.g.::

    from xGATE.utilities import create_network_from_adj_matrix, embedding_recon

Kept import-light on purpose: importing :mod:`xGATE` does not eagerly pull in the
heavy dependencies (torch/scanpy/biopython) — those load only when
:mod:`xGATE.utilities` is imported.
"""

__version__ = "1.0.0"
