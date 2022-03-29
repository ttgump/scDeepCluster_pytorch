# scDeepCluster_pytorch


scDeepCluster, a model-based deep embedded clustering for Single Cell RNA-seq data. 

**Reference:**

Tian, T., Wan, J., Song, Q., & Wei, Z. (2019). Clustering single-cell RNA-seq data with a model-based deep learning approach. Nature Machine Intelligence, 1(4), 191-198. https://www.nature.com/articles/s42256-019-0037-0.

**Requirements:**

Scanpy -- 1.7

Pytorch -- 1.8

**Usage:**

python run_scDeepCluster.py --data_file data.h5 --n_clusters 0

Set data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix, and Y is the true labels. Y is optional), n_clusters to the number of clusters (0 for automatically estimating by the Louvain algorithm on the pretrained latent features).

python run_scDeepClusterBatch.py --data_file data.h5 --n_clusters 0

This is the script for clustering analysis of datasets with batches (stored in h5 format, with three components X, B and Y, where X is the cell by gene count matrix, B is the one-hot encoded batch IDs, and Y is the true labels. Y is optional). n_clusters to the number of clusters (0 for automatically estimating by the Louvain algorithm on the pretrained latent features).
