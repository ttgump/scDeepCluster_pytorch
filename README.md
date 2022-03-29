# scDeepCluster_pytorch


scDeepCluster, a model-based deep embedded clustering for Single Cell RNA-seq data. See details in our paper: "Clustering single-cell RNA-seq data with a model-based deep learning approach" published in Nature Machine Intelligence https://www.nature.com/articles/s42256-019-0037-0.

Requirements:

Scanpy -- 1.7

Pytorch -- 1.8

Usage:

python scDeepCluster.py --data_file data.h5 --n_clusters 0

set data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix, and Y is the true labels. Y is optional), n_clusters to the number of clusters (0 for automatically estimating by the Louvain algorithm on the pretrained latent features).
