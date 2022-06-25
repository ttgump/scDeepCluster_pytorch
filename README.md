# scDeepCluster_pytorch


scDeepCluster, a model-based deep embedding clustering for Single Cell RNA-seq data. 

![alt text](https://github.com/ttgump/scDeepCluster_pytorch/blob/main/network.png?raw=True)

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

**Parameters:**

--n_clusters: number of clusters, if setting as 0, it will be estimated by the Louvain alogrithm on the latent features.<br/>
--knn: number of nearest neighbors, which is used in the Louvain algorithm, default = 20.<br/>
--resolution: resolution in the Louvain algorith, default = 0.8. Larger value will result to more cluster numbers.<br/>
--select_genes: number of selected genes for the analysis, default = 0. Recommending to select top 2000 genes, but it depends on different datasets.<br/>
--batch_size: batch size, default = 256.<br/>
--data_file: file name of data.<br/>
--maxiter: max number of iterations in the clustering stage, default = 2000.<br/>
--pretrain_epochs: pretraining iterations, default = 300.<br/>
--gamma: coefficient of the clustering loss, default = 1.<br/>
--sigma: coefficient of the random Gaussian noise, default = 2.5.<br/>
--update_interval: number of iteration to update clustering targets, default = 1.<br/>
--tol: tolerance to terminate the clustering stage, which is the delta of predicted labels between two consecutive iterations, default = 0.001.<br/>
--final_latent_file: file name to output final latent representations of the autoencoder, default = final_latent_file.txt.<br/>
--predict_label_file: file name to output clustering labels, default = pred_labels.txt.<br/>

**Online app**

Online app website: https://app.superbio.ai
