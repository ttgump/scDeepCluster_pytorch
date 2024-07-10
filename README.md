# scDeepCluster_pytorch


The pytorch version of scDeepCluster, a model-based deep embedding clustering for Single Cell RNA-seq data. <br/>

Comparing to the original Keras version, I introduced two new features:<br/>
1. The Louvain clustering is implemented after pretraining to allow estimating number of clusters.<br/>
2. A new script "scDeepClusterBatch" uses conditional autoencoder technic to integrate single-cell data from different batches.<br/>

**updates:

11/27/2023: I updated the model to use float64 precision.

## Table of contents
- [Network diagram](#diagram)
- [Requirements](#requirements)
- [Usage](#usage)
- [Parameters](#parameters)
- [Outputs](#outputs)
- [Reference](#reference)
- [Online app](#app)
- [Contact](#contact)

## <a name="diagram"></a>Network diagram
![alt text](https://github.com/ttgump/scDeepCluster_pytorch/blob/main/network.png?raw=True)

## <a name="requirements"></a>Requirements

Scanpy -- 1.7 (https://scanpy.readthedocs.io/en/stable/)

Pytorch -- 1.8 (https://pytorch.org)

## <a name="usage"></a>Usage

For single-cell count data:

```sh
python run_scDeepCluster.py --data_file data.h5 --n_clusters 0
```

Set data_file to the destination to the data (**stored in h5 format, with two components X and Y, where X is the cell by gene count matrix, and Y is the true labels. Y is optional**), n_clusters to the number of clusters (0 for automatically estimating by the Louvain algorithm on the pretrained latent features).

For single-cell count data from multiple batches:

```sh
python run_scDeepClusterBatch.py --data_file data.h5 --n_clusters 0
```

This is the script for clustering analysis of datasets from different batches (**stored in h5 format, with three components X, B and Y, where X is the cell by gene count matrix, B is the one-hot encoded batch IDs, and Y is the true labels. Y is optional**). Following the idea from scVI paper (https://doi.org/10.1038/s41592-018-0229-2), we use the conditional autoencoder (https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) technic to integrate different batches. n_clusters to the number of clusters (0 for automatically estimating by the Louvain algorithm on the pretrained latent features).

## <a name="parameters"></a>Parameters

**--n_clusters:** number of clusters, if setting as 0, it will be estimated by the Louvain alogrithm on the latent features after pretraining. If setting as an integer > 0, then the model will use the user defined value as number of clusters.<br/>
**--knn:** number of nearest neighbors, which is used in the Louvain algorithm, default = 20. Not used when setting n_clusters > 0<br/>
**--resolution:** resolution in the Louvain algorith, default = 0.8. Larger value will result to more cluster numbers. Not used when setting n_clusters > 0.<br/>
**--select_genes:** number of selected genes for the analysis, default = 0. It will use the mean-variance relationship to select informative genes. Recommending to select top 2000 genes, but it depends on different datasets.<br/>
**--batch_size:** batch size, default = 256.<br/>
**--data_file:** file name of data.<br/>
**--maxiter:** max number of iterations in the clustering stage, default = 2000.<br/>
**--pretrain_epochs:** pretraining iterations, default = 300.<br/>
**--gamma:** coefficient of the clustering loss, default = 1.<br/>
**--sigma:** coefficient of the random Gaussian noise, default = 2.5.<br/>
**--update_interval:** number of iteration to update clustering targets, default = 1.<br/>
**--tol:** tolerance to terminate the clustering stage, which is the delta of predicted labels between two consecutive iterations, default = 0.001.<br/>
**--final_latent_file:** file name to output final latent representations of the autoencoder, default = final_latent_file.txt.<br/>
**--predict_label_file:** file name to output clustering labels, default = pred_labels.txt.<br/>

## <a name="outputs"></a>Outputs

- **final_latent:** low dimensional representations of scRNA-seq data, default shape (n_cells, 32), which can be visualized by t-SNE or UMAP.<br/>
- **predict_label:** predicted clustering labels, shape (n_cells).<br/>

## <a name="reference"></a>Reference

Tian, T., Wan, J., Song, Q., & Wei, Z. (2019). Clustering single-cell RNA-seq data with a model-based deep learning approach. *Nature Machine Intelligence*, 1(4), 191-198. https://www.nature.com/articles/s42256-019-0037-0.

## <a name="app"></a>Online app

Online app website: https://app.superbio.ai/apps/107

## <a name="contact"></a>Contact

Tian Tian tiantianwhu@163.com
