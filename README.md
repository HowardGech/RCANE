# RCANE

RCANE is a deep learning algorithm designed for predicting somatic copy number aberrations (SCNAs) across multiple cancer types using RNA-seq data. The dataset used for training and evaluating the method, along with a pre-trained model based on TCGA data and a fine-tuned model using DepMap cell line data, can be found [here](https://doi.org/10.5281/zenodo.13953644).

## Installation
To install RCANE, simply clone this repository and navigate to its root directory. Alternatively, you can download the zip file and unzip it. To reproduce the following results, please also download the zip file from [here](https://doi.org/10.5281/zenodo.13953644), unzip it, and merge the two root folders.


## Prediction

To predict SCNAs using RNA-seq data, please follow the steps for [preprocessing](#preprocessing), [model running](#model-running), and [postprocessing](#postprocessing).

### Preprocessing

Your RNA expression file should be a `.csv` file where rows represent samples and columns represent genes. The file should begin with one column labeled `SampleID`, containing the sample IDs, and one column labeled `CancerType`, containing the cancer types using [TCGA abbreviations](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations). The following columns should contain gene names, such as *EGFR* and *AC096540.1*. To ensure accurate and robust results, please make sure your expression values are normalized by $\log_2(1+\mathrm{TPM})$ and that batch effects have been corrected. See `data/predict/TCGA_test_RNA.csv` for reference.

Open a terminal in the root directory and run the following command:

```{r, engine = 'bash', eval = FALSE}
for foo in (ls bar)
do
  echo $foo
done
```

Replace `path/to/RNAseq.csv` with the actual path to your RNA-seq file, and `saved_file` with the name you want for the preprocessed output (without the file extension). Other optional arguments include:

- `save-dir`: the directory where the preprocessed files will be saved
- `log-dir`: the directory for the logging file
- `gene-names`: the path to the gene names file used to match the training data

For default values and other information of these parameters, refer to `preprocess.py`.

The preprocessed file should be a NumPy zip file `saved_file.npz` containing three components: `rna`, `cohort` and `ID`.