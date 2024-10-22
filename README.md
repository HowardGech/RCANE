# RCANE

RCANE is a deep learning algorithm designed for predicting somatic copy number aberrations (SCNAs) across multiple cancer types using RNA-seq data. The dataset used for training and evaluating the method, along with a pre-trained model based on TCGA data and a fine-tuned model using DepMap cell line data, can be found [here](https://doi.org/10.5281/zenodo.13953644).

## Installation
To install RCANE, simply clone this repository and navigate to its root directory. Alternatively, you can download the zip file and unzip it. To reproduce the following results, please also download the zip file from [here](https://doi.org/10.5281/zenodo.13953644), unzip it, and merge the two root folders.


## Prediction

To predict SCNAs using RNA-seq data, please follow the steps for [preprocessing](#step-1-preprocessing), [model running](#step-2-model-running), and [postprocessing](#step-3-postprocessing).

### Step 1. Preprocessing

Your RNA expression file should be a `.csv` file where rows represent samples and columns represent genes. The file should begin with one column labeled `SampleID`, containing the sample IDs, and one column labeled `CancerType`, containing the cancer types using [TCGA abbreviations](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations). The following columns should contain gene names, such as *EGFR* and *AC096540.1*. To ensure accurate and robust results, please make sure your expression values are normalized by $\log_2(1+\mathrm{TPM})$ and that batch effects have been corrected. See `data/predict/TCGA_test_RNA.csv` as an example.

Open a terminal in the root directory and run the following command:

```bash
python preprocess.py --data-path path/to/RNAseq.csv --save-name saved_file
```

Replace `path/to/RNAseq.csv` with the actual path to your RNA-seq file, and `saved_file` with the name you want for the preprocessed output (without file extension). Other optional arguments include:

- `save-dir`: the directory where the preprocessed files will be saved
- `gene-names`: the path to the gene names file used to match the training data

For default values and other information of these parameters, refer to `preprocess.py`.

The preprocessed file should be a NumPy zip file `saved_file.npz` containing three components: `rna`, `cohort`, and `ID`, along with a gene names file `gene_names_saved_file.csv` denoting the matched genes of the input RNA file.

### Step 2. Model Running

To run the model, execute the following command in the terminal:

```bash
python predict.py --model-path path/to/model.pth --data-path path/to/data.npz --save-name pred_file
```

Replace `path/to/model.pth` with the path to the pretrained model, `path/to/data.npz` with the preprocessed RNA file, and `pred_file` with your name of the prediction file (without file extension). Other optional arguments include:

- `save-dir`: the directory where the predicted file will be saved
- `predict-config`: the YAML configuration file for prediction
- `gene-names`: the path to the gene names file matched to the training data
- `edge-path`: the path to graph edge indices used in the Graph Attention layer

For default values and further details on these parameters, refer to `predict.py`.

The SCNA prediction file is a NumPy zip file `pred_file.npz` containing `rna`, `cna`, `cohort`, and `ID`.

### Step 3. Postprocessing

After running the RCANE model, you can reformat the prediction file to a human-readable format. Run the following in the terminal:

```bash
python postprocess.py --data-path path/to/pred_file.npz --save-name post_file
```

Replace `path/to/pred_file.npz` with the path to the prediction file, and `post_file` with the name of your post-processed file (without file extension). Other optional arguments include:

- `segment-path`: the path to the segment information containing three columns
    - `chrom`: the chromosome each segment belongs to (e.g., 1, 2, 3, ..., X)
    - `loc.start`: the start location of each segment on the chromosome
    - `loc.end`: the end location of each segment on the chromosome
- `save-dir`: the directory where the post-processed file will be saved
- `gene-names`: the path to the gene names file (should be the same as used in [Model Running](#step-2-model-running))
- `threshold`: two values between which are considered SCNA neutral

For default values and further details on these arguments, refer to `postprocess.py`.

The post-processed file is a tab-separated `post_file.tsv` containing 6 columns: `ID`, `chrom`, `loc.start`, `loc.end`, `seg.mean`, `status`, and `gene`.

## Measurement

The `measurement` folder contains three files for data analysis and visualization:

- `performance_evaluation.ipynb`: evaluates the prediction performance by comparing real and predicted SCNA data.
- `plot_weights.ipynb`: plots Pearson's correlation coefficients between RNA and SCNA data versus the weights assigned to each gene in the trained RCANE model, highlighting the selected genes.
- `visualization.ipynb`: visualizes the predicted whole-genome SCNAs.

To use these notebooks, replace the file paths to the `RCANE` project and the appropriate file names, then click `Run All` in Jupyter Notebook to generate the results.


## Training and Fine-Tuning

For model training and fine-tuning, a `.npz` file containing RNA-seq data, copy number intensity data, and cancer types is required. To train a model from scratch, use the following command:

```bash
python train.py --new --data-path path/to/data.npz --save-dir folder/to/save/model
```

To fine-tune a model using a pretrained model, use the command:

```bash
python train.py --fine-tune --data-path path/to/data.npz --model-path path/to/pretrained/model.pth --save-dir folder/to/save/model 
```

Replace `path/to/data.npz` with the path to your data, `folder/to/save/model` with the directory to save the trained or fine-tuned model, and `path/to/pretrained/model.pth` with the path to the pretrained model for fine-tuning. Other important optional arguments include:

- `train-config`: YAML configuration file for model training, including training device, optimizer, epochs, and more (see `train_config.yaml` for details)
- `model-param`: YAML configuration file for building a new model, including segment lengths, neural network layers, activation functions, and more (see `model_param.yaml` for details)
- `gene-names`: the path to the gene names file
- `edge-path`: the path to graph edge indices used in the Graph Attention layer

The trained model will be saved in `folder/to/save/model`, with filenames starting with `RCANE`, followed by information such as training time, and ending with the `.pth` file extension.
