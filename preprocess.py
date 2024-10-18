import numpy as np
import pandas as pd
import time, logging, argparse, sys, os, csv
from tqdm import tqdm
ctime = time.strftime("%Y%m%d_%H%M%S")

class TeeStderr:
    """
    Custom class that writes to both the original stderr and a log file.
    """
    def __init__(self, original_stderr, log_file_path):
        self.original_stderr = original_stderr  
        self.log_file = open(log_file_path, 'a') 

    def write(self, message):
        self.original_stderr.write(message)
        self.log_file.write(message)

    def flush(self):
        self.original_stderr.flush()
        self.log_file.flush()
        
parser = argparse.ArgumentParser(description='Preprocess the RNA-seq data for training and prediction')
parser.add_argument('--data-path', type=str, help='The path to load the RNA-seq data for preprocessing')
parser.add_argument('--save-dir', type=str, default='data/predict',help='The directory to save the preprocessed result')
parser.add_argument('--save-name', type=str, help='The name of the preprocessed result ending with .npz')
parser.add_argument('--log-dir', type=str, default='data/log', help='The directory to save the log')
parser.add_argument('--gene-names', type=str, default='data/gene_names_in_segments.csv', help='The path to gene names per segment')

if __name__ == "__main__":
    args = parser.parse_args()
    data_file = ".".join(args.data_path.split("/")[-1].split(".")[:-1])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.save_name is not None:
        save_name = args.save_name
    else:
        save_name = f"{data_file}.npz"
    log_file = f'{args.log_dir}/preprocess_log_{data_file}_{ctime}.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file)
                        ])
    original_stderr = sys.stderr
    sys.stderr = TeeStderr(original_stderr, log_file)
    logging.info(args)
    logging.info(f'Loading RNA-seq data from {args.data_path}...')
    data = pd.read_csv(args.data_path)
    gene_names = np.loadtxt(args.gene_names, delimiter=',', dtype=str)
    
    logging.info(f'Loaded {data.shape[0]} samples and {data.shape[1]-2} genes')
    data_profile = data['Profile']
    data_cohort = data['Cohort']
    data_rna = np.empty((data.shape[0], gene_names.shape[0], gene_names.shape[1]), dtype=float)
    pbar = tqdm([(i, j) for i in range(gene_names.shape[0]) for j in range(gene_names.shape[1])], file=sys.stdout)
    existing_genes = np.empty(gene_names.shape, dtype='U25')
    
    for i, j in pbar:
        pbar.set_description(f'Processing gene {gene_names[i][j]}')
        if gene_names[i][j] == "None":
            data_rna[:, i, j] = np.nan
            existing_genes[i, j] = "None"
            continue
        gene_name = gene_names[i][j]
        if gene_name not in data.columns:
            data_rna[:, i, j] = np.nan
            existing_genes[i, j] = "None"
            continue
        existing_genes[i, j] = gene_name
        data_rna[:, i, j] = data[gene_name]
        
    logging.info(f'Preprocessed RNA-seq data with shape {data_rna.shape}')
    logging.info(f'Saving preprocessed data to {args.save_dir}/{save_name}...')
    np.savez(f'{args.save_dir}/{save_name}', rna=data_rna, profile=data_profile, cohort=data_cohort)
    with open(f'data/gene_names_{".".join(save_name.split(".")[:-1])}.csv', 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        writer.writerows(existing_genes)
    logging.info(f'Saved preprocessed data to {args.save_dir}/{save_name}. The gene names are saved to {args.save_dir}/gene_names_{".".join(save_name.split(".")[:-1])}.csv')
    print(f'Saved preprocessed data to {args.save_dir}/{save_name}. The gene names are saved to data/gene_names_{".".join(save_name.split(".")[:-1])}.csv')
    sys.stderr = original_stderr
    
        