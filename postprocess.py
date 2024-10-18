import numpy as np
import pandas as pd
import time, logging, argparse, sys, os
from tqdm import trange
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
        
parser = argparse.ArgumentParser(description='Postprocess the predicted copy number to segmentation file')
parser.add_argument('--data-path', type=str, help='The path to load the predicted data')
parser.add_argument('--segment-path', type=str, default='data/start_end_chr_in_segs.csv', help='The path to load the segment information')
parser.add_argument('--save-dir', type=str, default='data/predict',help='The directory to save the postprocessed result')
parser.add_argument('--save-name', type=str, help='The name of the postprocessed result without extension')
parser.add_argument('--log-dir', type=str, default='data/log', help='The directory to save the log')
parser.add_argument('--gene-names', type=str, default='data/gene_names_in_segments.csv', help='The path to gene names per segment')
parser.add_argument('--threshold', nargs=2, type=float, default=[-0.25, 0.2], help='The threshold for deletion and amplification')

if __name__ == "__main__":
    args = parser.parse_args()
    data_file = ".".join(args.data_path.split("/")[-1].split(".")[:-1])
    log_file = f'{args.log_dir}/postprocess_log_{data_file}_{ctime}.log'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file)
                        ])
    original_stderr = sys.stderr
    sys.stderr = TeeStderr(original_stderr, log_file)
    logging.info(args)
    logging.info(f'Loading prediction data from {args.data_path}...')
    data_array = np.load(args.data_path, allow_pickle=True)
    gene_names = np.loadtxt(args.gene_names, delimiter=',', dtype=str)
    # transform gene_names to a list of list of strings
    gene_names = [','.join(gene_names[i][gene_names[i] != "None"]) for i in range(gene_names.shape[0])]
    
    data_cna = data_array['cna']
    data_profile = data_array['profile']
    data_segment = pd.read_csv(args.segment_path)
    data_chr = list(data_segment['chr'])
    data_start = list(data_segment['start'])
    data_end = list(data_segment['end'])
    data_cna_status = np.empty(data_cna.shape, dtype=str)
    data_cna_status[np.isnan(data_cna)] = 'NA'
    data_cna_status[data_cna<=args.threshold[0]] = '-'
    data_cna_status[data_cna>=args.threshold[1]] = '+'
    data_cna_status[(data_cna>args.threshold[0]) & (data_cna<args.threshold[1])] = '0'
    logging.info(f'Loaded the predicted data with {data_profile.shape[0]} samples')
    logging.info('Postprocessing the predicted data...')
    
    
    pbar = trange(data_cna.shape[0], desc='Postprocessing', file=sys.stdout)
    profile_list, cna_list, cna_status_list, gene_list = [], [], [], []
    for i in pbar:
        profile_list += [data_profile[i]] * data_cna.shape[1]
        cna_list += list(data_cna[i])
        cna_status_list += list(data_cna_status[i])
        gene_list += gene_names
        
        
    data_frame = pd.DataFrame({'Sample': profile_list, "Chromosome": data_chr * data_cna.shape[0], "Start": data_start * data_cna.shape[0], "End": data_end * data_cna.shape[0], 'SegMean': cna_list, 'Status': cna_status_list, 'Gene': gene_list})
    # save the data frame to a tsv file
    if args.save_name is not None:
        save_name = f'{args.save_dir}/{args.save_name}.tsv'
    else:
        save_name = f'{args.save_dir}/{data_file}_postprocessed.tsv'
    data_frame.to_csv(save_name, sep='\t', index=False)
    logging.info(f'Saved the postprocessed data to {save_name}')
    print(f'Saved the postprocessed data to {save_name}')
    sys.stderr = original_stderr