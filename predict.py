from model.model_architecture import RCANE
from model.utils_model import *
import torch
import yaml, argparse, sys, os, logging, time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
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
        
parser = argparse.ArgumentParser(description='Predict the copy number using the trained RCANE model')
parser.add_argument('--data-path', type=str, help='The path to load the data')
parser.add_argument('--model-path', type=str, help='The path to load the model')
parser.add_argument('--save-dir', type=str, default='data/predict',help='The directory to save the predicted result without extension')
parser.add_argument('--save-name', type=str, help='The name of the predicted result file')
parser.add_argument('--log-dir', type=str, default='data/log', help='The directory to save the log')
parser.add_argument('--predict-config', type=str, default='predict_config.yaml', help='The path to prediction configuration')
parser.add_argument('--gene-names', type=str, default='data/gene_names_in_segments.csv', help='The path to gene names per segment')
parser.add_argument('--edge-path', type=str, default='data/edge_indices.pt', help='The path to segment edge indices')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.predict_config, 'r') as f:
        predict_config_dict = yaml.safe_load(f)
    device = predict_config_dict['device']
    log_file = f'{args.log_dir}/predict_log_{ctime}_{device}.log'
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file)
                        ])
    original_stderr = sys.stderr
    sys.stderr = TeeStderr(original_stderr, log_file)
    logging.info(args)
    bs = predict_config_dict['batch_size'] if 'batch_size' in predict_config_dict else 1
    logging.info(f'Loading model from {args.model_path}...')
    model = torch.load(args.model_path, map_location=device)
    logging.info(f'Loading data from {args.data_path}...')
    data_array = np.load(args.data_path, allow_pickle=True)
    data_rna = torch.tensor(data_array['rna'], dtype=torch.float).to(device)
    cohort_str = data_array['cohort']
    COHORTS = model.cancer_types
    all_edges = torch.load(args.edge_path)
    edge_pos_list = [all_edges['positive'][cancer].to(device) for cancer in model.cancer_types]
    edge_neg_list = [all_edges['negative'][cancer].to(device) for cancer in model.cancer_types]
    data_cohort = torch.tensor([COHORTS.index(cohort_str[i]) for i in range(len(cohort_str))]).long().to(device)
    if 'ID' in data_array.files:
        data_profile = data_array['ID']
    data_tensor = TensorDataset(data_rna, data_cohort)
    data_loader = DataLoader(data_tensor, batch_size=bs, shuffle=False)
    gene_names = np.loadtxt(args.gene_names, delimiter=',', dtype=str)
    end_of_chr = torch.Tensor(gene_names == 'None').bool()
    logging.info(f'Loaded RNA expression with shape {data_rna.shape} and cohorts with shape {data_cohort.shape}')
    
    
    logging.info(f'Predicting copy numbers...')
    model.eval()
    pbar = trange(len(data_loader), desc='Predicting batch', file=sys.stdout)
    with torch.no_grad():
        for i, (rna, cohort) in enumerate(data_loader):
            rna = rna.to(device)
            cohort = cohort.to(device)
            edge_pos_batch = prepare_graphs_by_cohorts(cohort, edge_pos_list, device)
            edge_neg_batch = prepare_graphs_by_cohorts(cohort, edge_neg_list, device)
            mask = prepare_masking(rna, end_of_chr, 0., device)
            pred_cna = model(rna, cohort, edge_pos_batch, edge_neg_batch, mask)
            if i == 0:
                pred_cna_all = pred_cna
            else:
                pred_cna_all = torch.cat((pred_cna_all, pred_cna), dim=0)
            pbar.update(1)
    pbar.close()
    pred_cna_all = pred_cna_all.detach().cpu().numpy()
    logging.info(f'Saving predicted copy numbers to {args.save_dir}...')
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)
    if args.save_name is not None:
        pred_file = f'{args.save_dir}/{args.save_name}.npz'
    else:
        pred_file = f'{args.save_dir}/predicted_{".".join(args.data_path.split("/")[-1].split(".")[:-1])}.npz'
    if 'profile' in data_array.files:
        np.savez(pred_file, rna = data_array['rna'], cna = pred_cna_all, cohort = cohort_str, ID = data_profile)
    else:
        np.savez(pred_file, rna = data_array['rna'], cna = pred_cna_all, cohort = cohort_str)
    logging.info(f'Predicted data saved!')
    print(f'Predicted data saved to {pred_file}')
    sys.stderr = original_stderr