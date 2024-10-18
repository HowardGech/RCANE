from model.model_architecture import RCANE
from model.utils_model import *
import yaml, argparse, sys, os, logging, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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

parser = argparse.ArgumentParser(description='Train the RCANE model')
parser.add_argument('--new', action='store_true', help='Training a new RCANE model from scratch')
parser.add_argument('--fine-tune', action='store_true', help='Fine-tuning the RCANE model')
parser.add_argument('--data-path', type=str, help='The path to load the data')
parser.add_argument('--model-path', type=str, help='The path to load the model')
parser.add_argument('--save-dir', type=str, default='model/trained_model',help='The directory to save the model')
parser.add_argument('--log-dir', type=str, default='data/log', help='The directory to save the log')
parser.add_argument('--train-config', type=str, default='train_config.yaml', help='The path to training configuration')
parser.add_argument('--model-param', type=str, help='The path to model parameters')
parser.add_argument('--gene-names', type=str, default='data/gene_names_in_segments.csv', help='The path to gene names per segment')
parser.add_argument('--edge-path', type=str, default='data/edge_indices.pt', help='The path to segment edge indices')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.train_config, 'r') as f:
        train_config_dict = yaml.safe_load(f)
    device = train_config_dict['device']
    if args.fine_tune:
        log_file = f'{args.log_dir}/train_log_mp{train_config_dict["mask_prob"]}_{ctime}_finetune_{device}.log'
        trajectory_file = f'{args.log_dir}/train_trajectory_mp{train_config_dict["mask_prob"]}_{ctime}_finetune_{device}.pdf'
        loss_file = f'{args.log_dir}/train_loss_mp{train_config_dict["mask_prob"]}_{ctime}_finetune_{device}.txt'
        model_file = f'{args.save_dir}/RCANE_mp{train_config_dict["mask_prob"]}_{ctime}_finetune_{device}.pth'
        temp_model_file = f'{args.save_dir}/RCANE_mp{train_config_dict["mask_prob"]}_{ctime}_finetune_{device}_temp.pth'
    else:
        log_file = f'{args.log_dir}/train_log_mp{train_config_dict["mask_prob"]}_{ctime}_{device}.log'
        trajectory_file = f'{args.log_dir}/train_trajectory_mp{train_config_dict["mask_prob"]}_{ctime}_{device}.pdf'
        loss_file = f'{args.log_dir}/train_loss_mp{train_config_dict["mask_prob"]}_{ctime}_{device}.txt'
        model_file = f'{args.save_dir}/RCANE_mp{train_config_dict["mask_prob"]}_{ctime}_{device}.pth'
        temp_model_file = f'{args.save_dir}/RCANE_mp{train_config_dict["mask_prob"]}_{ctime}_{device}_temp.pth'
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file)
                        ])
    original_stderr = sys.stderr
    sys.stderr = TeeStderr(original_stderr, log_file)
    logging.info(args)
    if args.new:
        if args.fine_tune:
            sys.exit('Please specify the model to fine-tune')
        if args.model_param is None:
            sys.exit('Please specify the model parameters')
        with open(args.model_param, 'r') as f:
            model_param_dict = yaml.safe_load(f)
        model = RCANE(model_param_dict).to(device)
    else:
        model = torch.load(args.model_path, map_location=device)
    
    # Load the data
    COHORTS = model.cancer_types
    logging.info(f'Loading edge indices from {args.edge_path}')
    all_edges = torch.load(args.edge_path)
    edge_pos_list = [all_edges['positive'][cancer].to(device) for cancer in model.cancer_types]
    edge_neg_list = [all_edges['negative'][cancer].to(device) for cancer in model.cancer_types]
    logging.info(f'Loaded edge_pos with length {len(edge_pos_list)}, edge_neg with length {len(edge_neg_list)}')
    logging.info(f'Loading data from {args.data_path}')
    bs = train_config_dict['batch_size'] if 'batch_size' in train_config_dict else 1
    data_array = np.load(args.data_path, allow_pickle=True)
    data_rna = torch.tensor(data_array['rna'], dtype=torch.float).to(device)
    data_cna = torch.tensor(data_array['cna'], dtype=torch.float).to(device)
    cohort_str = data_array['cohort']
    data_cohort = torch.tensor([COHORTS.index(cohort_str[i]) for i in range(len(cohort_str))]).long().to(device)

    data_tensor = TensorDataset(data_rna, data_cna, data_cohort)
    data_loader = DataLoader(data_tensor, batch_size=bs, shuffle=True)
    gene_names = np.loadtxt(args.gene_names, delimiter=',', dtype=str)
    end_of_chr = torch.Tensor(gene_names == 'None').bool()
    logging.info(f'Loaded data with RNA shape {data_rna.shape}, CNA shape {data_cna.shape}, cohort shape {data_cohort.shape}')
    
    # delete the temporary data to save memory
    del data_array
    del data_rna
    del data_cna
    del data_cohort
    del cohort_str
    del all_edges

    
    # Train the model
    
    if args.fine_tune:
        param_to_update = list(model.mlp_out.parameters()) + list(model.debiaser.parameters())
        for param in model.parameters():
            param.requires_grad = False
        for param in param_to_update:
            param.requires_grad = True
    else:
        param_to_update = list(model.parameters())
        for param in param_to_update:
            param.requires_grad = True
    
    optimizer = getattr(optim, train_config_dict['optimizer'])(param_to_update, **train_config_dict['optimizer_args'])
    criterion = getattr(nn, train_config_dict['criterion'])(**train_config_dict['criterion_args'])
    epochs = train_config_dict['epochs'] if 'epochs' in train_config_dict else 100
    accumulation_steps = train_config_dict['accumulation_steps'] if 'accumulation_steps' in train_config_dict else 1
    log_steps = train_config_dict['log_steps'] if 'log_steps' in train_config_dict else 1
    save_steps = train_config_dict['save_steps'] if 'save_steps' in train_config_dict else 10
    
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)
    model.train()
    model.zero_grad()
    logging.info(f'Start training with optimizer {train_config_dict["optimizer"]} and criterion {train_config_dict["criterion"]} on device {device}')
    logging.info(f'Optimizer args: {train_config_dict["optimizer_args"]}, Criterion args: {train_config_dict["criterion_args"]}')
    logging.info(f'Batch size: {bs}, Epochs: {epochs}, Accumulation steps: {accumulation_steps}, Log steps: {log_steps}, Save steps: {save_steps}')
    logging.info(f'The temporary model will be saved to {temp_model_file}')
    train_loss = []
    pbar = trange(epochs, file=sys.stdout)
    for epoch in pbar:
        loss_each_epoch = 0
        for i, (rna, cna, cohort) in enumerate(data_loader):
            optimizer.zero_grad()
            edge_pos_batch = prepare_graphs_by_cohorts(cohort, edge_pos_list, device)
            edge_neg_batch = prepare_graphs_by_cohorts(cohort, edge_neg_list, device)
            mask = prepare_masking(rna, end_of_chr, train_config_dict['mask_prob'], device)
            output = model(rna, cohort, edge_pos_batch, edge_neg_batch, mask)
            cna_nan = torch.isnan(cna)
            loss = criterion(output[~cna_nan], cna[~cna_nan])
            loss = torch.mean(loss)
            loss.backward()
            loss_each_epoch += loss * rna.size(0)
            if (i+1) % accumulation_steps == 0 or i == len(data_loader)-1:
                optimizer.step()
                model.zero_grad()
            pbar.set_description(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(data_loader)}], Loss [{loss.item():.4f}]")
        loss_each_epoch /= len(data_loader.dataset)
        train_loss.append(loss_each_epoch.detach().cpu().numpy())
        np.savetxt(loss_file, np.array(train_loss))
        if (epoch+1) % log_steps == 0 or epoch == epochs-1:
            logging.info(f'Epoch {epoch+1}/{epochs}, loss: {loss_each_epoch}')
            plt.plot(train_loss)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(trajectory_file)
            plt.close()
        if (epoch+1) % save_steps == 0 or epoch == epochs-1:
            model.eval()
            torch.save(model, temp_model_file)
            model.train()
            
    logging.info(f'Finished training! Saving the model to {model_file}')
    model.eval()
    torch.save(model, model_file)
    os.remove(temp_model_file)
    logging.info('Model saved!')
    sys.stderr = original_stderr