import torch
import operator

def prepare_graphs_by_cohorts(cohort, edge_index, device):
    """
    Prepare the graphs for each sample based on the cohort information
    
    Args:
    - cohort: a torch.long tensor of cohort information for each sample
    - edge_index: a list of torch.long tensors with shape (2, num_edges) for each cohort representing the edge index
    - device: a torch.device object
    
    Returns:
    - graphs: a list of torch.long tensors with shape (2, num_edges) for each sample representing the edge index

    """
    edge_index_batch = operator.itemgetter(*(cohort.tolist()))(edge_index)
    edge_index_batch_list = list(edge_index_batch) if isinstance(edge_index_batch, tuple) else [edge_index_batch]
    edge_index_batch_list = [edge_index.to(device) for edge_index in edge_index_batch_list]
    return edge_index_batch_list

def prepare_masking(data, end_of_chr, mask_prob, device):
    """
    Prepare the masking for the input data
    
    Args:
    - data: a torch.float tensor with shape (batch_size, num_seq, num_genes)
    - end_of_chr: a torch.bool tensor with shape (num_seq, num_genes) indicating the end of chromosome maskings
    - mask_prob: a float indicating the random masking probability
    - device: a torch.device object
    
    Returns:
    - mask: a torch.float tensor with shape (batch_size, num_seq, num_genes) representing the masking for the input data
    """
    mask_index = torch.rand(data.shape) < mask_prob
    mask_index[:, end_of_chr] = True
    mask_index = mask_index.bool()
    mask = torch.zeros(data.shape)
    mask[mask_index] = float('-inf')
    mask = mask.to(device)
    return mask
def combine_edge_indices(edge_indices, num_nodes_per_graph):
    """
    Combine edge indices for multiple graphs into a single tensor as input to the GATTN layer.
    
    Parameters:
    -----------
    edge_indices : list of torch.Tensor
        A list of edge indices for each graph.
    num_nodes_per_graph : int
        The number of nodes in each graph.
    
    Returns:
    --------
    torch.Tensor
        A single tensor containing the combined edge indices for all graphs.
    """
    combined_edge_index = []
    for i, edge_index in enumerate(edge_indices):
        # Calculate the offset for the current graph
        offset = i * num_nodes_per_graph
        
        # Offset the edge indices
        offset_edge_index = edge_index + offset
        combined_edge_index.append(offset_edge_index)
    
    # Concatenate all edge indices into a single tensor
    combined_edge_index = torch.cat(combined_edge_index, dim=1)
    return combined_edge_index