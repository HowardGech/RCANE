import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from .utils_model import *
import math


class stacked_fc(nn.Module):
    """
    A stacked fully connected layer class that allows flexible stacking
    of layers with the ability to apply activation functions.

    Parameters:
    -----------
    stack_dim : int
        The dimension of the stack, representing how many layers to stack.
    input_dim : int
        The input feature size for each layer.
    output_dim : int
        The output feature size for each layer.
    dropout : float, optional (default: 0)
        Dropout rate to apply to the output tensor.
    activation : nn.Module, optional (default: nn.LeakyReLU(0.1))
        The activation function to apply after the linear transformation.
    use_act : bool, optional (default: True)
        Whether or not to apply the activation function.

    Methods:
    --------
    forward(data):
        Performs the forward pass, computing the linear transformation 
        followed by the optional activation.
    """
    def __init__(self, stack_dim, input_dim, output_dim, dropout=0, activation=nn.LeakyReLU(0.1), use_act=True):
        super(stacked_fc, self).__init__()
        self.activation = activation
        self.use_act = use_act
        self.stack_dim = stack_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Dropout = nn.Dropout(dropout)
        
        # Initialize weights and biases with uniform distribution based on input and output dimensions
        stdv_in = 1. / math.sqrt(input_dim)
        self.w = nn.Parameter(2*stdv_in*torch.rand(stack_dim, input_dim, output_dim) - stdv_in, requires_grad=True)
        self.b = nn.Parameter(2*stdv_in*torch.rand(stack_dim, 1, output_dim) - stdv_in, requires_grad=True)
        
    def forward(self, data):
        """
        Forward pass through the stacked fully connected layer.

        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, stack_dim, input_dim)
            The input tensor to be transformed.

        Returns:
        --------
        torch.Tensor of shape (batch_size, stack_dim, output_dim)
            The output tensor after applying the linear transformation 
            and the optional activation function.
        """
        # Apply the linear transformation
        data = data.view(-1, self.stack_dim, 1, self.input_dim)
        out = torch.matmul(data, self.w) + self.b
        
        # Apply activation if specified
        if self.use_act:
            out = self.activation(out)
            
        # Reshape the output tensor to (batch_size, stack_dim, output_dim)
        out = out.view(-1, self.stack_dim, self.output_dim)
        out = self.Dropout(out)
        return out
    
class ML_stacked_fc(nn.Module):
    """
    A multi-layer stacked fully connected layer class that allows flexible stacking of layers.
    
    Parameters:
    -----------
    stack_dim : int
        The dimension of the stack, representing how many layers to stack.
    input_dim : int
        The input feature size for first layer.
    hidden_dim : int
        The input and output feature size for hidden layers.
    output_dim : int
        The output feature size for final layer.
    depth : int, optional (default: 2)
        Number of layers to stack.
    dropout : float, optional (default: 0)
        Dropout rate to apply to the output tensor.
    activation : nn.Module, optional (default: nn.LeakyReLU(0.1))
        The activation function to apply after the linear transformation.
    use_act : bool, optional (default: True)
        Whether or not to apply the activation function.

    Methods:
    --------
    forward(data):
        Performs the forward pass, computing the linear transformation 
        followed by the optional activation.
        
    """
    def __init__(self, stack_dim, input_dim, hidden_dim, output_dim, depth=2, dropout=0, activation=nn.LeakyReLU(0.1), use_act=True):
        super(ML_stacked_fc, self).__init__()
        if depth == 1:
            self.mlp = stacked_fc(stack_dim, input_dim, output_dim, use_act=False)
        else:
            self.mlp = nn.Sequential(stacked_fc(stack_dim, input_dim, hidden_dim, dropout, activation, use_act), *nn.ModuleList([stacked_fc(stack_dim, hidden_dim, hidden_dim, dropout, activation, use_act) for _ in range(depth-2)]), stacked_fc(stack_dim, hidden_dim, output_dim, use_act=False))
    
    def forward(self, data):
        """
        Forward pass through the multi-layer stacked fully connected layer.
        
        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, stack_dim, input_dim)
            The input tensor to be transformed.

        Returns:
        --------
        torch.Tensor of shape (batch_size, stack_dim, output_dim)
            The output tensor after applying the linear transformation 
            and the optional activation function.
        """
        return self.mlp(data)
    
class stacked_univ_layer(nn.Module):
    """
    A stacked univariate layer with a learnable weight and bias.

    Parameters:
    -----------
    stacked_dim : int
        The dimension of the stack, representing how many layers to stack.
    dropout : float, optional (default: 0)
        Dropout rate to apply to the output tensor.
    activation : nn.Module, optional (default: nn.LeakyReLU(0.1))
        The activation function to be applied.
    use_act : bool, optional (default: True)
        Whether or not to apply the activation function.

    Methods:
    --------
    forward(data):
        Applies the learnable transformation and optional activation function to the data.
    """
    def __init__(self, stacked_dim, dropout=0, activation=nn.LeakyReLU(0.1), use_act=True):
        super(stacked_univ_layer, self).__init__()
        self.stacked_dim = stacked_dim
        self.Dropout = nn.Dropout(dropout)
        
        # Learnable parameters for the linear transformation
        self.w = nn.Parameter(torch.ones(stacked_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(stacked_dim), requires_grad=True)
        
        # Activation function
        self.act = activation
        self.use_act = use_act
        
    def forward(self, x):
        """
        Forward pass through the stacked univariate layer.

        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, stacked_dim)
            Input tensor representing stacked univariate data.

        Returns:
        --------
        torch.Tensor of shape (batch_size, stacked_dim)
            Output after applying the learnable transformation and the optional activation function.
        """
        out = torch.mul(x, self.w) + self.b
        if self.use_act:
            out = self.act(out)
        
        out = self.Dropout(out)
        return out
    
class ML_stacked_univ_layer(nn.Module):
    """
    A multi-layer stacked univariate layer class that allows flexible stacking of layers.
    
    Parameters:
    -----------
    stacked_dim : int
        The dimension of the stack, representing how many layers to stack.
    depth : int, optional (default: 2)
        Number of layers to stack.
    dropout : float, optional (default: 0)
        Dropout rate to apply to the output tensor.
    activation : nn.Module, optional (default: nn.LeakyReLU(0.1))
        The activation function to apply after the linear transformation.
    use_act : bool, optional (default: True)
        Whether or not to apply the activation function.
        
    Methods:
    --------
    forward(data):
        Performs the forward pass, computing the linear transformation 
        followed by the optional activation.
    """
    def __init__(self, stacked_dim, depth=2, dropout=0, activation=nn.LeakyReLU(0.1), use_act=True):
        super(ML_stacked_univ_layer, self).__init__()
        self.mlp = nn.Sequential(*nn.ModuleList([stacked_univ_layer(stacked_dim, dropout, activation, use_act) for _ in range(depth-1)]), stacked_univ_layer(stacked_dim, use_act=False))
        
    def forward(self, data):
        """
        Forward pass through the multi-layer stacked univariate layer.
        
        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, stacked_dim)
            The input tensor to be transformed.

        Returns:
        --------
        torch.Tensor of shape (batch_size, stacked_dim)
            The output tensor after applying the learnable transformation 
            and the optional activation function.
        """
        return self.mlp(data)
    


class activated_att(nn.Module):
    """
    A single layer of the graph attention network with an activation function.
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels (features per node).
    out_channels : int
        Number of output channels (features per node).
    heads : int, optional (default: 1)
        Number of attention heads used in the multi-head attention mechanism.
    dropout : float, optional (default: 0)
        Dropout rate to apply to the attention coefficients.
    activation : nn.Module, optional (default: nn.LeakyReLU(0.1))
        Activation function to apply after the linear transformation.
    use_act : bool, optional (default: True)
        Whether or not to apply the activation function.
        
    Methods:
    --------
    forward(x, edge_index):
        Performs the forward pass of the graph attention layer.
    """
    def __init__(self, in_channels, out_channels, heads=1, dropout=0, activation=nn.LeakyReLU(0.1), use_act=True):
        super(activated_att, self).__init__()
        self.activation = activation
        self.use_act = use_act
        self.att = TransformerConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, combined_data):
        """
        Forward pass through the graph attention layer.
        
        Parameters:
        -----------
        combined_data : tuple of torch.Tensor and torch.long tensor
            Tuple containing the input node feature matrix (x) and edge index.
        x : torch.Tensor of shape (batch_size, num_nodes, in_channels)
            Input node feature matrix.
        edge_index : torch.long tensor of shape (2, num_edges)
            Edge index indicating the connections between nodes.
            
        Returns:
        --------
        combined_data : tuple of torch.Tensor and torch.long tensor
            Tuple containing the output node feature matrix (x) and edge index.
        x : torch.Tensor of shape (batch_size, num_nodes, out_channels)
            Output after applying the graph attention layer.
        edge_index : torch.long tensor of shape (2, num_edges)
            Edge index indicating the connections between nodes.
        """
        x, edge_index = combined_data
        x_out = self.att(x, edge_index)
        if self.use_act:
            x_out = self.activation(x_out)
        return x_out, edge_index
        
    
    
class ML_GATTN(nn.Module):
    """
    Multi-layer Graph Transformer Convolution utilizing the TransformerConv from 
    the PyTorch Geometric library for message passing in graph neural networks.

    Parameters:
    -----------
    in_channels : int
        Number of input channels (features per node).
    hidden_channels : int
        Number of hidden channels for the intermediate layers.
    out_channels : int
        Number of output channels (features per node).
    depth : int, optional (default: 2)
        Number of transformer convolution layers to stack.
    heads : int, optional (default: 1)
        Number of attention heads used in the multi-head attention mechanism.
    dropout : float, optional (default: 0)
        Dropout rate to apply to the attention coefficients.
    activation : nn.Module, optional (default: nn.LeakyReLU(0.1))
        Activation function to apply after the linear transformation.
    use_act : bool, optional (default: True)
        Whether or not to apply the activation function.
        

    Methods:
    --------
    forward(x, graph_index, edge_index_list):
        Performs the forward pass of the GATTN layer on graphs.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, depth=2, heads=1, dropout=0, activation=nn.LeakyReLU(0.1), use_act=True):
        super(ML_GATTN, self).__init__()
        self.activation = activation
        self.use_act = use_act
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize the transformer convolution layers
        if depth == 1:
            self.conv = activated_att(in_channels, out_channels, heads, dropout, use_act=False)
        else:
            self.conv = nn.Sequential(activated_att(in_channels, hidden_channels, heads, dropout, activation, use_act), *nn.ModuleList([activated_att(hidden_channels, hidden_channels, heads, dropout, activation, use_act) for _ in range(depth-2)]), activated_att(hidden_channels, out_channels, heads, use_act=False))
        

    def forward(self, x, edge_index_list):
        """
        Forward pass through the graph transformer convolution layer.
        
        Parameters:
        -----------
        x : torch.Tensor of shape (batch_size, num_nodes, in_channels)
            Input node feature matrix.
        edge_index_list : list of torch.Tensor of shape (2, num_edges)
            A list of edge indices for each graph.
            
        Returns:
        --------
        torch.Tensor of shape (batch_size, num_nodes, out_channels)
            Output after applying the graph transformer convolution.
        
        """
        # Flatten the batch and nodes for processing
        batch_size, num_nodes, _ = x.size()
        x = x.view(batch_size * num_nodes, self.in_channels)
        
        # Combine the edge indices for all graphs
        flat_edge_index = combine_edge_indices(edge_index_list, num_nodes)
        
        # Perform the convolution
        x, _ = self.conv((x, flat_edge_index))
        
        # Reshape back to shape (batch_size, num_nodes, out_channels)
        x = x.view(batch_size, num_nodes, -1)
        return x
    
class Perceptron(nn.Module):
    """
    A simple perceptron class with a single linear transformation and optional activation function.
    
    Parameters:
    -----------
    input_dim : int
        The input feature size.
    output_dim : int
        The output feature size.
    dropout : float, optional (default: 0)
        Dropout rate to apply to the output tensor.
    activation : nn.Module, optional (default: nn.LeakyReLU(0.1))
        The activation function to apply after the linear transformation.
    use_act : bool, optional (default: True)
        Whether or not to apply the activation function.
        
    Methods:
    --------
    forward(data):
        Performs the forward pass, computing the linear transformation 
        followed by the optional activation.
    """
    def __init__(self, input_dim, output_dim, dropout=0, activation=nn.LeakyReLU(0.1), use_act=True):
        super(Perceptron, self).__init__()
        self.use_act = use_act
        self.Dropout = nn.Dropout(dropout)
        if use_act:
            self.fc = nn.Sequential(nn.Linear(input_dim, output_dim), activation)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
            
    def forward(self, data):
        """
        Forward pass through the perceptron layer.
        
        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, input_dim)
            The input tensor to be transformed.
            
        Returns:
        --------
        torch.Tensor of shape (batch_size, output_dim)
            The output tensor after applying the linear transformation 
            and the optional activation function.
        """
        return self.Dropout(self.fc(data))
    
        
    
class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) class that allows
    flexible stacking of layers with the ability to apply activation functions.
    
    Parameters:
    -----------
    input_dim : int
        The input feature size for the first layer.
    hidden_dim : int
        The input and output feature size for hidden layers.
    output_dim : int
        The output feature size for the final layer.
    depth : int, optional (default: 2)
        Number of layers to stack.
    dropout : float, optional (default: 0)
        Dropout rate to apply to the output tensor.
    activation : nn.Module, optional (default: nn.LeakyReLU(0.1))
        The activation function to apply after the linear transformation.
    use_act : bool, optional (default: True)
        Whether or not to apply the activation function.
        
    Methods:
    --------
    forward(data):
        Performs the forward pass, computing the linear transformation 
        followed by the optional activation.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2, dropout=0, activation=nn.LeakyReLU(0.1), use_act=True):
        super(MLP, self).__init__()
        if depth == 1:
            self.mlp = nn.Linear(input_dim, output_dim)
        else:
            self.mlp = nn.Sequential(Perceptron(input_dim, hidden_dim, dropout, activation, use_act), *nn.ModuleList([Perceptron(hidden_dim, hidden_dim, dropout, activation, use_act) for _ in range(depth-2)]), Perceptron(hidden_dim, output_dim, use_act=False))
        
    def forward(self, data):
        """
        Forward pass through the multi-layer perceptron.
        
        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, input_dim)
            The input tensor to be transformed.
            
        Returns:
        --------
        torch.Tensor of shape (batch_size, output_dim)
            The output tensor after applying the linear transformation 
            and the optional activation function.
        """
        return self.mlp(data)


    
class RCANE(nn.Module):
    """
    RNA to Copy Number Alteration Network (RCANE) model for pan-cancer prediction of somatic copy number alterations from RNA expression data.
    
    This model consists of the following components:
    - RNA MLP: A multi-layer perceptron for processing RNA expression data.
    - Weighted Sum: A weighted sum layer for combining RNA and cancer type data.
    - Concatenation MLP: A multi-layer perceptron for processing concatenated RNA and cancer type data.
    - LSTM: A bidirectional LSTM layer for processing the data in each chromosome segment.
    - Graph Attention Network (GATTN): A graph attention network for processing the data in each chromosome segment.
    - Output MLP: A multi-layer perceptron for processing the output.
    - Debiaser: A debiasing layer for reducing bias in the output predictions.
    
    Parameters:
    -----------
    param_dict : dict
        Dictionary containing the hyperparameters for the model.
        
    Methods:
    --------
    forward(rna_exp, cohort, mask):
        Performs the forward pass of the RCANE model on the input data.
    """
    def __init__(self, param_dict):
        super(RCANE, self).__init__()
        
        self.seg_nums = param_dict["seg_nums"]
        self.genes_in_seg = param_dict["genes_in_seg"]
        chr_len = torch.Tensor(param_dict["chr_index"]).long()
        seg_chr_index = torch.repeat_interleave(torch.arange(len(chr_len)),chr_len).long()
        self.chr_index = [torch.tensor(torch.where(seg_chr_index==i)[0]) for i in range(len(chr_len))]
        self.total_chrs = len(chr_len)
        self.cancer_types = param_dict["cancer_types"]
        self.total_cohorts = len(self.cancer_types)
        
        self.dropout = param_dict["dropout"] if "dropout" in param_dict else 0
        self.activation = getattr(nn, param_dict["activation"])(**param_dict["activation_args"]) if "activation" in param_dict else nn.LeakyReLU(0.1)
        self.use_act = param_dict["use_act"] if "use_act" in param_dict else True
        self.include_offset = param_dict["include_offset"] if "include_offset" in param_dict else True
        self.include_layernorm = param_dict["include_layernorm"] if "include_layernorm" in param_dict else True
        self.include_LSTM = param_dict["include_LSTM"] if "include_LSTM" in param_dict else True
        self.include_GATTN = param_dict["include_GATTN"] if "include_GATTN" in param_dict else True
        self.include_debiaser = param_dict["include_debiaser"] if "include_debiaser" in param_dict else True
        self.replace_MLP = (not self.include_LSTM and not self.include_GATTN)
        
        #Hyperparameters for the embedding layer and cancer type MLP
        self.emb_out = param_dict["emb_out"] if "emb_out" in param_dict else 32
        self.mlp_offset_depth = param_dict["mlp_offset_depth"] if "mlp_offset_depth" in param_dict else 2
        self.mlp_offset_hidden = param_dict["mlp_offset_hidden"] if "mlp_offset_hidden" in param_dict else self.emb_out
        self.mlp_offset_activation = param_dict["mlp_offset_activation"] if "mlp_offset_activation" in param_dict else self.activation
        self.mlp_offset_use_act = param_dict["mlp_offset_use_act"] if "mlp_offset_use_act" in param_dict else self.use_act
        self.mlp_offset_dropout = param_dict["mlp_offset_dropout"] if "mlp_offset_dropout" in param_dict else self.dropout
        self.mlp_cohort_depth = param_dict["mlp_cohort_depth"] if "mlp_cohort_depth" in param_dict else 2
        self.mlp_cohort_hidden = param_dict["mlp_cohort_hidden"] if "mlp_cohort_hidden" in param_dict else self.emb_out
        self.mlp_cohort_activation = param_dict["mlp_cohort_activation"] if "mlp_cohort_activation" in param_dict else self.activation
        self.mlp_cohort_use_act = param_dict["mlp_cohort_use_act"] if "mlp_cohort_use_act" in param_dict else self.use_act
        self.mlp_cohort_dropout = param_dict["mlp_cohort_dropout"] if "mlp_cohort_dropout" in param_dict else self.dropout
        
        # Hyperparameters for the RNA MLP
        self.mlp_rna_depth = param_dict["mlp_rna_depth"] if "mlp_rna_depth" in param_dict else 2
        self.mlp_rna_out = param_dict["mlp_rna_out"] if "mlp_rna_out" in param_dict else 32
        self.mlp_rna_hidden = param_dict["mlp_rna_hidden"] if "mlp_rna_hidden" in param_dict else self.mlp_rna_out
        self.mlp_rna_activation = param_dict["mlp_rna_activation"] if "mlp_rna_activation" in param_dict else self.activation
        self.mlp_rna_use_act = param_dict["mlp_rna_use_act"] if "mlp_rna_use_act" in param_dict else self.use_act
        self.mlp_rna_dropout = param_dict["mlp_rna_dropout"] if "mlp_rna_dropout" in param_dict else self.dropout
        
        # Hyperparameters for the Concatenation MLP
        self.mlp_concat_depth = param_dict["mlp_concat_depth"] if "mlp_concat_depth" in param_dict else 2
        self.mlp_concat_activation = param_dict["mlp_concat_activation"] if "mlp_concat_activation" in param_dict else self.activation
        self.mlp_concat_use_act = param_dict["mlp_concat_use_act"] if "mlp_concat_use_act" in param_dict else self.use_act
        self.mlp_concat_dropout = param_dict["mlp_concat_dropout"] if "mlp_concat_dropout" in param_dict else self.dropout
        self.mlp_concat_out = param_dict["mlp_concat_out"] if "mlp_concat_out" in param_dict else 64
        self.mlp_concat_hidden = param_dict["mlp_concat_hidden"] if "mlp_concat_hidden" in param_dict else self.mlp_concat_out
        
        # Hyperparameters for the LSTM
        self.lstm_depth = param_dict["lstm_depth"] if "lstm_depth" in param_dict else 2
        self.lstm_dropout = param_dict["lstm_dropout"] if "lstm_dropout" in param_dict else self.dropout
        self.lstm_out = param_dict["lstm_out"] if "lstm_out" in param_dict else 32
        self.lstm_activation = param_dict["lstm_activation"] if "lstm_activation" in param_dict else self.activation
        self.lstm_use_act = param_dict["lstm_use_act"] if "lstm_use_act" in param_dict else self.use_act
        
        # Hyperparameters for the GATTN
        self.gattn_out = param_dict["gattn_out"] if "gattn_out" in param_dict else 32
        self.gattn_hidden = param_dict["gattn_hidden"] if "gattn_hidden" in param_dict else self.gattn_out
        self.gattn_head = param_dict["gattn_head"] if "gattn_head" in param_dict else 2
        self.gattn_depth = param_dict["gattn_depth"] if "gattn_depth" in param_dict else 2
        self.gattn_activation = param_dict["gattn_activation"] if "gattn_activation" in param_dict else self.activation
        self.gattn_use_act = param_dict["gattn_use_act"] if "gattn_use_act" in param_dict else self.use_act
        self.gattn_dropout = param_dict["gattn_dropout"] if "gattn_dropout" in param_dict else self.dropout
        
        # Hyperparameters for the output MLP
        self.mlp_out_input = self.mlp_concat_out if self.replace_MLP else 2*self.lstm_out + 2*self.gattn_out
        self.mlp_out_hidden = param_dict["mlp_out_hidden"] if "mlp_out_hidden" in param_dict else self.mlp_out_input
        self.mlp_out_activation = param_dict["mlp_out_activation"] if "mlp_out_activation" in param_dict else self.activation
        self.mlp_out_use_act = param_dict["mlp_out_use_act"] if "mlp_out_use_act" in param_dict else self.use_act
        self.mlp_out_dropout = param_dict["mlp_out_dropout"] if "mlp_out_dropout" in param_dict else self.dropout
        self.mlp_out_depth = param_dict["mlp_out_depth"] if "mlp_out_depth" in param_dict else 1
        
        # Hyperparameters for the debiaser
        self.debiaser_depth = param_dict["debiaser_depth"] if "debiaser_depth" in param_dict else 3
        
        
        # Initialize the model components
        
        self.mlp_rna = ML_stacked_fc(self.seg_nums*self.genes_in_seg, 1, self.mlp_rna_hidden, self.mlp_rna_out, self.mlp_rna_depth, self.mlp_rna_dropout, self.mlp_rna_activation, self.mlp_rna_use_act)
        self.mlp_concat = ML_stacked_fc(self.seg_nums, self.mlp_rna_out, self.mlp_concat_hidden, self.mlp_concat_out, self.mlp_concat_depth, self.mlp_concat_dropout, self.mlp_concat_activation, self.mlp_concat_use_act)
                    
        self.weights = nn.Parameter(torch.rand(self.seg_nums, self.genes_in_seg+1 ,1), requires_grad=True)
        self.embedding = nn.Embedding(self.total_cohorts, self.emb_out)
        self.mlp_cohort = MLP(self.emb_out, self.mlp_cohort_hidden, self.mlp_rna_out, self.mlp_cohort_depth, self.mlp_cohort_dropout, self.mlp_cohort_activation, self.mlp_cohort_use_act)
        if self.include_offset:
            self.mlp_offset = MLP(self.emb_out, self.mlp_offset_hidden, self.seg_nums*self.genes_in_seg, self.mlp_offset_depth, self.mlp_offset_dropout, self.mlp_offset_activation, self.mlp_offset_use_act)
        
        if self.include_layernorm:
            self.layernorm = nn.LayerNorm(self.mlp_rna_out)
        
        if self.include_LSTM:
            self.lstms = nn.ModuleList([nn.LSTM(self.mlp_concat_out, self.lstm_out, self.lstm_depth, batch_first=True,bidirectional=True,dropout=self.lstm_dropout) for _ in range(self.total_chrs)])
        
        if self.include_GATTN:
            self.gattn_pos = ML_GATTN(self.mlp_concat_out, self.gattn_hidden, self.gattn_out, self.gattn_depth, self.gattn_head, self.gattn_dropout, self.gattn_activation, self.gattn_use_act)
            self.gattn_neg = ML_GATTN(self.mlp_concat_out, self.gattn_hidden, self.gattn_out, self.gattn_depth, self.gattn_head, self.gattn_dropout, self.gattn_activation, self.gattn_use_act)
        
        self.mlp_out = MLP(self.mlp_out_input, self.mlp_out_hidden, 1, self.mlp_out_depth, self.mlp_out_dropout, self.mlp_out_activation, self.mlp_out_use_act)
        self.debiaser = ML_stacked_univ_layer(self.seg_nums, self.debiaser_depth, self.mlp_out_dropout, self.mlp_out_activation, self.mlp_out_use_act)            
        self.sm = nn.Softmax(dim=2)
        
    def forward(self, rna_exp, cohort, pos_edge_list, neg_edge_list, mask):
        """
        Forward pass of the RCANE model on the input data.
        
        Parameters:
        -----------
        rna_exp : torch.Tensor of shape (batch_size, seg_nums, genes_in_seg)
            The input RNA expression data.
        cohort : torch.long tensor of shape (batch_size)
            The input cancer cohort data.
        pos_edge_list : list of torch.Tensor of shape (2, num_edges)
            The positive edge indices for the graph attention network.
        neg_edge_list : list of torch.Tensor of shape (2, num_edges)
            The negative edge indices for the graph attention network.
        mask : torch.Tensor of shape (batch_size, seg_nums, genes_in_seg)
            The input mask data. Values of 0 indicate non-masked data, while values of -inf indicate masked data.
            
        Returns:
        --------
        torch.Tensor of shape (batch_size, seg_nums)
            The output tensor of copy number intensity prediction after applying the RCANE model.
        """
        cohort_mask = torch.zeros((rna_exp.size(0), self.seg_nums, 1, 1)).to(rna_exp.device)
        rna_mask = mask.view(-1, self.seg_nums, self.genes_in_seg, 1)
        rna = rna_exp.view(-1, self.seg_nums*self.genes_in_seg)
        cohort_emb = self.embedding(cohort)
        if self.include_offset:
            cohort_offset = self.mlp_offset(cohort_emb.view(-1, self.emb_out))
            rna = rna + cohort_offset
        cohort_mlp = self.mlp_cohort(cohort_emb)
        cohort_mlp = cohort_mlp.view(-1, 1, 1, self.mlp_rna_out).repeat(1, self.seg_nums, 1, 1)
        rna = rna.view(-1, self.genes_in_seg*self.seg_nums, 1)
        rna = self.mlp_rna(rna)
        rna = rna.view(-1, self.seg_nums, self.genes_in_seg, self.mlp_rna_out)
        concat_data = torch.cat((rna,cohort_mlp), 2)
        if self.include_layernorm:
            concat_data = self.layernorm(concat_data)
        concat_mask = torch.cat((rna_mask,cohort_mask), 2)
        masked_weight = self.sm(self.weights + concat_mask)
        concat_data = concat_data * masked_weight
        concat_data = torch.sum(concat_data, 2).view(-1, self.seg_nums, self.mlp_rna_out)
        concat_data = self.mlp_concat(concat_data)
        if self.include_LSTM:
            lstm_data = torch.zeros(rna.size(0),rna.size(1), 2*self.lstm_out).to(rna_exp.device)
            for i in range(23):
                lstm_data[:, self.chr_index[i], :], _ = self.lstms[i](concat_data[:, self.chr_index[i], :])
            lstm_data = self.lstm_activation(lstm_data)
        else:
            lstm_data = torch.zeros(rna.size(0),rna.size(1), 2*self.lstm_out).to(rna_exp.device)
        if self.include_GATTN:
            gattn_pos_data = self.gattn_pos(concat_data, pos_edge_list)
            gattn_neg_data = self.gattn_neg(concat_data, neg_edge_list)
        else:
            gattn_pos_data = torch.zeros(rna.size(0),rna.size(1), self.gattn_out).to(rna_exp.device)
            gattn_neg_data = torch.zeros(rna.size(0),rna.size(1), self.gattn_out).to(rna_exp.device)
        if self.replace_MLP:
            ms_data = concat_data
        else:
            ms_data = torch.cat((lstm_data, gattn_pos_data, gattn_neg_data), 2)
        ms_data = self.mlp_out(ms_data)
        ms_data = ms_data.view(-1,self.seg_nums)
        if self.include_debiaser:
            ms_data = self.debiaser(ms_data)
        return ms_data
    
