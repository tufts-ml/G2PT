from datasets import IterableDataset
from rdkit import Chem
from rdkit.Chem import rdchem
from torch.utils.data import Dataset
import torch
from torch_geometric.utils import degree
from torch_geometric.data import Data
from collections import deque
import numpy as np
import os
from torch_geometric.utils import to_networkx
from torch_geometric.utils.convert import from_networkx
import re
from functools import partial
import networkx as nx


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence
        
def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)
    
def get_smiles(mol):
    smiles = mol2smiles(mol)
    if smiles is not None:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)
            return smiles
        except Chem.rdchem.AtomValenceException:
            print("Valence error in GetmolFrags")
            return None
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
            return None
    else:
        return None

def seq_to_nxgraph(seq_str):
    tokens = seq_str.split()

    ctx_start = tokens.index('<boc>') + 1
    ctx_end = tokens.index('<eoc>')
    ctx_tokens = tokens[ctx_start:ctx_end+1]

    id_node_lookup = set()
    for i in range(0, len(ctx_tokens), 3):
        atom_id = ctx_tokens[i + 1]
        id_node_lookup.add(atom_id)

    edge_start = tokens.index('<bog>') + 1
    edge_end = tokens.index('<eog>')
    edge_tokens = [token for token in tokens[edge_start:edge_end] if token != '<sepg>']
    edges = []
    for i in range(0, len(edge_tokens), 3):
        src_id = edge_tokens[i]
        dest_id = edge_tokens[i + 1]
        
        if src_id in id_node_lookup and dest_id in id_node_lookup and src_id != dest_id:
            edges.append((src_id, dest_id))
    G = nx.from_edgelist(edges)
    return G

def seq_to_mol(seq_str):
    tokens = seq_str.split()
    mol = Chem.RWMol()

    ctx_start = tokens.index('<boc>') + 1
    ctx_end = tokens.index('<eoc>')
    ctx_tokens = tokens[ctx_start:ctx_end+1]

    id_atom_lookup = {}
    for i in range(0, len(ctx_tokens), 3):
        atom_type = ctx_tokens[i]
        atom_id = ctx_tokens[i + 1]
        atomic_symbol = atom_type.split('_')[1]
        atomic_num = Chem.Atom(atomic_symbol).GetAtomicNum()
        mol.AddAtom(Chem.Atom(atomic_num))
        id_atom_lookup[atom_id] = mol.GetNumAtoms() - 1

    # Extract bond tokens
    bond_start = tokens.index('<bog>') + 1
    bond_end = tokens.index('<eog>')
    bond_tokens = [token for token in tokens[bond_start:bond_end] if token != '<sepg>']

    for i in range(0, len(bond_tokens), 3):
        src_id = bond_tokens[i]
        dest_id = bond_tokens[i + 1]
        bond_type = bond_tokens[i + 2]
        bond_type_rdkit = {
            'BOND_SINGLE': rdchem.BondType.SINGLE,
            'BOND_DOUBLE': rdchem.BondType.DOUBLE,
            'BOND_TRIPLE': rdchem.BondType.TRIPLE,
            'BOND_AROMATIC': rdchem.BondType.AROMATIC
        }[bond_type]
        
        if src_id in id_atom_lookup and dest_id in id_atom_lookup:
            mol.AddBond(id_atom_lookup[src_id], id_atom_lookup[dest_id], bond_type_rdkit)

    return mol

def seq_to_molecule_with_partial_charges(seq_str):
    ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

    tokens = seq_str.split()
    mol = Chem.RWMol()

    ctx_start = tokens.index('<boc>') + 1
    ctx_end = tokens.index('<eoc>')
    ctx_tokens = tokens[ctx_start:ctx_end+1]

    id_atom_lookup = {}
    for i in range(0, len(ctx_tokens), 3):
        atom_type = ctx_tokens[i]
        atom_id = ctx_tokens[i + 1]
        atomic_symbol = atom_type.split('_')[1]
        atomic_num = Chem.Atom(atomic_symbol).GetAtomicNum()
        mol.AddAtom(Chem.Atom(atomic_num))
        id_atom_lookup[atom_id] = mol.GetNumAtoms() - 1

    # Extract bond tokens
    bond_start = tokens.index('<bog>') + 1
    bond_end = tokens.index('<eog>')
    bond_tokens = [token for token in tokens[bond_start:bond_end] if token != '<sepg>']

    for i in range(0, len(bond_tokens), 3):
        src_id = bond_tokens[i]
        dest_id = bond_tokens[i + 1]
        bond_type = bond_tokens[i + 2]
        bond_type_rdkit = {
            'BOND_SINGLE': rdchem.BondType.SINGLE,
            'BOND_DOUBLE': rdchem.BondType.DOUBLE,
            'BOND_TRIPLE': rdchem.BondType.TRIPLE,
            'BOND_AROMATIC': rdchem.BondType.AROMATIC
        }[bond_type]
        
        if src_id in id_atom_lookup and dest_id in id_atom_lookup:
            mol.AddBond(id_atom_lookup[src_id], id_atom_lookup[dest_id], bond_type_rdkit)
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol

class LobsterDataset(Dataset):
    def __init__(self, num_data, process_fn=lambda x: x, min_node = 10, max_node=100):
        self.num_data = num_data
        self.min_node = min_node
        self.max_node = max_node
        self.process_fn = process_fn
        self.indices = torch.randperm(num_data)
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        if idx == len(self): 
            raise IndexError 
        while True:
            G = nx.random_lobster(int((self.min_node+self.max_node)/2), 0.7, 0.7)
            if len(G.nodes()) >= self.min_node and len(G.nodes()) <= self.max_node:
                break
        pyg = from_networkx(G)
        X = torch.ones(pyg.num_nodes, 1, dtype=torch.float)
        edge_attr = torch.zeros(pyg.edge_index.shape[-1], 2, dtype=torch.float)
        return self.process_fn({'x': X, 'edge_index': pyg.edge_index, 'edge_attr': edge_attr})

class NumpyBinDataset(Dataset):
    def __init__(self, path, num_data, num_node_class, num_edge_calss, shape, process_fn=lambda x: x):
        self.path = path
        self.num_data = num_data
        self.num_node_class = num_node_class
        self.num_edge_calss = num_edge_calss

        self.process_fn = process_fn
        
        self.xs = np.memmap(os.path.join(path, 'xs.bin'), dtype=np.int16, mode='r', shape=shape['x'])
        self.edge_indices = np.memmap(os.path.join(path, 'edge_indices.bin'), dtype=np.int16, mode='r', shape=shape['edge_index'])
        self.edge_attrs = np.memmap(os.path.join(path, 'edge_attrs.bin'), dtype=np.int16, mode='r', shape=shape['edge_attr'])
        self.indices = torch.randperm(num_data)
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.xs[idx]).astype(np.int64))
        x = torch.nn.functional.one_hot(x[x!=-100], num_classes=self.num_node_class)
        edge_index = torch.from_numpy(np.array(self.edge_indices[idx]).astype(np.int64))
        edge_index = edge_index[edge_index!=-100].reshape(2, -1)
        
        edge_attr = torch.from_numpy(np.array(self.edge_attrs[idx]).astype(np.int64))
        edge_attr = torch.nn.functional.one_hot(edge_attr[edge_attr!=-100], num_classes=self.num_edge_calss)
        
        return self.process_fn({'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr})

def randperm_node(x, edge_index):
    num_nodes = x.shape[0]

    perm = torch.randperm(num_nodes)

    # Create a mapping from old node indices to new node indices
    mapping = torch.empty_like(perm)
    mapping[perm] = torch.arange(num_nodes)

    # Permute node features
    new_x = x[perm]
    # Update edge indices using the mapping
    new_edge_index = mapping[edge_index]

    return new_x, new_edge_index

def remove_edge_with_attr(graph, edge_to_remove):
    """
    Remove an edge and its attributes from a PyTorch Geometric graph.
    
    Args:
        graph (torch_geometric.data.Data): Input graph.
        edge_to_remove (tuple): Edge to remove, specified as (source, target).
        
    Returns:
        torch_geometric.data.Data: Graph with the specified edge and its attributes removed.
    """
    new_graph = graph.clone()
    edge_index = new_graph.edge_index
    edge_attr = new_graph.edge_attr

    # Find edges to keep
    mask1 = ~((edge_index[0] == edge_to_remove[0]) & (edge_index[1] == edge_to_remove[1]))
    mask2 = ~((edge_index[1] == edge_to_remove[0]) & (edge_index[0] == edge_to_remove[1]))
    mask = mask1.logical_and(mask2)
    # Apply the mask to edge_index and edge_attr
    new_edge_index = edge_index[:, mask]
    if edge_attr is not None:
        new_edge_attr = edge_attr[mask]
    else:
        new_edge_attr = None
    if len(edge_attr.shape) == 2:# one hot
        poped_edge_attr = edge_attr[~mask1].argmax().item()
    else:
        poped_edge_attr = edge_attr[~mask1].item()
    # Update the graph
    new_graph.edge_index = new_edge_index
    new_graph.edge_attr = new_edge_attr
    return new_graph, poped_edge_attr
   
def bfs_with_all_edges(G, source):
    visited = set()
    edges = set()
    edges_bfs = []

    queue = deque([source])
    visited.add(source)

    while queue:
        node = queue.popleft()
        for neighbor in G[node]:
            if neighbor not in visited:
                edges.add(tuple(sorted((node, neighbor))))
                edges_bfs.append((node, neighbor))

                visited.add(neighbor)
                queue.append(neighbor)
            else:
                if tuple(sorted((neighbor, node))) not in edges:
                    edges.add(tuple(sorted((neighbor, node))))
                    edges_bfs.append((node, neighbor))

    return  edges_bfs

def to_seq_by_bfs(data, atom_type, bond_type):
    
    x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
    x, edge_index = randperm_node(x, edge_index)
    ctx = [['<sepc>', atom_type[node_type.item()], f'IDX_{node_idx}'] for node_idx, node_type in enumerate(x.argmax(-1))]
    ctx = sum(ctx, [])
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    outputs = []
    
    G = to_networkx(data)

    #get edge order from dfs,begin from node 0, G is nx graph
    # _,edges_order_dfs = dfs_with_all_edges(G,0)
    edges_order_bfs = bfs_with_all_edges(G,0)
    for selected_source_node_idx, selected_dest_node_idx in edges_order_bfs:
        #get_edge_attr
        edge_mask = ((data.edge_index[0] == selected_source_node_idx) & (data.edge_index[1] == selected_dest_node_idx)) | \
            ((data.edge_index[0] == selected_dest_node_idx) & (data.edge_index[1] == selected_source_node_idx))  
        edge_indices = edge_mask.nonzero(as_tuple=True)[0]
        if len(edge_indices) > 0:
            removed_edge_type = data.edge_attr[edge_indices][0].argmax().item()
        outputs.append(['<sepg>', f'IDX_{selected_source_node_idx}', f'IDX_{selected_dest_node_idx}', bond_type[removed_edge_type-1]])

    ctx[0] = '<boc>'
    ctx.append('<eoc>')
    outputs = sum(outputs,[])
    outputs[0] = '<bog>'
    outputs.append('<eog>')
    
    return {"text": [" ".join(ctx + outputs)]}

def to_seq_by_deg(data, atom_type, bond_type):
    
    x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
    x, edge_index = randperm_node(x, edge_index)
    num_nodes = x.shape[0]

    ctx = [['<sepc>', atom_type[node_type.item()], f'IDX_{node_idx}'] for node_idx, node_type in enumerate(x.argmax(-1))]
    ctx = sum(ctx, [])
    data_t = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    outputs = []
    INF = 100
    while True:
        source_nodes_t = data_t.edge_index[0]
        node_degrees_t = degree(source_nodes_t, num_nodes=num_nodes)
        if torch.all(node_degrees_t==0):
            break
        node_degrees_t[node_degrees_t==0] = INF
        # sample a source node with minimum deg
        candidate_source_nodes = torch.where(node_degrees_t==node_degrees_t.min())[0]
        selected_index = torch.randint(0, candidate_source_nodes.shape[0], (1,)).item()
        selected_source_node_idx = candidate_source_nodes[selected_index].item()
        
        # get the dest node with minimum deg 
        source_node_mask = source_nodes_t==selected_source_node_idx
        candidate_dest_nodes = data_t.edge_index[1][source_node_mask].unique()
        
        candidate_dest_degrees = node_degrees_t[candidate_dest_nodes]
        min_dest_degree = candidate_dest_degrees.min()
        
        indices = torch.where(candidate_dest_degrees == min_dest_degree)[0]
        selected_index = indices[torch.randint(0, len(indices), (1,)).item()]
        selected_dest_node_idx = candidate_dest_nodes[selected_index].item()

        # get new graph at t-1
        data_tminus1, removed_edge_type = remove_edge_with_attr(data_t, (selected_source_node_idx, selected_dest_node_idx))
        # selected_source_node_type = data.x[selected_source_node_idx].argmax(-1).item()
        # selected_dest_node_type = data.x[selected_dest_node_idx].argmax(-1).item()
        outputs.append(['<sepg>', f'IDX_{selected_source_node_idx}', f'IDX_{selected_dest_node_idx}', bond_type[removed_edge_type-1]])
        data_t = data_tminus1
        
    ctx[0] = '<boc>'
    ctx.append('<eoc>')
    outputs = outputs[::-1]
    outputs = sum(outputs,[])
    outputs[0] = '<bog>'
    outputs.append('<eog>')
    return {"text": [" ".join(ctx + outputs)]}

def get_datasets(dataset_name, tokenizer, order='bfs'):
    if order == 'bfs':
        order_function = to_seq_by_bfs
    elif order == 'deg':
        order_function = to_seq_by_deg
    else:
        raise NotImplementedError(f"Order function {order} is not implemented")
    
    def pre_tokenize_function(examples, atom_type, bond_type):
        data = order_function(examples, atom_type, bond_type)
        data = tokenizer(data['text'],padding='max_length', return_tensors='pt')
        data['input_ids'] = data['input_ids'].squeeze(0)
        data['attention_mask'] = data['attention_mask'].squeeze(0)
        data['labels'] = data['input_ids'].clone()
        return data
    
    if dataset_name == 'lobster':
        ATOM_TYPE = ['NODE']
        BOND_TYPE = ['EDGE']    
    
        train_datasets = LobsterDataset(num_data=256,
                                      process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))
        eval_datasets = LobsterDataset(num_data=64,
                                     process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))
        
        return train_datasets, eval_datasets
    
    if dataset_name == 'moses':
        ATOM_TYPE = ['ATOM_C', 'ATOM_N', 'ATOM_S', 'ATOM_O', 'ATOM_F', 'ATOM_Cl', 'ATOM_Br', 'ATOM_H']
        BOND_TYPE = ['BOND_SINGLE', 'BOND_DOUBLE', 'BOND_TRIPLE', 'BOND_AROMATIC']    
    
        train_shape = {'x': (1419512, 27), 'edge_index': (1419512, 2, 62), 'edge_attr': (1419512, 62)}
        eval_shape = {'x': (156176, 27), 'edge_index': (156176, 2, 62), 'edge_attr': (156176, 62)}
        
    elif dataset_name == 'guacamol':
        ATOM_TYPE = ['ATOM_C', 'ATOM_N', 'ATOM_O', 'ATOM_F', 'ATOM_B', 'ATOM_Br', 'ATOM_Cl', 'ATOM_I', 'ATOM_P', 'ATOM_S', 'ATOM_Se', 'ATOM_Si']
        BOND_TYPE = ['BOND_SINGLE', 'BOND_DOUBLE', 'BOND_TRIPLE', 'BOND_AROMATIC']    
    
        train_shape = {'x': (1118633, 88), 'edge_index': (1118633, 2, 174), 'edge_attr': (1118633, 174)}
        eval_shape = {'x': (69926, 76), 'edge_index': (69926, 2, 158), 'edge_attr': (69926, 158)}
        
    elif dataset_name == 'qm9':
        ATOM_TYPE = ['ATOM_C', 'ATOM_N', 'ATOM_O', 'ATOM_F']
        BOND_TYPE = ['BOND_SINGLE', 'BOND_DOUBLE', 'BOND_TRIPLE', 'BOND_AROMATIC']    
    
        train_shape = {'x': (97732, 9), 'edge_index': (97732, 2, 28), 'edge_attr': (97732, 28)}
        eval_shape = {'x': (20042, 9), 'edge_index': (20042, 2, 26), 'edge_attr': (20042, 26)}  
        
    elif dataset_name == 'tree':
        ATOM_TYPE = ['NODE']
        BOND_TYPE = ['EDGE']    
    
        train_shape = {'x': (256, 64), 'edge_index': (256, 2, 126), 'edge_attr': (256, 126)}
        eval_shape = {'x': (64, 64), 'edge_index': (64, 2, 126), 'edge_attr': (64, 126)}
        
    elif dataset_name == 'sbm':
        ATOM_TYPE = ['NODE']
        BOND_TYPE = ['EDGE']    
    
        train_shape = {'x': (256, 187), 'edge_index': (256, 2, 2258), 'edge_attr': (256, 2258)}
        eval_shape = {'x': (64, 172), 'edge_index': (64, 2, 1808), 'edge_attr': (64, 1808)}
    
    elif dataset_name == 'planar':
        ATOM_TYPE = ['NODE']
        BOND_TYPE = ['EDGE']    
    
        train_shape = {'x': (256, 64), 'edge_index': (256, 2, 362), 'edge_attr': (256, 362)}
        eval_shape = {'x': (64, 64), 'edge_index': (64, 2, 362), 'edge_attr': (64, 362)}
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented")
    
    num_train = train_shape['x'][0]
    num_eval = eval_shape['x'][0]    
    train_datasets = NumpyBinDataset(f'./datasets/{dataset_name}/train',     
                                     num_train,
                                    len(ATOM_TYPE), 
                                    len(BOND_TYPE)+1, 
                                    shape=train_shape, 
                                    process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))
    eval_datasets = NumpyBinDataset(f'./datasets/{dataset_name}/eval', 
                                    num_eval, 
                                    len(ATOM_TYPE), 
                                    len(BOND_TYPE)+1, 
                                    shape=eval_shape, 
                                    process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))
    
    return train_datasets, eval_datasets
    
