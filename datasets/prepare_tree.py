from types import SimpleNamespace
from digress_datasets import spectre_dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import json
CFG = SimpleNamespace(
    dataset=SimpleNamespace(
        name='tree',
        datadir='./datasets/tree/tree_pyg/',
        filter=True,
    ),
    train=SimpleNamespace(
        batch_size=32,
        num_workers=1,
    ),
    general=SimpleNamespace(
        name='none'
    )
)


if __name__ == '__main__':
    datamodule = spectre_dataset.SpectreGraphDataModule(CFG)
    data_meta = {}
    dataset_split = {
        'train': datamodule.train_dataset,
        'eval': datamodule.val_dataset
    }
    
    for split in dataset_split:
        os.makedirs(f'tree/{split}')
        xs = []
        edge_indices = []
        edge_attrs = []
        for data in tqdm(dataset_split[split]):
            xs.append(data.x.argmax(-1))
            edge_indices.append(data.edge_index.transpose(0,1))
            edge_attrs.append(data.edge_attr.argmax(-1))

        xs = pad_sequence(xs, batch_first=True, padding_value=-100).numpy()
        edge_indices = pad_sequence(edge_indices, batch_first=True, padding_value=-100).transpose(2,1).numpy()
        edge_attrs = pad_sequence(edge_attrs, batch_first=True, padding_value=-100).numpy()
            
        xs_data = np.memmap(f'tree/{split}/xs.bin', dtype=np.int16, mode='w+', shape=xs.shape)
        xs_data[:] = xs.astype(np.int16)
        xs_data.flush()

        edge_indices_data = np.memmap(f'tree/{split}/edge_indices.bin', dtype=np.int16, mode='w+', shape=edge_indices.shape)
        edge_indices_data[:] = edge_indices.astype(np.int16)
        edge_indices_data.flush()

        edge_attrs_data = np.memmap(f'tree/{split}/edge_attrs.bin', dtype=np.int16, mode='w+', shape=edge_attrs.shape)
        edge_attrs_data[:] = edge_attrs.astype(np.int16)
        edge_attrs_data.flush()
        data_meta[f'{split}_shape'] = {
            'xs': xs.shape,
            'edge_indices': edge_indices.shape,
            'edge_attrs': edge_attrs.shape
        }
    
    json.dump(data_meta, open('tree/data_meta.json', 'w'))