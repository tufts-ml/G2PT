import os
import os.path as osp
import pickle
from typing import Any, Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from src import metrics
from .utils import to_dense
from tqdm import tqdm


def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(
        to_keep,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=len(to_keep),
    )
    return Data(
        x=data.x[to_keep] - 1,  # Shift onehot encoding to match atom decoder
        charges=data.charges[to_keep],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
    )


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def files_exist(files) -> bool:
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class Statistics:
    def __init__(
        self,
        num_nodes,
        atom_types,
        bond_types,
        degree_hist=None,
        charge_types=None,
        valencies=None,
    ):
        self.num_nodes = num_nodes
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.degree_hist=degree_hist
        self.charge_types = charge_types
        self.valencies = valencies


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


def compute_reference_metrics(dataset_infos, datamodule):
    ref_metrics_path = os.path.join(
        datamodule.train_dataloader().dataset.processed_dir, f"ref_metrics.pkl"
    )

    # Only compute the reference metrics if they haven't been computed already
    if not os.path.exists(ref_metrics_path):
        print("Reference metrics not found. Computing them now.")
        # Transform the training dataset into a list of graphs in the appropriate format
        training_graphs = []
        print("Converting training dataset to placeholders.")
        for batch in tqdm(datamodule.train_dataloader()):
            training_graphs.append(
                to_dense(batch, dataset_infos).collapse(
                    dataset_infos.collapse_charges
                    if hasattr(dataset_infos, "collapse_charges")
                    else None
                )
            )

        print("Computing validation reference metrics.")
        val_sampling_metrics = metrics.sampling_metrics.SamplingMetrics(
            dataset_infos=dataset_infos,
            test=False,
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.val_dataloader(),
        )
        val_reference_metrics = val_sampling_metrics.domain_metrics.forward(
            training_graphs, current_epoch=None, local_rank=0
        )
        print("Computing test reference metrics.")
        test_sampling_metrics = metrics.sampling_metrics.SamplingMetrics(
            dataset_infos=dataset_infos,
            test=False,
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.test_dataloader(),  # datamodule.test_dataloader(),
        )
        test_reference_metrics = test_sampling_metrics.domain_metrics.forward(
            training_graphs, current_epoch=None, local_rank=0
        )
        print("Saving reference metrics.")
        # print(f"deg: {test_reference_metrics['degree']} | clus: {test_reference_metrics['clustering']} | orbit: {test_reference_metrics['orbit']}")
        # breakpoint()
        save_pickle((val_reference_metrics, test_reference_metrics), ref_metrics_path)

    print("Loading reference metrics.")
    (
        dataset_infos.val_reference_metrics,
        dataset_infos.test_reference_metrics,
    ) = load_pickle(ref_metrics_path)
    # print(dataset_infos.test_reference_metrics)
    # breakpoint()
