# Dataset Preparation

## Available Datasets
- guacamol
- moses
- planar
- sbm
- tree
- lobster (no preprocessing required)

## Preprocessing Steps

To prepare a dataset, run the corresponding preparation script:
```bash
python prepare_<dataset>.py
```

For example:
```bash
python prepare_guacamol.py    # For GuacaMol dataset
python prepare_moses.py       # For MOSES dataset
python prepare_planar.py      # For planar graphs
```

Each preparation script:
1. Loads raw data
2. Converts graphs to the required format
3. Saves processed data in `./<dataset>/`


## Data Format

Processed data is stored in binary format with three components:
- `xs.bin`: Node features
- `edge_indices.bin`: Edge connectivity
- `edge_attrs.bin`: Edge features

Each dataset includes train and evaluation splits. 