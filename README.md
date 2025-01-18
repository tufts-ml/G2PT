# G2PT: Graph Generative Pre-trained Transformer Framework

G2PT is an auto-regressive transformer model designed to learn graph structures through next-token prediction.

📑 **paper:** [https://www.arxiv.org/abs/2501.01073](https://www.arxiv.org/abs/2501.01073)       

🤗 **checkpoints**: [G2PT Collection](https://huggingface.co/collections/xchen16/g2pt-677f692eab4f83d4aa4231aa)

![](./assets/g2pt.gif)

## Quick Start with 🤗 HuggingFace

### Loading Pre-trained Models

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("xchen16/g2pt-guacamol-small-deg")
model = AutoModelForCausalLM.from_pretrained("xchen16/g2pt-guacamol-small-deg")
```

### Generating Graphs using Pre-trained Models
See [example_sample_hf.py](example_sample_hf.py)

```python
# Generate sequences
inputs = tokenizer(['<boc>'], return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_length=tokenizer.model_max_length,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=1.0
)
sequences = tokenizer.batch_decode(outputs)

# Converting sequences to Smiles/RDKit Molecules/nx graphs
...
```

### Available Pre-trained Models
<table>
  <!-- Header row: Column A spans 2 rows, Columns B & C are grouped under a single heading -->
  <tr>
    <th rowspan="2"> </th>
    <th> </th>
    <th colspan="3" style="text-align: center">Datasets</th>
  </tr>
  <!-- Second row of the header for B and C -->
  <tr>
    <th> </th>
    <th style="text-align: center">QM9</th>
    <th style="text-align: center">Moses</th>
    <th style="text-align: center">GuacaMol</th>
  </tr>

  <!-- First data row: the left cell has an extra rowspan -->
  <tr>
    <td rowspan="2">Small</td>
    <td>BFS</td>
    <th style="text-align: center"> ✅ </td>
    <th style="text-align: center"> ✅ </td>
    <th style="text-align: center"> ✅ </td>
  </tr>
  <!-- Second data row under the first row’s span -->
  <tr>
    <td>DEG</td>
    <th style="text-align: center"> ✅ </td>
    <th style="text-align: center"> ✅ </td>
    <th style="text-align: center"> ✅ </td>
  </tr>
  <tr>
    <td rowspan="2">Base</td>
    <td>BFS</td>
    <th> </th>
    <th style="text-align: center"> ✅ </td>
    <th style="text-align: center"> ✅ </td>
  </tr>
  <tr>
    <td>DEG</td>
    <th> </th>
    <th style="text-align: center"> ✅ </td>
    <th style="text-align: center"> ✅ </td>
  </tr>
  <tr>
    <td rowspan="2">Large</td>
    <td>BFS</td>
    <th> </th>
    <th style="text-align: center"> ✅ </td>
    <th style="text-align: center"> ✅ </td>
  </tr>
  <tr>
    <td>DEG</td>
    <th> </th>
    <th style="text-align: center"> ✅ </td>
    <th style="text-align: center"> ✅ </td>
  </tr>
</table>

More coming soon...

## Training Your Own Model

### Prerequisites and Installation

1. First, get the code:
   ```bash
   git clone https://github.com/tufts-ml/g2pt_hf.git
   cd g2pt_hf
   ```

2. Set up your Python environment:
   ```bash
   conda create -n g2pt python==3.10
   conda activate g2pt
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

For dataset preparation instructions, please refer to [datasets/README.md](datasets/README.md). For using custom data, make sure to provide the corresponding tokenizer configurations, see [tokenizers](tokenizers).


### Model Training

Launch training with the provided script:
```bash
sh scripts/pretrain.sh
```

Default training configuration:
- To distributed train across N GPUs, set --nproc_per_node=N
- Modify [configs/datasets](configs/datasets) and [configs/networks](configs/networks) for your tasks. Training arguments are in [configs/default.py](configs/default.py)

### Sampling

Generate new graphs using:
```bash
sh scripts/sample.sh
```

## Citation

If you use G2PT in your research, please cite our paper:

```bibtex
@article{chen2025graph,
  title={Graph Generative Pre-trained Transformer},
  author={Chen, Xiaohui and Wang, Yinkai and He, Jiaxing and Du, Yuanqi and Hassoun, Soha and Xu, Xiaolin and Liu, Li-Ping},
  journal={arXiv preprint arXiv:2501.01073},
  year={2025}
}
```
