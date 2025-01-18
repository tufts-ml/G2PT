"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
import numpy as np
from model import GPTConfig, GPT
from transformers import AutoTokenizer
from datasets_utils import seq_to_mol, get_smiles, seq_to_molecule_with_partial_charges, seq_to_nxgraph
import argparse
from contextlib import nullcontext

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from a trained model')
    parser.add_argument('--out_dir', type=str, default='results/moses-small-bfs',
                        help='Directory containing model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizers/moses',
                        help='Path to tokenizer') 
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    return parser.parse_args()

def setup_device(seed):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    return device, ctx

def load_model(out_dir, device):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    hf_model = model.to_hf()
    hf_model.eval()
    hf_model.to(device)
    
    return hf_model

def generate_sequences(model, tokenizer, batch_size, num_samples, device, prefix=None, temperature=1.0):
    if prefix is None:
        inputs = tokenizer(['<boc>']*batch_size, return_tensors="pt")
    else:
        inputs = tokenizer([prefix]*batch_size, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    generated_sequences = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    for _ in range(num_batches):
        ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=tokenizer.model_max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
        )
        seq_strs = tokenizer.batch_decode(ids)
        generated_sequences.extend(seq_strs)
        
    return generated_sequences[:num_samples]

if __name__ == '__main__':
    args = parse_args()
    device, ctx = setup_device(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.out_dir, device)
    
   
    if any(dataset_name in args.tokenizer_path for dataset_name in ['guacamol', 'qm9', 'moses']):
        with ctx:
            generated_sequences = generate_sequences(
                model, 
                tokenizer,
                args.batch_size,
                args.num_samples,
                device,
            )
        # save smiles
        smiles = []
        for seq_str in generated_sequences:
            try:
                if 'guacamol' in args.tokenizer_path:
                    mol = seq_to_molecule_with_partial_charges(seq_str)
                else:
                    mol = seq_to_mol(seq_str) 
                smile = get_smiles(mol)
                if smile:
                    smiles.append(smile)
                else:
                    smiles.append(None)
            except:
                # handling sequence invalid error (we ignore decoding errors as it can be easily fixed by constrained sampling)
                continue
        smiles = [str(s) for s in smiles]
        open(f'{args.out_dir}/generated_smiles.txt', 'w').write('\n'.join(smiles))
        
    elif any(dataset_name in args.tokenizer_path for dataset_name in ['planar', 'sbm', 'tree', 'lobster']):
        if 'planar' in args.tokenizer_path:
            prefix = sum([['NODE', f'IDX_{i}', '<sepc>'] for i in range(64)],[])
            prefix[-1] = '<eoc>'
            prefix = ' '.join(['<boc>'] + prefix)
            temperature = 0.3
        else:
            prefix = None
            temperature = 1.0
        with ctx:
            generated_sequences = generate_sequences(
                model, 
                tokenizer,
                args.batch_size,
                args.num_samples,
                device,
                prefix=prefix,
                temperature=temperature,
            )
        # save nx graph
        import pickle
        nx_graphs = []

        for seq_str in generated_sequences:
            try: 
                graph = seq_to_nxgraph(seq_str)
                nx_graphs.append(graph) 
            except:
                continue
        open(f'{args.out_dir}/generated_graphs.pkl', 'wb').write(pickle.dumps(nx_graphs))

