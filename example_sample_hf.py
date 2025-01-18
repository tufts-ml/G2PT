"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets_utils import seq_to_mol, get_smiles, seq_to_molecule_with_partial_charges
import argparse
from contextlib import nullcontext


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

def generate_sequences(model, tokenizer, batch_size, num_samples, device):
    inputs = tokenizer(['<boc>']*batch_size, return_tensors="pt")
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
            temperature=1.0
        )
        seq_strs = tokenizer.batch_decode(ids)
        generated_sequences.extend(seq_strs)
        
    return generated_sequences[:num_samples]

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from a trained model')
    parser.add_argument('--model_name_or_path', type=str, default='xchen16/g2pt-moses-small-bfs',
                        help='Directory containing model checkpoint')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device, ctx = setup_device(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()
    
    with ctx:
        generated_sequences = generate_sequences(
            model, 
            tokenizer,
            args.batch_size,
            args.num_samples,
            device
        )
    
    smiles = []
    for seq_str in generated_sequences:
        try:
            if 'guacamol' in args.model_name_or_path:
                mol = seq_to_molecule_with_partial_charges(seq_str)
            else:
                mol = seq_to_mol(seq_str)
            smile = get_smiles(mol)
            if smile:
                smiles.append(smile)
            else:
                smiles.append(None)
        except:
            # handling sequence invalid error
            continue
    smiles = [str(s) for s in smiles]
    open(f'generated_smiles.txt', 'w').write('\n'.join(smiles))
