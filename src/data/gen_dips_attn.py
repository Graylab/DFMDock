import esm
import os
import torch
import os.path as path
from collections import defaultdict
from tqdm import tqdm
from src.utils.use_dill import get_data
from Bio.Data.IUPACData import protein_letters_3to1


def get_esm_attn(seq_prim, batch_converter, esm_model, device):
    # Use ESM-1b format.
    # The length of tokens is:
    # L (sequence length) + 2 (start and end tokens)
    seq = [
        ("seq", seq_prim)
    ]
    out = batch_converter(seq)
    with torch.no_grad():
        results = esm_model(out[-1].to(device), repr_layers=[33], return_contacts=True)
        attn = results["attentions"].squeeze(0)[:, :, 1:-1, 1:-1]
        output = attn.permute(2, 3, 0, 1).flatten(2, 3)

    return output
    

if __name__ == '__main__':
    data_dir = "/home/lchu11/scr4_jgray21/lchu11/data/dips/pairs_pruned"
    data_list = "/home/lchu11/scr4_jgray21/lchu11/data/dips/pairs_pruned/pairs-postprocessed.txt"
    save_dir = "/home/lchu11/scr4_jgray21/lchu11/data/dips/pt_attn"

    os.makedirs(save_dir, exist_ok=True)

    with open(data_list, 'r') as f:
        lines = f.readlines()
    file_list = [line.strip() for line in lines] 

    # Load esm
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet('/home/lchu11/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = esm_model.to(device).eval()

    # save 
    for _id in tqdm(file_list):
        pdb_file = path.join(data_dir, _id)
        split_string = _id.split('/')
        _id = split_string[0] + '_' + split_string[1].rsplit('.', 1)[0]

        # Get data from files  
        data = get_data(pdb_file)

        # Convert res from 3 to 1 
        aa_code = defaultdict(lambda: "<unk>")
        aa_code.update(
            {k.upper():v for k,v in protein_letters_3to1.items()})
        
        seq1 = "".join(aa_code[s] for s in data['receptor']['res'])
        seq2 = "".join(aa_code[s] for s in data['ligand']['res'])
        seq = seq1 + seq2


        # ESM embedding
        attn = get_esm_attn(seq, batch_converter, esm_model, device).half()

        torch.save(attn, path.join(save_dir, _id+'.pt'))
        break
