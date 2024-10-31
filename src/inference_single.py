import argparse
from inference import inference

def parse_args():
    parser = argparse.ArgumentParser(description="Process two required PDB files.")
    parser.add_argument("pdb_1", type=str, help="Path to the first PDB file")
    parser.add_argument("pdb_2", type=str, help="Path to the second PDB file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference(args.pdb_1, args.pdb_2)
