import argparse
from inference_base import inference_multiple_poses

def parse_args():
    parser = argparse.ArgumentParser(description="Process two required PDB files.")
    parser.add_argument("pdb_1", type=str, help="Path to the first PDB file")
    parser.add_argument("pdb_2", type=str, help="Path to the second PDB file")
    parser.add_argument("--num_samples", type=int, default=40, help="Number of output poses/samples, default=40")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference_multiple_poses(args.pdb_1, args.pdb_2, args.num_samples)
