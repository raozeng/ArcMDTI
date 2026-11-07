# -*- coding:utf-8 -*-
import os
import pickle
import argparse


def convert_fasta_to_pkl(fasta_path, output_name):
    """Convert a NetSurfP-3.0 secondary structure FASTA file into a {Protein_ID: SS_String} pickle."""
    if not os.path.exists(fasta_path):
        print(f"Error: File not found: {fasta_path}")
        return

    ss_dict = {}
    with open(fasta_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_id = None
    count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current_id = line[1:].split()[0]
        elif current_id:
            if current_id in ss_dict:
                ss_dict[current_id] += line
            else:
                ss_dict[current_id] = line
                count += 1

    with open(output_name, "wb") as f:
        pickle.dump(ss_dict, f)
    print(f"Converted {count} secondary structure sequences -> {output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SS FASTA to Pickle")
    parser.add_argument("--input", type=str, default="./ion_channel.fasta", help="Input FASTA path")
    parser.add_argument("--output", type=str, default="DrugBank_ss.pkl", help="Output PKL path")
    args = parser.parse_args()
    convert_fasta_to_pkl(args.input, args.output)

