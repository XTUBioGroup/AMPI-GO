
import os
from tqdm import tqdm

# Input files
found_ids_file = "found_ids.txt"  # PDB Chain -> UniProt AC
aliases_file = r"F:\QLDownload\protein.aliases.v12.0.txt"  # STRING alias file
output_file = "pdb_uniprot_string.txt"
missing_file = "missing_uniprot.txt"

# Count file lines first for the progress bar
with open(aliases_file, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

# Read the UniProt AC -> STRING ID mapping
uniprot2string = {}
with open(aliases_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="Loading UniProt→STRING"):
        if line.startswith("#"):  # Skip comment lines
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            string_id, alias, source = parts[0], parts[1], parts[2]
            if source == "UniProt_AC":  # Use only UniProt_AC for mapping
                uniprot2string[alias] = string_id

print(f"[INFO] Loaded UniProt-to-STRING mappings: {len(uniprot2string)}")

# Read found_ids and generate three columns
missing_count = 0
with open(found_ids_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout, \
     open(missing_file, "w", encoding="utf-8") as fmiss:

    for line in tqdm(fin, desc="Mapping PDB→UniProt→STRING"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        pdb_chain, uniprot_ac = parts[0], parts[1]
        string_id = uniprot2string.get(uniprot_ac, "NA")
        fout.write(f"{pdb_chain}\t{uniprot_ac}\t{string_id}\n")
        if string_id == "NA":
            missing_count += 1
            fmiss.write(f"{pdb_chain}\t{uniprot_ac}\n")

print(f"[INFO] Generated {output_file}")
print(f"[INFO] UniProt entries without a STRING ID: {missing_count}")
print(f"[INFO] Missing UniProt entries written to {missing_file}")
