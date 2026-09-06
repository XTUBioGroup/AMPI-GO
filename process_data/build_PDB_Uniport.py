import csv
import pickle



## Build a PDB ID-to-UniProt mapping table


# import csv
# import pickle
#
# # Input file (extracted TSV)
# sifts_file = r"F:\QLDownload\pdb_chain_uniprot.tsv"
#
# # Output mapping table
# out_pkl = "pdb2uniprot.pkl"
# out_txt = "pdb2uniprot.txt"
#
# pdb2uniprot = {}
#
# with open(sifts_file, "r", encoding="utf-8") as f:
#     # Skip comment lines beginning with #
#     for line in f:
#         if line.startswith("#"):
#             continue
#         header = line.strip().split("\t")
#         break  # Stop at the actual header
#
#     reader = csv.DictReader(f, delimiter="\t", fieldnames=header)
#
#     for row in reader:
#         pdb_id = row["PDB"]
#         chain_id = row["CHAIN"]
#         uniprot_ac = row["SP_PRIMARY"]
#
#         pdb_chain = f"{pdb_id.upper()}-{chain_id}"
#         pdb2uniprot[pdb_chain] = uniprot_ac
#
# print(f"[INFO] Mapping count: {len(pdb2uniprot)}")
#
# # Save as pickle
# with open(out_pkl, "wb") as fw:
#     pickle.dump(pdb2uniprot, fw)
#
# # Save as TXT (each line: PDB-CHAIN \t UniProt AC)
# with open(out_txt, "w", encoding="utf-8") as fw:
#     for pdb_chain, uniprot_ac in pdb2uniprot.items():
#         fw.write(f"{pdb_chain}\t{uniprot_ac}\n")
#
# print(f"[INFO] Saved mapping tables to {out_pkl} and {out_txt}")
#
# # Test
#
# # Test
# test_id = "2V3B-A"
# if test_id in pdb2uniprot:
#     print(f"{test_id} → UniProt AC: {pdb2uniprot[test_id]}")
# else:
#     print(f"{test_id} is not in the mapping table")
import requests

# ## Find missing IDs
# # File paths
# # File paths
train_file = "protein_id_and_sequence_train.txt"
valid_file = "protein_id_and_sequence_valid.txt"
mapping_file = "pdb2uniprot.txt"

# Read the existing mapping
pdb2uniprot = {}
with open(mapping_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            pdb2uniprot[parts[0]] = parts[1]

print(f"[INFO] Loaded mappings: {len(pdb2uniprot)}")

# Collect PDB IDs to check
def load_pdb_ids(file_path):
    pdb_ids = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            pdb_id = line.strip().split()[0]  # Extract 4MID-A
            pdb_ids.append(pdb_id)
    return pdb_ids

all_ids = set(load_pdb_ids(train_file) + load_pdb_ids(valid_file))

# Categorize
missing_ids = [pid for pid in all_ids if pid not in pdb2uniprot]
found_ids = [pid for pid in all_ids if pid in pdb2uniprot]

print(f"[INFO] Total IDs: {len(all_ids)}; missing: {len(missing_ids)}; found: {len(found_ids)}")

# Write missing IDs
with open("missing_ids.txt", "w") as fw:
    for pid in missing_ids:
        fw.write(pid + "\n")
print(f"[INFO] Wrote {len(missing_ids)} missing IDs to missing_ids.txt")

# Write found IDs
with open("found_ids.txt", "w") as fw:
    for pid in found_ids:
        fw.write(f"{pid}\t{pdb2uniprot[pid]}\n")
print(f"[INFO] Wrote {len(found_ids)} found IDs to found_ids.txt")

