# import json
# import os

# # ================= Configuration =================
# INPUT_JSON = "pdb2go_final_clean.json"  # Prefer the latest updated and propagated version
# OUTPUT_DIR = "dataset_splits"       # Output directory
# # =======================================

# def main():
#     if not os.path.exists(INPUT_JSON):
#         print(f"❌ Error: {INPUT_JSON} not found")
#         return

#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)

#     print(f"📖 Reading {INPUT_JSON} ...")
#     with open(INPUT_JSON, 'r') as f:
#         data = json.load(f)

#     print(f"🔹 Total original IDs: {len(data)}")

#     # Initialize three lists
#     mf_ids = []
#     bp_ids = []
#     cc_ids = []

#     # Iterate over and filter the data
#     for pid, annotations in data.items():
#         # 1. Check Molecular Function
#         # Support both formats: list of strings ["GO:1,GO:2"] or list of GOs ["GO:1", "GO:2"]
#         mf_list = annotations.get("molecular_function", [])
#         if mf_list and len(mf_list) > 0:
#             # Confirm that the entries are not empty strings
#             if isinstance(mf_list[0], str) and len(mf_list[0].strip()) > 0:
#                 mf_ids.append(pid)
        
#         # 2. Check Biological Process
#         bp_list = annotations.get("biological_process", [])
#         if bp_list and len(bp_list) > 0:
#             if isinstance(bp_list[0], str) and len(bp_list[0].strip()) > 0:
#                 bp_ids.append(pid)

#         # 3. Check Cellular Component
#         cc_list = annotations.get("cellular_component", [])
#         if cc_list and len(cc_list) > 0:
#             if isinstance(cc_list[0], str) and len(cc_list[0].strip()) > 0:
#                 cc_ids.append(pid)

#     # Print split statistics
#     print("\n📊 Split statistics:")
#     print(f"   🧬 Molecular Function (MF) samples: {len(mf_ids)}")
#     print(f"   🔄 Biological Process (BP) samples: {len(bp_ids)}")
#     print(f"   🏠 Cellular Component (CC) samples: {len(cc_ids)}")

#     # Save ID lists
#     def save_ids(filename, id_list):
#         path = os.path.join(OUTPUT_DIR, filename)
#         with open(path, 'w') as f:
#             for pid in id_list:
#                 f.write(f"{pid}\n")
#         print(f"✅ Saved: {path}")

#     save_ids("mf_ids_all.txt", mf_ids)
#     save_ids("bp_ids_all.txt", bp_ids)
#     save_ids("cc_ids_all.txt", cc_ids)

#     print("\n🎉 Step one complete! You now have three clean ID lists without empty-label samples.")
#     print("Next, each list will be split into training, validation, and test sets.")

# if __name__ == "__main__":
#     main()


########_____________##################### Split the dataset
import os

# ================= Configuration =================
# Original sequence file (format: ID \t Sequence or FASTA)
SOURCE_SEQ_FILE = "protein_id_and_sequence.txt" 
OUTPUT_DIR = "dataset_splits"
# =======================================

def load_sequences(seq_file):
    """Read the ID -> Sequence mapping."""
    seq_map = {}
    with open(seq_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # Ensure a consistent ID format (for example, uppercase)
                pid = parts[0].strip()
                seq = parts[1].strip()
                seq_map[pid] = seq
    return seq_map

def generate_fasta_for_ontology(ont, seq_map):
    id_file = os.path.join(OUTPUT_DIR, f"{ont}_ids_all.txt")
    fasta_file = os.path.join(OUTPUT_DIR, f"{ont}_all.fasta")
    
    if not os.path.exists(id_file):
        print(f"⚠️ Skipping {ont}: ID list not found.")
        return

    with open(id_file, 'r') as f_in, open(fasta_file, 'w') as f_out:
        count = 0
        for line in f_in:
            pid = line.strip()
            if pid in seq_map:
                # Write in FASTA format
                f_out.write(f">{pid}\n{seq_map[pid]}\n")
                count += 1
            else:
                # Record IDs that may be absent from the sequence file
                pass 
    print(f"✅ Generated {fasta_file}: {count} sequences")

def main():
    print("📖 Loading the original sequence library...")
    seq_map = load_sequences(SOURCE_SEQ_FILE)
    
    for ont in ['mf', 'bp', 'cc']:
        generate_fasta_for_ontology(ont, seq_map)

if __name__ == "__main__":
    main()
