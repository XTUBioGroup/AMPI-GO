# import json
# import os
# from collections import defaultdict
# from tqdm import tqdm

# # ================= Configuration =================
# OLD_JSON_FILE = "process_data/pdb2go.json"           # Provides keys
# TSV_FILE = "F:\download\pdb_chain_go.tsv\pdb_chain_go.tsv"           # Provides values (GO data)
# OBO_FILE = "process_data/go.obo"                     # Provides structure (classification + propagation)
# OUTPUT_FILE = "pdb2go_updated.json"     # Final result
# # =======================================

# def load_obo_structure(obo_path):
#     """
#     Parse the OBO file to obtain namespaces and parent-child relationships.
#     """
#     print(f"📖 Parsing OBO: {obo_path} ...")
#     parents_map = defaultdict(list)
#     ns_map = {}
#     current_id = None
    
#     with open(obo_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith("id: GO:"):
#                 current_id = line.split()[1]
#             elif line.startswith("namespace:") and current_id:
#                 ns_map[current_id] = line.split()[1]
#             elif line.startswith("is_a:") and current_id:
#                 parent = line.split()[1].split('!')[0].strip()
#                 parents_map[current_id].append(parent)
                
#     return parents_map, ns_map

# def get_ancestors(go_id, parents_map, memo):
#     """
#     Recursively retrieve ancestor nodes (the core of propagation).
#     """
#     if go_id in memo: return memo[go_id]
    
#     ancestors = set()
#     for p in parents_map.get(go_id, []):
#         ancestors.add(p)
#         ancestors.update(get_ancestors(p, parents_map, memo))
    
#     memo[go_id] = ancestors
#     return ancestors

# def main():
#     if not os.path.exists(OBO_FILE) or not os.path.exists(OLD_JSON_FILE):
#         print("❌ Required file missing (go.obo or pdb2go.json)")
#         return

#     # 1. Parse the OBO file
#     parents_map, ns_map = load_obo_structure(OBO_FILE)

#     # 2. Read the old JSON only to obtain keys (the allowlist)
#     print(f"📖 Reading keys from the old JSON: {OLD_JSON_FILE} ...")
#     with open(OLD_JSON_FILE, 'r', encoding='utf-8') as f:
#         old_data = json.load(f)
    
#     # Build an allowlist set of keys for fast lookup
#     # Assume keys use a format such as "155C-A"
#     valid_keys = set(old_data.keys())
#     print(f"✅ Selected {len(valid_keys)} target PDB chains.")

#     # 3. Preprocess TSV data in memory
#     # Structure: tsv_data["155C-A"] = {"GO:001", "GO:002", ...} (mixed namespaces)
#     tsv_data = defaultdict(set)
    
#     print(f"📖 Scanning TSV: {TSV_FILE} ...")
#     with open(TSV_FILE, 'r', encoding='utf-8') as f:
#         for line in tqdm(f, desc="Scanning TSV"):
#             if line.startswith("#") or not line.strip(): continue
#             parts = line.strip().split('\t')
#             if len(parts) <= 5: continue
            
#             # Build the key: uppercase PDB + "-" + chain
#             pdb = parts[0].upper()
#             chain = parts[1]
#             key = f"{pdb}-{chain}"
            
#             # 🔥 Key point: Extract only keys present in the old JSON
#             if key in valid_keys:
#                 go_id = parts[5]
#                 if go_id in ns_map: # Process only valid GO terms
#                     tsv_data[key].add(go_id)

#     # 4. Build a new dictionary and propagate
#     print("🚀 Building the new dictionary and propagating parent nodes...")
#     final_dict = {}
#     ancestor_memo = {}
    
#     for key in tqdm(valid_keys, desc="Propagating"):
#         # Initialize an empty structure (retain the key even without TSV data to prevent errors)
#         entry = {
#             "molecular_function": set(),
#             "biological_process": set(),
#             "cellular_component": set()
#         }
        
#         # Get leaf GO terms corresponding to this key in the TSV
#         leaf_gos = tsv_data.get(key, set())
        
#         # Iterate over every leaf GO term and propagate
#         all_gos = set(leaf_gos) # Include the terms themselves
#         for go_id in leaf_gos:
#             ancestors = get_ancestors(go_id, parents_map, ancestor_memo)
#             all_gos.update(ancestors)
            
#         # Classify all propagated GO terms and add them to the entry
#         for go_id in all_gos:
#             ns = ns_map.get(go_id)
#             if ns:
#                 entry[ns].add(go_id)
        
#         # Format for saving: Convert to lists
#         final_dict[key] = {
#             "molecular_function": list(entry["molecular_function"]),
#             "biological_process": list(entry["biological_process"]),
#             "cellular_component": list(entry["cellular_component"])
#         }
        
#         # Uncomment below to support the old string-list format ["GO:1,GO:2"]
#         # for ns_key in final_dict[key]:
#         #     if final_dict[key][ns_key]:
#         #         merged_str = ",".join(final_dict[key][ns_key])
#         #         final_dict[key][ns_key] = [merged_str]

#     # 5. Save
#     print(f"💾 Saving to {OUTPUT_FILE} ...")
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         json.dump(final_dict, f, indent=2)
        
#     print(f"🎉 Complete!")
#     print(f"Retained keys: {len(final_dict)}")
#     print("All data has been replaced with the latest TSV version and now includes every parent node.")

# if __name__ == "__main__":
#     main()


import json
import os

# ================= Path configuration =================
JSON_FILE = "pdb2go_updated.json"        # JSON file to clean
TRAIN_FILE = "protein_id_and_sequence_train.txt" # Training-set ID list
VALID_FILE = "protein_id_and_sequence_valid.txt" # Validation-set ID list
OUTPUT_FILE = "pdb2go_final_clean.json"  # Output file
# ===========================================

def load_ids_from_file(filepath):
    """
    Read an ID-list file.
    Assume one ID per line or an "ID SEQUENCE" format.
    For ID+sequence files, the first column is normally the ID.
    """
    ids = set()
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: File not found: {filepath}; skipping.")
        return ids
        
    print(f"📖 Reading ID list: {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Adjust this section to match the file format
            # For a plain ID list:
            # pid = line
            
            # For "ID sequence" (space- or tab-delimited):
            parts = line.split()
            pid = parts[0]
            
            # Remove a possible ">" prefix (common in FASTA)
            if pid.startswith(">"):
                pid = pid[1:]
                
            ids.add(pid)
    
    print(f"   └─ Found {len(ids)} unique IDs.")
    return ids

def main():
    # 1. Build the allowlist
    train_ids = load_ids_from_file(TRAIN_FILE)
    valid_ids = load_ids_from_file(VALID_FILE)
    
    valid_keys = train_ids.union(valid_ids)
    
    print(f"✅ Allowlist complete. {len(valid_keys)} protein IDs are permitted.")
    
    # 2. Read the JSON file
    if not os.path.exists(JSON_FILE):
        print(f"❌ Error: {JSON_FILE} not found")
        return

    print(f"📖 Reading JSON file: {JSON_FILE} ...")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    initial_count = len(data)
    print(f"   └─ Original data contains {initial_count} entries.")
    
    # 3. Clean the data
    print("🧹 Starting cleanup...")
    cleaned_data = {}
    removed_count = 0
    
    for pid, content in data.items():
        # Check whether the key is in the allowlist
        # Note: A JSON key may be "155C-A" while the file uses "155C_A" or "155C"; ensure formats match
        # Exact matching is assumed here
        if pid in valid_keys:
            cleaned_data[pid] = content
        else:
            removed_count += 1
            
    # 4. Save
    print(f"💾 Saving the cleaned file to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2)
        
    print(f"🎉 Complete!")
    print(f"   Original count: {initial_count}")
    print(f"   Removed count: {removed_count}")
    print(f"   Retained count: {len(cleaned_data)}")
    print(f"   Result file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
