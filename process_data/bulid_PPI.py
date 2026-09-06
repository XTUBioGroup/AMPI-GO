import os
import torch
import pandas as pd
import dgl
from tqdm import tqdm
from collections import defaultdict

# ===== 1. Path configuration =====
ppi_dir = "ppi_from_species_2hop_valid_100_10_supplement"   
embed_dir = [
    "extra_protT5_embeds_train",
    "extra_protT5_embeds_train_supplemem"
]            
save_dir = "ppi_graphs_supplemnt"                        
mapping_file = "pdb_uniprot_string_supplement.txt" 

os.makedirs(save_dir, exist_ok=True)

# ===== 2. Global cache =====
# embed_cache contains both "1MZH-A" (PDB) and "9606.ENSP..." (STRING) keys
embed_cache = {}       
string_to_pdbs = defaultdict(list) 

# ===== 3. Preload embeddings =====
def preload_embeddings():
    print(f"🚀 Starting to load embeddings from three directories...")
    
    total_files = 0
    
    for d in embed_dir:
        if not os.path.exists(d):
            print(f"⚠️ Warning: Path does not exist: {d}; skipping")
            continue
            
        files = [f for f in os.listdir(d) if f.endswith(".pt")]
        print(f"📂 Loading {d} ({len(files)} files)...")
        
        for f in tqdm(files, desc="Loading", leave=False):
            node_id = os.path.splitext(f)[0] # Case-sensitive (WSL environment)
            
            # Avoid duplicate loads (for identical names across directories, retain the first loaded file)
            if node_id in embed_cache:
                continue
                
            try:
                path = os.path.join(d, f)
                emb = torch.load(path, map_location="cpu")
                if emb.ndim == 2: emb = emb.squeeze(0)
                embed_cache[node_id] = emb
            except: pass
            
        total_files += len(files)

    print(f"✅ All embeddings loaded! Memory contains {len(embed_cache)} unique vectors.")

# ===== 4. Load the mapping table =====
def load_mapping():
    print(f"⏳ [2/4] Loading mapping table {mapping_file} ...")
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                # 1. Length check (guard against empty rows or missing columns)
                if len(parts) < 3: 
                    continue
                
                pdb_id = parts[0]   # 1MZH-A
                # string_id is the third column
                string_id = parts[2] 

                # 2. Filter "NA" strings and other invalid values
                # Some files may use "N/A", "-", "null", etc.
                if string_id in ["NA", "N/A", "-", "nan", "None"]:
                    continue
                
                # 3. Store in the dictionary
                string_to_pdbs[string_id].append(pdb_id)
                
    except Exception as e:
        print(f"❌ Failed to read the mapping table: {e}")

# ===== 5. Core ID-conversion logic (with fallback) =====
def get_best_node_id(raw_id, current_center_pdb):
    """
    raw_id: ID in the PPI file (usually a STRING ID)
    current_center_pdb: PDB ID corresponding to the current file
    """
    
    # If raw_id is already cached (for example, a PDB ID or a STRING ID requiring no conversion),
    # do not return immediately; first check whether it maps to current_center_pdb (highest priority).
    # PPI files usually contain only STRING IDs, so proceed directly to mapping logic.
    
    if raw_id in string_to_pdbs:
        candidates = string_to_pdbs[raw_id] # Get the corresponding PDB list
        
        # 👑 Priority 1: The mapping list contains the center protein (current filename)
        if current_center_pdb in candidates:
            return current_center_pdb
        
        # 🥈 Priority 2: Map to another PDB with an embedding
        for pdb in candidates:
            if pdb in embed_cache:
                return pdb
    
    # 🥉 Priority 3 (fallback):
    # - The mapping table does not contain this STRING ID
    # - Or none of the mapped PDBs has an embedding
    # -> Return the original STRING ID directly
    #    (feature construction later looks up this STRING ID in embed_cache)
    return raw_id

# ===== 6. Graph-construction logic =====
def build_ppi_graph(ppi_file):
    filename = os.path.basename(ppi_file)
    center_pdb_id = os.path.splitext(filename)[0] 

    # 1. Read the file
    try:
        sep = "\t" if not ppi_file.endswith(".csv") else ","
        df = pd.read_csv(ppi_file, sep=sep)
    except Exception as e:
        print(f"❌ [Read failed] {filename}: {e}")
        return None # Skip corrupted files
    
    # 2. Rename columns (adapt to STRING format)
    rename_map = {
        "protein1": "PDB_A",
        "protein2": "PDB_B",
        "score": "score",
        "combined_score": "score" # And so on
    }
    df = df.rename(columns=rename_map)

    # 3. Check whether required columns exist
    if not {"PDB_A", "PDB_B", "score"}.issubset(df.columns):
        # Print an error and skip only when critical columns are missing
        # print(f"❌ [Invalid columns] {filename} contains only: {list(df.columns)}")
        return None # Skip incorrectly formatted files
    
    # 4. Clean invalid data (force numeric conversion and remove empty rows)
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=["PDB_A", "PDB_B", "score"])

    # 5. Check whether the data is empty
    if df.empty: 
        # Returning None here works with `if g` in the main program to skip the file
        # print(f"⚠️ [Empty data] {filename} (skipped)")
        return None 
    df["score"] = df["score"].clip(lower=0, upper=1000.0) / float(1000.0)
    # Get all involved raw IDs
    raw_nodes = set(df["PDB_A"]).union(set(df["PDB_B"]))
    
    # --- ID conversion ---
    id_map = {} # Old -> New
    final_node_list = [center_pdb_id] # Force index 0 to be the center PDB
    seen_final_ids = {center_pdb_id}

    for raw in raw_nodes:
        # Apply the three-level priority logic here
        best_id = get_best_node_id(raw, center_pdb_id)
        id_map[raw] = best_id
        
        if best_id not in seen_final_ids:
            final_node_list.append(best_id)
            seen_final_ids.add(best_id)

    node_to_idx = {n: i for i, n in enumerate(final_node_list)}
    num_nodes = len(final_node_list)

    # 6. Build the edge list manually (core change: create all edges at once)
    src_list = []
    dst_list = []
    weight_list = []

    # (A) Add bidirectional PPI edges
    for _, row in df.iterrows():
        # Get converted IDs
        u_final = id_map.get(row["PDB_A"])
        v_final = id_map.get(row["PDB_B"])
        
        if u_final in node_to_idx and v_final in node_to_idx:
            u = node_to_idx[u_final]
            v = node_to_idx[v_final]
            w = float(row["score"])

            # Forward u->v
            src_list.append(u)
            dst_list.append(v)
            weight_list.append(w)

            # Reverse v->u (avoid duplicate self-loops)
            if u != v:
                src_list.append(v)
                dst_list.append(u)
                weight_list.append(w)

    # (B) Add self-loops (weight = 1.0)
    for i in range(num_nodes):
        src_list.append(i)
        dst_list.append(i)
        weight_list.append(1.0) # Self-loop weight

    # 7. Build the DGL graph
    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    w   = torch.tensor(weight_list, dtype=torch.float32)

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    
    # ✅ Assignment is safe here because the graph structure is fixed
    g.edata["weight"] = w 

    # --- Mark the center ---
    mask = torch.zeros(len(final_node_list), dtype=torch.bool)
    mask[0] = True
    g.ndata["is_center"] = mask

    # --- Populate features ---
    feats = []
    missing = 0
    for node_id in final_node_list:
        # This is where the fallback logic takes effect:
        # If best_id above returns a PDB ID, look up the PDB embedding
        # If best_id above returns a STRING ID, look up the STRING embedding
        if node_id in embed_cache:
            feats.append(embed_cache[node_id])
        else:
            missing += 1
            feats.append(torch.zeros(1024)) # Use zeros if no embedding is available

    g.ndata["x"] = torch.stack(feats)
    
    return g

# ===== 7. Main program =====
if __name__ == "__main__":
    preload_embeddings()
    load_mapping()

    files = [f for f in os.listdir(ppi_dir) if f.endswith((".txt", ".tsv"))]
    print(f"🚀 [3/4] Starting construction of {len(files)} graphs...")

    success = 0
    for f in tqdm(files):
        g = build_ppi_graph(os.path.join(ppi_dir, f))
        if g:
            out = os.path.join(save_dir, os.path.splitext(f)[0] + ".bin")
            try:
                dgl.save_graphs(out, [g])
                success += 1
            except: pass

    print(f"🎉 [4/4] All done! Successful: {success}")




# import os
# import torch
# import pandas as pd
# import dgl
# from tqdm import tqdm
# from collections import defaultdict
#
# # ===== 1. Path configuration =====
# ppi_dir = "ppi_from_species_2hop_100_10"
# embed_dir = [
#     "extra_protT5_embeds_train"
# ]
# save_dir = "ppi_graphs_big"
# mapping_file = "final_mapping_full_scan.tsv"
#
# # [New] Load embeddings only for these IDs
# target_id_file = "big_extracted_sequences_from_folder.txt"
#
# os.makedirs(save_dir, exist_ok=True)
#
# # ===== 2. Global cache =====
# embed_cache = {}
# string_to_pdbs = defaultdict(list)
#
# # ===== [New] Load the target ID list =====
# def load_target_ids():
#     print(f"📋 Reading target ID list: {target_id_file} ...")
#     valid_ids = set()
#     if not os.path.exists(target_id_file):
#         print(f"❌ Error: {target_id_file} not found; no embeddings can be loaded!")
#         return valid_ids
#
#     with open(target_id_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             # Assume the first column is the ID (support tabs or spaces)
#             parts = line.split()
#             if parts:
#                 valid_ids.add(parts[0])
#
#     print(f"✅ Target IDs loaded: {len(valid_ids)} unique IDs require loading.")
#     return valid_ids
#
# # ===== 3. [Modified] Load embeddings on demand =====
# def preload_embeddings(valid_ids):
#     print(f"🚀 Loading embeddings based on the ID list...")
#
#     if not valid_ids:
#         print("⚠️ Warning: The target ID list is empty; no embeddings will be loaded!")
#         return
#
#     total_loaded = 0
#
#     for d in embed_dir:
#         if not os.path.exists(d):
#             print(f"⚠️ Warning: Path does not exist: {d}; skipping")
#             continue
#
#         # Get all .pt files in the directory
#         files = [f for f in os.listdir(d) if f.endswith(".pt")]
#         print(f"📂 Scanning directory {d} ({len(files)} files)...")
#
#         # Display progress with tqdm
#         for f in tqdm(files, desc="Filtering & Loading", leave=False):
#             node_id = os.path.splitext(f)[0] # Remove the .pt suffix
#
#             # --- Core change: Filtering logic ---
#             # Load the file only when its filename is in valid_ids
#             if node_id not in valid_ids:
#                 continue
#             # -----------------------
#
#             # Avoid duplicate loads
#             if node_id in embed_cache:
#                 continue
#
#             try:
#                 path = os.path.join(d, f)
#                 emb = torch.load(path, map_location="cpu")
#                 if emb.ndim == 2: emb = emb.squeeze(0)
#                 embed_cache[node_id] = emb
#                 total_loaded += 1
#             except: pass
#
#     print(f"✅ Embedding loading complete! Loaded {len(embed_cache)} vectors into memory (hit rate: {len(embed_cache)}/{len(valid_ids)}).")
#
# # ===== 4. Load the mapping table (unchanged) =====
# def load_mapping():
#     print(f"⏳ [2/4] Loading mapping table {mapping_file} ...")
#     try:
#         with open(mapping_file, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) < 3: continue
#                 pdb_id = parts[0]
#                 string_id = parts[2]
#                 if string_id in ["NA", "N/A", "-", "nan", "None"]: continue
#                 string_to_pdbs[string_id].append(pdb_id)
#     except Exception as e:
#         print(f"❌ Failed to read the mapping table: {e}")
#
# # ===== 5. Core ID-conversion logic (unchanged) =====
# def get_best_node_id(raw_id, current_center_pdb):
#     if raw_id in string_to_pdbs:
#         candidates = string_to_pdbs[raw_id]
#         if current_center_pdb in candidates:
#             return current_center_pdb
#         for pdb in candidates:
#             if pdb in embed_cache:
#                 return pdb
#     return raw_id
#
# # ===== 6. Graph-construction logic (unchanged) =====
# def build_ppi_graph(ppi_file):
#     filename = os.path.basename(ppi_file)
#     center_pdb_id = os.path.splitext(filename)[0]
#
#     try:
#         sep = "\t" if not ppi_file.endswith(".csv") else ","
#         df = pd.read_csv(ppi_file, sep=sep)
#     except Exception as e:
#         print(f"❌ [Read failed] {filename}: {e}")
#         return None
#
#     rename_map = {"protein1": "PDB_A", "protein2": "PDB_B", "score": "score", "combined_score": "score"}
#     df = df.rename(columns=rename_map)
#
#     if not {"PDB_A", "PDB_B", "score"}.issubset(df.columns):
#         return None
#
#     df['score'] = pd.to_numeric(df['score'], errors='coerce')
#     df = df.dropna(subset=["PDB_A", "PDB_B", "score"])
#     if df.empty: return None
#
#     df["score"] = df["score"].clip(lower=0, upper=1000.0) / float(1000.0)
#     raw_nodes = set(df["PDB_A"]).union(set(df["PDB_B"]))
#
#     id_map = {}
#     final_node_list = [center_pdb_id]
#     seen_final_ids = {center_pdb_id}
#
#     for raw in raw_nodes:
#         best_id = get_best_node_id(raw, center_pdb_id)
#         id_map[raw] = best_id
#         if best_id not in seen_final_ids:
#             final_node_list.append(best_id)
#             seen_final_ids.add(best_id)
#
#     node_to_idx = {n: i for i, n in enumerate(final_node_list)}
#     num_nodes = len(final_node_list)
#
#     src_list = []
#     dst_list = []
#     weight_list = []
#
#     for _, row in df.iterrows():
#         u_final = id_map.get(row["PDB_A"])
#         v_final = id_map.get(row["PDB_B"])
#         if u_final in node_to_idx and v_final in node_to_idx:
#             u = node_to_idx[u_final]
#             v = node_to_idx[v_final]
#             w = float(row["score"])
#             src_list.append(u); dst_list.append(v); weight_list.append(w)
#             if u != v:
#                 src_list.append(v); dst_list.append(u); weight_list.append(w)
#
#     for i in range(num_nodes):
#         src_list.append(i); dst_list.append(i); weight_list.append(1.0)
#
#     src = torch.tensor(src_list, dtype=torch.long)
#     dst = torch.tensor(dst_list, dtype=torch.long)
#     w   = torch.tensor(weight_list, dtype=torch.float32)
#
#     g = dgl.graph((src, dst), num_nodes=num_nodes)
#     g.edata["weight"] = w
#     mask = torch.zeros(len(final_node_list), dtype=torch.bool)
#     mask[0] = True
#     g.ndata["is_center"] = mask
#
#     feats = []
#     missing = 0
#     for node_id in final_node_list:
#         if node_id in embed_cache:
#             feats.append(embed_cache[node_id])
#         else:
#             missing += 1
#             feats.append(torch.zeros(1024))
#
#     g.ndata["x"] = torch.stack(feats)
#     return g
#
# # ===== 7. Main program =====
# if __name__ == "__main__":
#     # 1. Load the ID list first
#     target_ids = load_target_ids()
#
#     # 2. Pass the ID list to the loading function
#     preload_embeddings(target_ids)
#
#     # 3. Continue with the remaining workflow
#     load_mapping()
#
#     files = [f for f in os.listdir(ppi_dir) if f.endswith((".txt", ".tsv"))]
#     print(f"🚀 [3/4] Starting construction of {len(files)} graphs...")
#
#     success = 0
#     for f in tqdm(files):
#         g = build_ppi_graph(os.path.join(ppi_dir, f))
#         if g:
#             out = os.path.join(save_dir, os.path.splitext(f)[0] + ".bin")
#             try:
#                 dgl.save_graphs(out, [g])
#                 success += 1
#             except: pass
#
#     print(f"🎉 [4/4] All done! Successful: {success}")
