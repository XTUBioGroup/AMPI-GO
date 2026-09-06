# import os
# import time
# import requests
# import pandas as pd
# from io import StringIO
# from tqdm import tqdm

# # ================== Replace with your actual paths ==================

# # Mapping table: Column 1 is the PDB chain ID; column 3 is the STRING ID
# MAPPING_FILE = r"pdb_uniprot_string_valid.txt"

# # Output directory: One file per PDB chain, with names such as 5CED-A.txt
# OUTPUT_DIR = r"F:\PythonProject1\ppi_2hop_per_pdb"

# # =======================================================

# API_BASE = "https://string-db.org/api/tsv/interaction_partners"
# HEADERS = {"User-Agent": "Mozilla/5.0"}

# # Sleep briefly between requests to avoid overloading STRING
# SLEEP_SECONDS = 0.01  # Reduce if needed, but setting it to 0 is not recommended


# def load_pdb_string_mapping(mapping_file):
#     """
#     Read pdb_uniprot_string_valid.txt.
#     Expected format: PDBCHAIN  UniProt  STRINGID
#     Example:
#       5CED-A  Q6MHT0  264462.Bd3459
#     Returns: list[(pdb_chain_id, string_id)]
#     """
#     mapping = []
#     with open(mapping_file, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
#             parts = line.split("\t")
#             if len(parts) < 3:
#                 continue
#             pdb_chain_id = parts[0]
#             string_id = parts[2]
#             mapping.append((pdb_chain_id, string_id))
#     print(f"Read {len(mapping)} PDB-STRING mappings from the mapping table")
#     return mapping


# def fetch_interaction_partners(string_id):
#     """
#     Use /api/tsv/interaction_partners to retrieve all partners for a STRING ID.
#     Returns list[(a, b, score)], where a=stringId_A and b=stringId_B.
#     """
#     # Parse the species ID from the STRING ID, for example 9606.ENSPXXX -> 9606
#     species = None
#     if "." in string_id:
#         prefix = string_id.split(".")[0]
#         if prefix.isdigit():
#             species = int(prefix)

#     params = {
#         "identifiers": string_id,
#         # Add filters if needed, for example:
#         # "required_score": 400,
#         # "limit": 0,  # In some versions, 0 means unlimited; see the STRING documentation
#     }
#     if species is not None:
#         params["species"] = species

#     try:
#         resp = requests.get(API_BASE, params=params, headers=HEADERS, timeout=60)
#         resp.raise_for_status()
#     except Exception as e:
#         print(f"[ERROR] interaction_partners request failed: STRING={string_id}, error={e}")
#         return []

#     text = resp.text.strip()
#     if not text:
#         return []

#     try:
#         df = pd.read_csv(StringIO(text), sep="\t")
#     except Exception as e:
#         print(f"[ERROR] Failed to parse TSV for STRING={string_id}, error={e}")
#         return []

#     required_cols = ["stringId_A", "stringId_B", "score"]
#     if not all(col in df.columns for col in required_cols):
#         print(f"[WARN] Columns returned for STRING={string_id} do not include {required_cols}")
#         return []

#     edges = []
#     for _, row in df.iterrows():
#         a = str(row["stringId_A"])
#         b = str(row["stringId_B"])
#         s = float(row["score"])
#         edges.append((a, b, s))

#     return edges


# def build_2hop_for_one_pdb(pdb_chain_id, string_id):
#     """
#     Build a 2-hop local PPI network centered on string_id for one PDB chain:
#       1. First pass: center -> 1-hop neighbors
#       2. Second pass: Expand all 1-hop neighbors by one more hop
#     Returns: DataFrame(columns=['protein1','protein2','score'])
#     """
#     all_edges = set()   # Store undirected edges (u, v, score)
#     neighbors = set()   # 1-hop neighbors from the first pass

#     # ---------- First pass: 1-hop neighbors of the center ----------
#     edges1 = fetch_interaction_partners(string_id)
#     time.sleep(SLEEP_SECONDS)

#     for a, b, s in edges1:
#         # Deduplicate undirected edges by sorting before storing
#         u, v = sorted([a, b])
#         all_edges.add((u, v, s))

#         # A 1-hop neighbor is the other node connected to string_id
#         if a == string_id:
#             neighbors.add(b)
#         elif b == string_id:
#             neighbors.add(a)
#         else:
#             # The API normally returns the input ID as stringId_A, but count both sides as neighbors for safety
#             neighbors.add(a)
#             neighbors.add(b)

#     # ---------- Second pass: Expand all 1-hop neighbors ----------
#     for nei in neighbors:
#         edges2 = fetch_interaction_partners(nei)
#         time.sleep(SLEEP_SECONDS)
#         for a, b, s in edges2:
#             u, v = sorted([a, b])
#             all_edges.add((u, v, s))

#     # Convert to a DataFrame with the usual columns: protein1 protein2 score
#     if not all_edges:
#         return pd.DataFrame(columns=["protein1", "protein2", "score"])

#     records = [{"protein1": u, "protein2": v, "score": s} for (u, v, s) in all_edges]
#     df = pd.DataFrame(records)
#     return df


# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     mapping = load_pdb_string_mapping(MAPPING_FILE)

#     for pdb_chain_id, string_id in tqdm(mapping, desc="Building 2-hop PPI for each PDB"):
#         out_path = os.path.join(OUTPUT_DIR, f"{pdb_chain_id}.txt")

#         # Add a check here to skip existing results if desired:
#         if os.path.exists(out_path):
#          continue

#         print(f"\n[INFO] Processing PDB={pdb_chain_id}, STRING={string_id}")
#         df = build_2hop_for_one_pdb(pdb_chain_id, string_id)

#         # Save as tab-delimited data
#         df.to_csv(out_path, sep="\t", index=False)
#         print(f"[INFO] Saved 2-hop PPI for {pdb_chain_id} to {out_path}; edges={len(df)}")


# if __name__ == "__main__":
#     main()

# import os
# import requests
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ====================== Configuration ======================

# # Input: Previously generated missing-record file (format: PDB_CHAIN  STRING_ID  TAXID)
# SPECIES_FILE = "missing_species_files.txt"   

# OUTPUT_DIR = "string_species_links_v12"
# BASE_URL = "https://stringdb-downloads.org/download/protein.links.v12.0"

# MAX_WORKERS = 6    # Number of workers
# TIMEOUT = 120       # Per-file timeout in seconds
# MAX_RETRIES = 3     # Number of retries after failure

# FAILED_LOG = "failed_taxids_retry.txt"    # Failure log

# # ===================================================


# def load_taxids_from_third_column(species_file):
#     """
#     Read the third column (TAXID) from missing_species_files.txt,
#     deduplicate it, and retain only unique taxids for download.
#     """
#     taxids = set()
#     try:
#         with open(species_file, "r") as f:
#             header = next(f, None) # Skip the header row (PDB_CHAIN STRING_ID TAXID)
            
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 parts = line.split("\t")
#                 if len(parts) >= 3:
#                     # Get the third column and remove a possible .0 suffix (for example, 9606.0 -> 9606)
#                     taxid = parts[2].split(".")[0] 
#                     taxids.add(taxid)
                    
#         print(f"✅ After reading and deduplication, {len(taxids)} species files must be downloaded")
#         return list(taxids)
        
#     except FileNotFoundError:
#         print(f"❌ File not found: {species_file}")
#         return []


# def download_one(taxid):
#     url = f"{BASE_URL}/{taxid}.protein.links.v12.0.txt.gz"
#     out_path = os.path.join(OUTPUT_DIR, f"{taxid}.protein.links.v12.0.txt.gz")
#     temp_path = out_path + ".tmp"

#     # ✅ 1. Check existing files
#     if os.path.exists(out_path):
#         # Simple check: Treat files larger than 1 KB as valid and skip them (STRING files are usually large)
#         if os.path.getsize(out_path) > 1024:
#             return ("SKIP", taxid, "file exists")
#         else:
#             # Empty or tiny files may be remnants of a failed download; delete and download again
#             try:
#                 os.remove(out_path)
#             except:
#                 pass

#     # ✅ 2. Attempt download with retries
#     for attempt in range(MAX_RETRIES):
#         try:
#             with requests.get(url, stream=True, timeout=TIMEOUT) as r:
#                 if r.status_code == 404:
#                     return ("FAIL", taxid, "HTTP 404 Not Found (TaxID may be invalid)")
#                 if r.status_code != 200:
#                     raise Exception(f"HTTP {r.status_code}")

#                 # Write to a temporary file to prevent corruption on interruption
#                 with open(temp_path, "wb") as f:
#                     for chunk in r.iter_content(chunk_size=1024 * 1024):
#                         if chunk:
#                             f.write(chunk)
            
#             # Rename after the download completes
#             os.replace(temp_path, out_path)
#             return ("OK", taxid, "downloaded")

#         except Exception as e:
#             if attempt < MAX_RETRIES - 1:
#                 continue # Retry
#             else:
#                 # Delete any remaining temporary file
#                 if os.path.exists(temp_path):
#                     try:
#                         os.remove(temp_path)
#                     except:
#                         pass
#                 return ("ERROR", taxid, str(e))

#     return ("ERROR", taxid, "Unknown error")


# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # Read TAXIDs
#     taxids = load_taxids_from_third_column(SPECIES_FILE)
    
#     if not taxids:
#         print("No download tasks are required.")
#         return

#     failed = []

#     print(f"🚀 Starting concurrent downloads (Workers={MAX_WORKERS})...")

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(download_one, taxid): taxid for taxid in taxids}

#         for future in as_completed(futures):
#             status, taxid, msg = future.result()

#             if status == "OK":
#                 print(f"[OK]   {taxid}")
#             elif status == "SKIP":
#                 print(f"[SKIP] {taxid}")
#             else:
#                 print(f"[FAIL] {taxid} -> {msg}")
#                 failed.append(taxid)

#     # Write the failure log
#     if failed:
#         with open(FAILED_LOG, "w") as f:
#             for t in failed:
#                 f.write(f"{t}\n")
#         print(f"\n⚠️ {len(failed)} files failed to download and were recorded in: {FAILED_LOG}")
#         print("Recommendation: Check the network connection and rerun this script to retry.")
#     else:
#         print("\n✅ All required species files are ready!")


# if __name__ == "__main__":
#     main()






import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ===================== Configuration =====================

SPECIES_DIR = r"string_species_links_v12"     # Directory containing per-species PPI files
MAPPING_FILE = r"pdb_uniprot_string_supplement.txt"
OUTPUT_DIR = r"ppi_from_species_2hop_valid_100_10_supplement" 

MISSING_LOG = "missing_species_files.txt"

# ✅ Core configuration: Different Top-K values for two levels
TOP_K_1HOP = 100  # First hop: Select the 100 strongest neighbors per seed
TOP_K_2HOP = 10   # Second hop: Select the 10 strongest neighbors per neighbor

# ===============================================


def load_mapping_grouped_by_taxid(mapping_file):
    """
    Read the PDB-STRING mapping table and group records by TaxID.
    """
    taxid_to_string_to_pdbs = defaultdict(lambda: defaultdict(list))
    records_all = []

    print(f"📖 Loading mapping file: {mapping_file} ...")
    with open(mapping_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            pdb_chain = parts[0]
            string_id = parts[2]
            
            if string_id == "NA": continue
            if "." not in string_id: continue

            taxid = string_id.split(".")[0]

            taxid_to_string_to_pdbs[taxid][string_id].append(pdb_chain)
            records_all.append((pdb_chain, string_id, taxid))

    print(f"✅ Total mappings: {len(records_all)}")
    print(f"✅ Species involved: {len(taxid_to_string_to_pdbs)}")
    return taxid_to_string_to_pdbs, records_all


def build_1hop_2hop_for_species_mixed_k(df, seeds_this_species, k1=100, k2=10):
    """
    Hybrid Top-K construction logic:
    1. Use k1 for 1-hop neighbors
    2. Use k2 for 2-hop neighbors
    """
    edges_1hop = defaultdict(list)
    edges_2hop = defaultdict(list)
    valid_neighbors_map = defaultdict(set)

    # -------------------------------------------------------
    # 1. Process 1-hop edges (Seed -> Neighbor, Top K1)
    # -------------------------------------------------------
    # Filter edges related to the seed
    mask1 = df["protein1"].isin(seeds_this_species) | df["protein2"].isin(seeds_this_species)
    df_1hop_all = df[mask1]

    # Expand bidirectionally so the seed is in protein1, simplifying grouping
    s1 = df_1hop_all[df_1hop_all["protein1"].isin(seeds_this_species)].copy()
    s1.columns = ["seed", "neighbor", "score"]
    
    s2 = df_1hop_all[df_1hop_all["protein2"].isin(seeds_this_species)].copy()
    s2.columns = ["neighbor", "seed", "score"]
    s2 = s2[["seed", "neighbor", "score"]] # Reorder columns
    
    df_1hop_unified = pd.concat([s1, s2], ignore_index=True)
    
    # Group by seed and select the K1 highest scores
    df_1hop_topk = (
        df_1hop_unified.sort_values(["seed", "score"], ascending=[True, False])
        .groupby("seed")
        .head(k1)
    )

    # Record results
    for row in df_1hop_topk.itertuples(index=False):
        # Exclude self-loops (STRING usually has none, but check for safety)
        if row.seed == row.neighbor: continue
        
        edges_1hop[row.seed].append((row.seed, row.neighbor, row.score))
        valid_neighbors_map[row.neighbor].add(row.seed) # Record which seed connects to this neighbor

    all_valid_neighbors = set(valid_neighbors_map.keys())
    
    if not all_valid_neighbors:
        return edges_1hop, edges_2hop

    # -------------------------------------------------------
    # 2. Process 2-hop edges (Neighbor -> Next_Neighbor, Top K2)
    # -------------------------------------------------------
    
    # Filter edges related to 1-hop neighbors
    mask2 = df["protein1"].isin(all_valid_neighbors) | df["protein2"].isin(all_valid_neighbors)
    df_2hop_all = df[mask2]
    
    n1 = df_2hop_all[df_2hop_all["protein1"].isin(all_valid_neighbors)].copy()
    n1.columns = ["neighbor", "next_neighbor", "score"]
    
    n2 = df_2hop_all[df_2hop_all["protein2"].isin(all_valid_neighbors)].copy()
    n2.columns = ["next_neighbor", "neighbor", "score"]
    n2 = n2[["neighbor", "next_neighbor", "score"]]
    
    df_2hop_unified = pd.concat([n1, n2], ignore_index=True)
    
    # Group by neighbor and select the K2 highest scores
    df_2hop_topk = (
        df_2hop_unified.sort_values(["neighbor", "score"], ascending=[True, False])
        .groupby("neighbor")
        .head(k2)
    )
    
    # Associate 2-hop edges back with the original seed
    for row in df_2hop_topk.itertuples(index=False):
        neighbor = row.neighbor
        next_neighbor = row.next_neighbor
        
        # Find all original seeds connected to this neighbor
        parent_seeds = valid_neighbors_map.get(neighbor, set())
        
        for seed in parent_seeds:
            # Prevent backtracking (A->B->A)
            if next_neighbor == seed:
                continue
            # Record (B, C, score) in the subgraph belonging to seed A
            edges_2hop[seed].append((neighbor, next_neighbor, row.score))

    return edges_1hop, edges_2hop


def write_pdb_files_with_resume(edges_1hop, edges_2hop, string_to_pdbs, outdir):
    """
    Write files.
    """
    os.makedirs(outdir, exist_ok=True)
    written_count = 0

    for seed, pdb_list in string_to_pdbs.items():
        # Check whether all PDB files for this seed already exist
        # If so, skip computation and writing
        targets = [os.path.join(outdir, f"{p}.txt") for p in pdb_list]
        if all(os.path.exists(t) for t in targets):
            continue

        # --- Merge data ---
        rec = []
        rec.extend(edges_1hop.get(seed, []))
        rec.extend(edges_2hop.get(seed, []))

        if not rec:
            df_seed = pd.DataFrame(columns=["protein1", "protein2", "score"])
        else:
            df_seed = pd.DataFrame(rec, columns=["p1_temp", "p2_temp", "score"])
            
            # Sorting ensures consistent undirected edges (min, max)
            a = df_seed["p1_temp"]
            b = df_seed["p2_temp"]
            df_seed["protein1"] = list(map(min, zip(a, b)))
            df_seed["protein2"] = list(map(max, zip(a, b)))
            
            # Deduplicate by retaining the highest score
            df_seed = (
                df_seed[["protein1", "protein2", "score"]]
                .sort_values(["protein1", "protein2", "score"], ascending=[True, True, False])
                .drop_duplicates(subset=["protein1", "protein2"], keep="first")
            )

        # Write PDB files one by one
        for pdb_chain in pdb_list:
            out_path = os.path.join(outdir, f"{pdb_chain}.txt")
            # Check again to guard against multiple threads or other races (though this code is single-threaded)
            if os.path.exists(out_path):
                continue
            
            df_seed.to_csv(out_path, sep="\t", index=False)
            written_count += 1
            
    return written_count


def main():
    taxid_to_string_to_pdbs, records_all = load_mapping_grouped_by_taxid(MAPPING_FILE)
    missing_lines = []

    # Iterate directly over all TaxIDs with a progress bar
    # Wrap keys in list() to ensure a fixed order
    all_taxids = list(taxid_to_string_to_pdbs.keys())
    
    print(f"🚀 Starting tasks for {len(all_taxids)} species...")
    
    pbar = tqdm(all_taxids, desc="Processing Species")
    
    for taxid in pbar:
        string_to_pdbs = taxid_to_string_to_pdbs[taxid]
        
        # ----------------------------------------------------
        # ⭐ Optimized resume check ⭐
        # Before loading the large CSV, check whether *all* PDBs for this species already have results
        # If all exist, continue immediately without reading the CSV
        # ----------------------------------------------------
        all_done = True
        # Checking every file may incur I/O overhead for species with many PDBs,
        # but this is acceptable compared with reading tens of gigabytes of CSV data.
        for pdb_list in string_to_pdbs.values():
            for p in pdb_list:
                if not os.path.exists(os.path.join(OUTPUT_DIR, f"{p}.txt")):
                    all_done = False
                    break
            if not all_done: break
        
        if all_done:
            # pbar.write(f"[{taxid}] Skipped (complete)")
            continue
        # ----------------------------------------------------

        pbar.set_postfix({"TaxID": taxid, "Status": "Init"})
        species_file = os.path.join(SPECIES_DIR, f"{taxid}.protein.links.v12.0.txt.gz")

        if not os.path.exists(species_file):
            # pbar.write(f"⚠️ Missing species file: {taxid}")
            for string_id, pdb_list in string_to_pdbs.items():
                for pdb_chain in pdb_list:
                    missing_lines.append(f"{pdb_chain}\t{string_id}\t{taxid}")
            continue

        try:
            # Step 1: Read CSV (C-engine acceleration + memory optimization)
            pbar.set_postfix({"TaxID": taxid, "Status": "Reading CSV"})
            
            df = pd.read_csv(
                species_file,
                sep=" ",  # STRING v12 is space-delimited
                compression="gzip",
                usecols=["protein1", "protein2", "combined_score"],
                dtype={"protein1": "string", "protein2": "string", "combined_score": "int32"}
            )
            df = df.rename(columns={"combined_score": "score"})
            
            # Step 2: Compute Top-K subgraphs
            pbar.set_postfix({"TaxID": taxid, "Status": "Graph Build"})
            seeds_this_species = set(string_to_pdbs.keys())
            
            edges_1hop, edges_2hop = build_1hop_2hop_for_species_mixed_k(
                df, seeds_this_species, k1=TOP_K_1HOP, k2=TOP_K_2HOP
            )

            # Step 3: Write files
            pbar.set_postfix({"TaxID": taxid, "Status": "Writing"})
            count = write_pdb_files_with_resume(edges_1hop, edges_2hop, string_to_pdbs, OUTPUT_DIR)
            
            # pbar.write(f"[{taxid}] Generated {count} files")

        except Exception as e:
            pbar.write(f"❌ Error processing species {taxid}: {e}")
            continue

    if missing_lines:
        with open(MISSING_LOG, "w") as f:
            for line in missing_lines:
                f.write(line + "\n")
        print(f"\n⚠️ Missing records written to: {MISSING_LOG}")
    else:
        print("\n✅ All tasks complete!")

if __name__ == "__main__":
    main()




