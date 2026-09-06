# import pandas as pd
# import pickle as pkl
# from scipy.sparse import csr_matrix
# from tqdm.auto import tqdm
# import numpy as np
# from pathlib import Path
# import os


# # ======= Read the PID list =======
# def load_pid_from_seq_file(path):
#     pid_list = []
#     with open(path, "r") as f:
#         for line in f:
#             if line.strip():
#                 pid_list.append(line.split()[0])
#     return pid_list



# # ======= Build the InterPro CSR matrix in PID order =======
# def get_interpro_matrix(pid_list, protein_info, domain_map, save_file):
#     rows, cols, data = [], [], []

#     for i, pid in enumerate(tqdm(pid_list, desc="Building InterPro matrix")):
#         if pid not in protein_info:
#             continue
#         for ipr in protein_info[pid]:
#             if ipr in domain_map:
#                 rows.append(i)
#                 cols.append(domain_map[ipr])
#                 data.append(1)

#     mat = csr_matrix(
#         (data, (rows, cols)),
#         shape=(len(pid_list), len(domain_map))
#     )

#     pkl.dump(mat, open(save_file, 'wb'))
#     print(f"✅ Saved InterPro matrix → {save_file}")
#     print(f"Matrix shape: {mat.shape}")

#     return mat



# # ======= Core function to modify =======
# def build_interpro_features_only_ipr(
#     pid_list,                # ← Change: Pass from train_pid_list
#     interpro_folder,
#     save_feature='interpro_feature.pkl',
#     save_map='domain_map.pkl'
# ):
#     interpro_folder = Path(interpro_folder)

#     protein_info = {}
#     domain_set = set()

#     print(f"📂 Building InterPro feature by given PID list ({len(pid_list)} proteins)")

#     # 🚀 Key change: Read the corresponding files strictly in pid_list order
#     for pid in tqdm(pid_list, desc="Reading InterProScan TSV"):
#         file = interpro_folder / f"{pid}.tsv"

#         # Missing or empty file -> empty domain list
#         if (not file.exists()) or os.path.getsize(file) == 0:
#             protein_info[pid] = []
#             continue

#         try:
#             df = pd.read_csv(file, sep='\t', header=None, comment='#', dtype=str)
#         except pd.errors.EmptyDataError:
#             protein_info[pid] = []
#             continue

#         # InterPro IDs are usually in column 11
#         if 11 not in df.columns:
#             protein_info[pid] = []
#             continue

#         # Keep only IPRxxxxx entries
#         ipr_list = [
#             x for x in df[11].dropna().unique()
#             if isinstance(x, str) and x.startswith("IPR")
#         ]

#         protein_info[pid] = ipr_list
#         domain_set.update(ipr_list)

#     # ===== Build domain_map (training set only) =====
#     domain_map = {ipr: i for i, ipr in enumerate(sorted(domain_set))}
#     pkl.dump(domain_map, open(save_map, 'wb'))
#     print(f"✅ Saved domain map → {save_map}  ({len(domain_map)} InterPro IDs)")

#     # ===== Build the InterPro matrix =====
#     get_interpro_matrix(pid_list, protein_info, domain_map, save_feature)

#     # Save PID order (useful for debugging)
#     pkl.dump(pid_list, open(Path(save_feature).with_suffix(".pids.pkl"), 'wb'))
#     print(f"📌 Saved PID order → {Path(save_feature).with_suffix('.pids.pkl')}")


# def build_interpro_valid(pid_list, interpro_folder, domain_map_path, save_feature):
#     """
#     Build validation-set InterPro features using the training-set domain_map.
#     """
#     interpro_folder = Path(interpro_folder)

#     # Load the training-set domain_map (do not rebuild it, or dimensions will differ)
#     domain_map = pkl.load(open(domain_map_path, "rb"))
#     domain_dim = len(domain_map)

#     rows, cols, data = [], [], []

#     print(f"📂 Building VALID InterPro with fixed domain_map ({domain_dim} dims)")

#     for i, pid in enumerate(tqdm(pid_list, desc="Reading valid InterPro")):
#         file = interpro_folder / f"{pid}.tsv"

#         # Empty file -> all zeros
#         if (not file.exists()) or os.path.getsize(file) == 0:
#             continue

#         try:
#             df = pd.read_csv(file, sep='\t', header=None, comment='#', dtype=str)
#         except:
#             continue

#         if 11 not in df.columns:
#             continue

#         # Scan each IPR
#         for x in df[11].dropna().unique():
#             if isinstance(x, str) and x in domain_map:
#                 rows.append(i)
#                 cols.append(domain_map[x])
#                 data.append(1)

#     # Build CSR (fixed shape = validation protein count x training domain_map dimension)
#     mat = csr_matrix(
#         (data, (rows, cols)),
#         shape=(len(pid_list), domain_dim)
#     )

#     pkl.dump(mat, open(save_feature, "wb"))
#     print(f"✅ Saved VALID InterPro → {save_feature}, shape={mat.shape}")

#     return mat


# # ======= main (for the training set) =======
# if __name__ == "__main__":
#     # Training PID list file (two columns: PDB_CHAIN + sequence)
#     train_pid_file = "protein_id_and_sequence_train.txt"

#     # InterProScan results directory
#     folder = "interpro_train"

#     # Output files
#     save_feature = "interpro_feature_train.pkl"
#     save_map = "domain_map_train.pkl"

#     # 1) Read PID order first (critical)
#     pid_list = load_pid_from_seq_file(train_pid_file)

#     # 2) Build InterPro features in that order
#     build_interpro_features_only_ipr(
#         pid_list,
#         interpro_folder=folder,
#         save_feature=save_feature,
#         save_map=save_map
#     )
#     # valid_pid_file = "protein_id_and_sequence_valid.txt"
#     # valid_pid_list = load_pid_from_seq_file(valid_pid_file)

#     # build_interpro_valid(
#     #     pid_list=valid_pid_list,
#     #     interpro_folder="interpro_valid",         # Directory containing validation .tsv files
#     #     domain_map_path="domain_map_train.pkl",   # Reuse the training map
#     #     save_feature="interpro_feature_valid.pkl"
#     # )



import pandas as pd
import pickle as pkl
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import os

# ======= Read the PID list =======
def load_pid_from_seq_file(path):
    pid_list = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                # split()[0] extracts the ID from either "ID sequence" format or a plain "ID" format
                pid_list.append(line.split()[0])
    return pid_list

# ======= Build the InterPro CSR matrix in PID order =======
def get_interpro_matrix(pid_list, protein_info, domain_map, save_file):
    rows, cols, data = [], [], []

    for i, pid in enumerate(tqdm(pid_list, desc="Building InterPro matrix", leave=False)):
        if pid not in protein_info:
            continue
        for ipr in protein_info[pid]:
            if ipr in domain_map:
                rows.append(i)
                cols.append(domain_map[ipr])
                data.append(1)

    mat = csr_matrix(
        (data, (rows, cols)),
        shape=(len(pid_list), len(domain_map))
    )

    pkl.dump(mat, open(save_file, 'wb'))
    print(f"  ✅ Saved InterPro matrix → {save_file.name} | Shape: {mat.shape}")

    return mat

# ======= Build the training set (generate and save domain_map) =======
def build_interpro_features_only_ipr(
    pid_list,
    interpro_folder,
    save_feature,
    save_map
):
    interpro_folder = Path(interpro_folder)
    protein_info = {}
    domain_set = set()

    for pid in tqdm(pid_list, desc="Reading Train TSVs", leave=False):
        file = interpro_folder / f"{pid}.tsv"

        if (not file.exists()) or os.path.getsize(file) == 0:
            protein_info[pid] = []
            continue

        try:
            df = pd.read_csv(file, sep='\t', header=None, comment='#', dtype=str, on_bad_lines='skip')
        except:
            protein_info[pid] = []
            continue

        if 11 not in df.columns:
            protein_info[pid] = []
            continue

        ipr_list = [
            x for x in df[11].dropna().unique()
            if isinstance(x, str) and x.startswith("IPR")
        ]

        protein_info[pid] = ipr_list
        domain_set.update(ipr_list)

    domain_map = {ipr: i for i, ipr in enumerate(sorted(domain_set))}
    pkl.dump(domain_map, open(save_map, 'wb'))
    print(f"  ✅ Saved domain map → {save_map.name} ({len(domain_map)} IPR IDs)")

    get_interpro_matrix(pid_list, protein_info, domain_map, save_feature)

# ======= Build validation/test sets (strictly reuse domain_map) =======
def build_interpro_valid(pid_list, interpro_folder, domain_map_path, save_feature):
    interpro_folder = Path(interpro_folder)
    domain_map = pkl.load(open(domain_map_path, "rb"))
    domain_dim = len(domain_map)

    rows, cols, data = [], [], []

    for i, pid in enumerate(tqdm(pid_list, desc="Reading Valid/Test TSVs", leave=False)):
        file = interpro_folder / f"{pid}.tsv"

        if (not file.exists()) or os.path.getsize(file) == 0:
            continue

        try:
            df = pd.read_csv(file, sep='\t', header=None, comment='#', dtype=str, on_bad_lines='skip')
        except:
            continue

        if 11 not in df.columns:
            continue

        for x in df[11].dropna().unique():
            if isinstance(x, str) and x in domain_map:
                rows.append(i)
                cols.append(domain_map[x])
                data.append(1)

    mat = csr_matrix(
        (data, (rows, cols)),
        shape=(len(pid_list), domain_dim)
    )

    pkl.dump(mat, open(save_feature, "wb"))
    print(f"  ✅ Saved VALID/TEST InterPro → {save_feature.name} | Shape={mat.shape}")
    return mat

# ==========================================
# ======= Automated batch-processing main program =======
# ==========================================
if __name__ == "__main__":
    # Path configuration (ensure paths match the server)
    # Directory containing the nine split lists generated previously
    SPLIT_DIR = Path("data_spilt_big")
    # Root directory containing all TSV files
    INTERPRO_DIR = Path("interpro_big")
    # Output directory for generated feature matrices
    OUT_DIR = Path("interpro_features_big")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    ontologies = ['bp', 'mf', 'cc']

    print("🚀 Starting batch construction of InterPro feature matrices...")

    for ont in ontologies:
        print(f"\n{'='*40}\n🌟 Processing Ontology: {ont.upper()}\n{'='*40}")
        
        # Define file paths
        train_txt = SPLIT_DIR / f"{ont}_train_ids.txt"
        test1_txt = SPLIT_DIR / f"{ont}_valid_ids.txt"
        test2_txt = SPLIT_DIR / f"{ont}_test_ids.txt"

        train_feat = OUT_DIR / f"{ont}_train_interpro.pkl"
        test1_feat = OUT_DIR / f"{ont}_valid_interpro.pkl"
        test2_feat = OUT_DIR / f"{ont}_test_interpro.pkl"
        domain_map_file = OUT_DIR / f"{ont}_domain_map.pkl"

        # ---------------- 1. Process Train (generate map) ----------------
        if train_txt.exists():
            print(f"\n🛠️ 1. Building Train Matrix for {ont.upper()}...")
            train_pids = load_pid_from_seq_file(train_txt)
            build_interpro_features_only_ipr(
                pid_list=train_pids,
                interpro_folder=INTERPRO_DIR,
                save_feature=train_feat,
                save_map=domain_map_file
            )
        else:
            print(f"❌ Missing {train_txt.name}, skipping {ont.upper()}...")
            continue

        # ---------------- 2. Process Test1 (reuse map) ----------------
        if test1_txt.exists():
            print(f"\n🛠️ 2. Building Test1 Matrix for {ont.upper()}...")
            test1_pids = load_pid_from_seq_file(test1_txt)
            build_interpro_valid(
                pid_list=test1_pids,
                interpro_folder=INTERPRO_DIR,
                domain_map_path=domain_map_file,
                save_feature=test1_feat
            )

        # ---------------- 3. Process Test2 (reuse map) ----------------
        if test2_txt.exists():
            print(f"\n🛠️ 3. Building Test2 Matrix for {ont.upper()}...")
            test2_pids = load_pid_from_seq_file(test2_txt)
            build_interpro_valid(
                pid_list=test2_pids,
                interpro_folder=INTERPRO_DIR,
                domain_map_path=domain_map_file,
                save_feature=test2_feat
            )

    print("\n🎉 All feature matrices have been built!")
