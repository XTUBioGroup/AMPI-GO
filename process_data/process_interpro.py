# import pandas as pd
# import pickle as pkl
# from scipy.sparse import csr_matrix
# from tqdm.auto import tqdm
# import numpy as np
# from pathlib import Path
# import os


# # ======= 读取 PID 列表 =======
# def load_pid_from_seq_file(path):
#     pid_list = []
#     with open(path, "r") as f:
#         for line in f:
#             if line.strip():
#                 pid_list.append(line.split()[0])
#     return pid_list



# # ======= 按 PID 顺序构造 InterPro CSR 矩阵 =======
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



# # ======= 这是你要修改的核心函数 =======
# def build_interpro_features_only_ipr(
#     pid_list,                # ← 改：从 train_pid_list 传入
#     interpro_folder,
#     save_feature='interpro_feature.pkl',
#     save_map='domain_map.pkl'
# ):
#     interpro_folder = Path(interpro_folder)

#     protein_info = {}
#     domain_set = set()

#     print(f"📂 Building InterPro feature by given PID list ({len(pid_list)} proteins)")

#     # 🚀 关键修改：严格按 pid_list 顺序读取对应文件
#     for pid in tqdm(pid_list, desc="Reading InterProScan TSV"):
#         file = interpro_folder / f"{pid}.tsv"

#         # 文件不存在或为空 → 空 domain 列表
#         if (not file.exists()) or os.path.getsize(file) == 0:
#             protein_info[pid] = []
#             continue

#         try:
#             df = pd.read_csv(file, sep='\t', header=None, comment='#', dtype=str)
#         except pd.errors.EmptyDataError:
#             protein_info[pid] = []
#             continue

#         # InterPro ID 一般在列 11
#         if 11 not in df.columns:
#             protein_info[pid] = []
#             continue

#         # 只保留 IPRxxxxx
#         ipr_list = [
#             x for x in df[11].dropna().unique()
#             if isinstance(x, str) and x.startswith("IPR")
#         ]

#         protein_info[pid] = ipr_list
#         domain_set.update(ipr_list)

#     # ===== 构建 domain_map（必须只用训练集） =====
#     domain_map = {ipr: i for i, ipr in enumerate(sorted(domain_set))}
#     pkl.dump(domain_map, open(save_map, 'wb'))
#     print(f"✅ Saved domain map → {save_map}  ({len(domain_map)} InterPro IDs)")

#     # ===== 构建 InterPro 矩阵 =====
#     get_interpro_matrix(pid_list, protein_info, domain_map, save_feature)

#     # 保存 PID 顺序（可用于 debug）
#     pkl.dump(pid_list, open(Path(save_feature).with_suffix(".pids.pkl"), 'wb'))
#     print(f"📌 Saved PID order → {Path(save_feature).with_suffix('.pids.pkl')}")


# def build_interpro_valid(pid_list, interpro_folder, domain_map_path, save_feature):
#     """
#     验证集 InterPro 构建，复用训练集 domain_map
#     """
#     interpro_folder = Path(interpro_folder)

#     # 加载训练集 domain_map（不能重新构建，否则维度不一致）
#     domain_map = pkl.load(open(domain_map_path, "rb"))
#     domain_dim = len(domain_map)

#     rows, cols, data = [], [], []

#     print(f"📂 Building VALID InterPro with fixed domain_map ({domain_dim} dims)")

#     for i, pid in enumerate(tqdm(pid_list, desc="Reading valid InterPro")):
#         file = interpro_folder / f"{pid}.tsv"

#         # 空文件 → 全 0
#         if (not file.exists()) or os.path.getsize(file) == 0:
#             continue

#         try:
#             df = pd.read_csv(file, sep='\t', header=None, comment='#', dtype=str)
#         except:
#             continue

#         if 11 not in df.columns:
#             continue

#         # 扫描每个 IPR
#         for x in df[11].dropna().unique():
#             if isinstance(x, str) and x in domain_map:
#                 rows.append(i)
#                 cols.append(domain_map[x])
#                 data.append(1)

#     # 构建 CSR（shape 固定 = 验证集蛋白数量 × 训练集 domain_map 维度）
#     mat = csr_matrix(
#         (data, (rows, cols)),
#         shape=(len(pid_list), domain_dim)
#     )

#     pkl.dump(mat, open(save_feature, "wb"))
#     print(f"✅ Saved VALID InterPro → {save_feature}, shape={mat.shape}")

#     return mat


# # ======= main（训练集用） =======
# if __name__ == "__main__":
#     # 你的训练集 PID 列表文件（两列：PDB_CHAIN + sequence）
#     train_pid_file = "protein_id_and_sequence_train.txt"

#     # InterProScan 的结果目录
#     folder = "interpro_train"

#     # 输出文件
#     save_feature = "interpro_feature_train.pkl"
#     save_map = "domain_map_train.pkl"

#     # 1) 先读取 PID 顺序（至关重要）
#     pid_list = load_pid_from_seq_file(train_pid_file)

#     # 2) 再按顺序构建 InterPro 特征
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
#     #     interpro_folder="interpro_valid",         # 你的验证集 .tsv 存放位置
#     #     domain_map_path="domain_map_train.pkl",   # 复用训练 map
#     #     save_feature="interpro_feature_valid.pkl"
#     # )



import pandas as pd
import pickle as pkl
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import os

# ======= 读取 PID 列表 =======
def load_pid_from_seq_file(path):
    pid_list = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                # 无论是 "ID 序列" 格式，还是纯 "ID" 格式，split()[0] 都能精准拿到 ID
                pid_list.append(line.split()[0])
    return pid_list

# ======= 按 PID 顺序构造 InterPro CSR 矩阵 =======
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

# ======= 训练集构建（生成并保存 domain_map） =======
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

# ======= 验证/测试集构建（严格复用 domain_map） =======
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
# ======= 自动化批处理主程序 =======
# ==========================================
if __name__ == "__main__":
    # 路径配置区 (请确保与服务器实际路径一致)
    # 你之前生成的 9 个拆分列表文件夹
    SPLIT_DIR = Path("data_spilt_big")
    # 你存放所有 tsv 文件的总文件夹
    INTERPRO_DIR = Path("interpro_big")
    # 生成的特征矩阵存放位置
    OUT_DIR = Path("interpro_features_big")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    ontologies = ['bp', 'mf', 'cc']

    print("🚀 开始批量构建 InterPro 特征矩阵...")

    for ont in ontologies:
        print(f"\n{'='*40}\n🌟 Processing Ontology: {ont.upper()}\n{'='*40}")
        
        # 定义文件路径
        train_txt = SPLIT_DIR / f"{ont}_train_ids.txt"
        test1_txt = SPLIT_DIR / f"{ont}_valid_ids.txt"
        test2_txt = SPLIT_DIR / f"{ont}_test_ids.txt"

        train_feat = OUT_DIR / f"{ont}_train_interpro.pkl"
        test1_feat = OUT_DIR / f"{ont}_valid_interpro.pkl"
        test2_feat = OUT_DIR / f"{ont}_test_interpro.pkl"
        domain_map_file = OUT_DIR / f"{ont}_domain_map.pkl"

        # ---------------- 1. 处理 Train (生成 Map) ----------------
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

        # ---------------- 2. 处理 Test1 (复用 Map) ----------------
        if test1_txt.exists():
            print(f"\n🛠️ 2. Building Test1 Matrix for {ont.upper()}...")
            test1_pids = load_pid_from_seq_file(test1_txt)
            build_interpro_valid(
                pid_list=test1_pids,
                interpro_folder=INTERPRO_DIR,
                domain_map_path=domain_map_file,
                save_feature=test1_feat
            )

        # ---------------- 3. 处理 Test2 (复用 Map) ----------------
        if test2_txt.exists():
            print(f"\n🛠️ 3. Building Test2 Matrix for {ont.upper()}...")
            test2_pids = load_pid_from_seq_file(test2_txt)
            build_interpro_valid(
                pid_list=test2_pids,
                interpro_folder=INTERPRO_DIR,
                domain_map_path=domain_map_file,
                save_feature=test2_feat
            )

    print("\n🎉 全部特征矩阵构建完毕！")