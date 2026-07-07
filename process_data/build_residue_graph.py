import os
import numpy as np
import torch
import dgl
from tqdm import tqdm

# 固定 DGL 后端
os.environ["DGLBACKEND"] = "pytorch"
os.environ["DGL_GRAPHBOLT_LOAD"] = "0"   # 禁用 GraphBolt

def build_structure_graph(prot_id, ca_dist_matrix, residue_feat, threshold=10.0):
    """
    构建一个 DGL 图
    - prot_id: 蛋白质 ID
    - ca_dist_matrix: numpy array, shape [L, L]
    - residue_feat: numpy array, shape [F, d]
    - threshold: 距离阈值，小于该值就连边
    """
    L = ca_dist_matrix.shape[0]
    F = residue_feat.shape[0]

    if L != F:
        print(f"[ERROR] {prot_id}: Distance matrix = {L}, Residue features = {F}")
        return None

    u, v, dis = [], [], []
    for i in range(L):
        for j in range(L):
            if i != j and ca_dist_matrix[i, j] < threshold:
                u.append(i)
                v.append(j)
                dis.append(ca_dist_matrix[i, j])

    g = dgl.graph((u, v), num_nodes=L)
    g = dgl.add_self_loop(g)
    g.edata['dis'] = torch.tensor(dis + [0.0] * L, dtype=torch.float32)  # 给自环赋 0 距离
    g.ndata['feat'] = torch.tensor(residue_feat, dtype=torch.float32)

    return g


# ----------------------
# 主流程
# ----------------------
ca_dir = "ca_matrices_train"
feat_dir = "protT5_embeddings_train"
graph_dir = "graphs_train"
os.makedirs(graph_dir, exist_ok=True)

protein_list_file = "protein_id_and_sequence_train.txt"
invalid_protein_file = os.path.join(graph_dir, "invalid_proteins.txt")

graphs = []

with open(protein_list_file, "r") as f, open(invalid_protein_file, "a") as invalid_f:
    for line in tqdm(f, desc="Building graphs"):
        if not line.strip():
            continue

        prot_id, seq = line.strip().split()

        out_path = os.path.join(graph_dir, f"{prot_id}.bin")

        # ------------------------------
        # ⭐ 如果文件已经存在 → 直接跳过
        # ------------------------------
        if os.path.exists(out_path):
            # print(f"[SKIP] {prot_id} 已存在 .bin 文件，跳过构图")
            continue

        # ------------------------------
        # 检查 CA 和 ProtT5 文件是否存在
        # ------------------------------
        ca_path = os.path.join(ca_dir, f"{prot_id}_ca.npy")
        feat_path = os.path.join(feat_dir, f"{prot_id}.pt")

        if not os.path.exists(ca_path) or not os.path.exists(feat_path):
            print(f"[MISSING] {prot_id}: 缺少 CA 或 T5，跳过。")
            invalid_f.write(f"{prot_id}\n")
            continue

        # ------------------------------
        # 加载 CA 矩阵与特征
        # ------------------------------
        ca_dist_matrix = np.load(ca_path)
        residue_feat = torch.load(feat_path).numpy()

        # ------------------------------
        # 构建 DGL 图
        # ------------------------------
        g = build_structure_graph(
            prot_id,
            ca_dist_matrix,
            residue_feat,
            threshold=10.0
        )

        if g is None:
            print(f"[INVALID] {prot_id}: 构图失败，跳过。")
            invalid_f.write(f"{prot_id}\n")
            continue

        # ------------------------------
        # 保存图
        # ------------------------------
        dgl.save_graphs(out_path, [g])
        graphs.append(g)

print(f"构建完成: 共生成 {len(graphs)} 个新图，已保存到 {graph_dir}/")
print(f"无效/缺失蛋白记录文件: {invalid_protein_file}")