import os
import numpy as np
import torch
import dgl
from tqdm import tqdm

# Pin the DGL backend
os.environ["DGLBACKEND"] = "pytorch"
os.environ["DGL_GRAPHBOLT_LOAD"] = "0"   # Disable GraphBolt

def build_structure_graph(prot_id, ca_dist_matrix, residue_feat, threshold=10.0):
    """
    Build a DGL graph.
    - prot_id: Protein ID
    - ca_dist_matrix: numpy array, shape [L, L]
    - residue_feat: numpy array, shape [F, d]
    - threshold: Distance threshold; connect nodes below this value
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
    g.edata['dis'] = torch.tensor(dis + [0.0] * L, dtype=torch.float32)  # Assign distance 0 to self-loops
    g.ndata['feat'] = torch.tensor(residue_feat, dtype=torch.float32)

    return g


# ----------------------
# Main workflow
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
        # ⭐ Skip directly if the file already exists
        # ------------------------------
        if os.path.exists(out_path):
            # print(f"[SKIP] {prot_id} already has a .bin file; skipping graph construction")
            continue

        # ------------------------------
        # Check whether the CA and ProtT5 files exist
        # ------------------------------
        ca_path = os.path.join(ca_dir, f"{prot_id}_ca.npy")
        feat_path = os.path.join(feat_dir, f"{prot_id}.pt")

        if not os.path.exists(ca_path) or not os.path.exists(feat_path):
            print(f"[MISSING] {prot_id}: CA or T5 data is missing; skipping.")
            invalid_f.write(f"{prot_id}\n")
            continue

        # ------------------------------
        # Load the CA matrix and features
        # ------------------------------
        ca_dist_matrix = np.load(ca_path)
        residue_feat = torch.load(feat_path).numpy()

        # ------------------------------
        # Build the DGL graph
        # ------------------------------
        g = build_structure_graph(
            prot_id,
            ca_dist_matrix,
            residue_feat,
            threshold=10.0
        )

        if g is None:
            print(f"[INVALID] {prot_id}: Graph construction failed; skipping.")
            invalid_f.write(f"{prot_id}\n")
            continue

        # ------------------------------
        # Save the graph
        # ------------------------------
        dgl.save_graphs(out_path, [g])
        graphs.append(g)

print(f"Build complete: Generated {len(graphs)} new graphs and saved them to {graph_dir}/")
print(f"Invalid/missing protein log: {invalid_protein_file}")
