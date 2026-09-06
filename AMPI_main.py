import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import torch
import numpy as np
import dgl
from dgl.dataloading import GraphDataLoader
import pickle as pkl

from tqdm.auto import tqdm
import json
from sklearn.preprocessing import MultiLabelBinarizer

from GCN_model import combine_inter_model
from objective import AverageMeter
from model_utils import test_performance_gnn_inter, FocalLoss


##########################################################
# Data-loading utility functions
##########################################################
def load_pid_from_seq_file(path):
    pid_list = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                pid_list.append(line.split()[0])
    return pid_list


def load_and_fix_graph(bin_path):
    g = dgl.load_graphs(bin_path)[0][0]
    if "feat" in g.ndata:
        g.ndata["x"] = g.ndata["feat"]
        del g.ndata["feat"]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g


def load_graphs_for_pid_list(pid_list, folder):
    graphs = []
    for pid in tqdm(pid_list, desc=f"Loading graphs from {folder}"):
        bin_path = os.path.join(folder, pid + ".bin")
        graphs.append(load_and_fix_graph(bin_path))
    return graphs

def load_ppi_graph_for_pid_list(pid_list, folder):
    """
    Load PPI graphs specifically:
    ⚠️ Preserve self-loops and edge weights from the graph-building script.
    """
    graphs = []
    for pid in tqdm(pid_list, desc=f"Loading PPI graphs from {folder}"):
        bin_path = os.path.join(folder, pid + ".bin")
        g = dgl.load_graphs(bin_path)[0][0]
        graphs.append(g)
    return graphs

def load_pdb2go(path):
    """Read the previously generated pdb2go.json file."""
    with open(path, "r") as f:
        return json.load(f)


def build_go_lists_for_ontology(pid_list, pdb2go, ont_key):
    """


    ont_key can be 'molecular_function', 'biological_process', or 'cellular_component'.
    """
    go_lists = []

    for pid in pid_list:
        term_set = set()

        if pid in pdb2go:
            # For example, pdb2go[pid]["molecular_function"] is a list whose items may look like "GO:0001234,GO:0005678".
            raw_list = pdb2go[pid].get(ont_key, [])
            for go_str in raw_list:
                for go in go_str.split(","):
                    go = go.strip()
                    if go:
                        term_set.add(go)

        # Append even when term_set is empty so the result stays aligned with pid_list.
        go_lists.append(term_set)

    return go_lists


#########################################################
# Train one ontology (MF / BP / CC)
#########################################################
def train_one_ontology(
    ont_name,
    train_y, valid_y,
    idx_goid, goid_idx,
    train_graph, valid_graph,
    train_ppi_graph, valid_ppi_graph,
    train_interpro, valid_interpro,
    train_pid_list, valid_pid_list,
    device,
    resume=True  # 🔥 [New parameter] Whether to load an existing model and resume training
):

    print(f"\n==============================")
    print(f" ⭐ Starting ontology training: {ont_name}")
    print(f"==============================\n")

    label_num = train_y.shape[1]
    
    # 1. DataLoader: Change the batch size to 128
    BATCH_SIZE = 128
    train_data = [(train_graph[i], i, train_y[i]) for i in range(len(train_y))]
    valid_data = [(valid_graph[i], i, valid_y[i]) for i in range(len(valid_y))]

    train_dataloader = GraphDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = GraphDataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Model initialization
    model = combine_inter_model(
        inter_size=train_interpro.shape[1],
        inter_hid=1024,
        graph_size=1024,
        graph_hid=1024,
        label_num=label_num,
        head=4,
        ppi_in_dim=1024,
        ppi_hid_dim=256,
        ppi_out_dim=256,
        ppi_heads1=4,
        ppi_heads2=4,
        fused_dim=256,
        use_ppi=False,          # ❌ Disable PPI
        use_interpro=False, 
        fusion_type='concat',    # (This parameter has no effect here)
        use_transformer=False
    ).to(device)

    # 3. Optimizer and loss
    # Note: If the batch size increases to 128, consider a slightly higher initial LR, or keep 1e-3 depending on results.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    # 🔥 [New] Learning-rate scheduler: halve the LR if Fmax does not improve for 10 epochs
    # patience=10 must be less than the early-stopping value of 20 so LR can be adjusted before training stops.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    # 4. Resume logic
    best_fmax = -1.0
    best_epoch = -1
    start_epoch = 0
    save_path = f"{ont_name}_best_model_wo_all.pt"

    if resume and os.path.exists(save_path):
        print(f"🔄 Found saved model {save_path}; loading weights to resume training...")
        try:
            model.load_state_dict(torch.load(save_path))
            print("✅ Model weights loaded successfully!")
            
            # Important: Run validation first to determine the current Fmax.
            print("📊 Benchmarking the loaded model...")
            curr_fmax, _, _, _, _ = test_performance_gnn_inter(
                model, valid_dataloader, valid_pid_list, valid_interpro, valid_y,
                idx_goid, goid_idx, ont_name, device,
                loss_fn=loss_fn, valid_ppi_graph=valid_ppi_graph
            )
            best_fmax = curr_fmax
            print(f"🚀 Resuming training from this point: Current Best Fmax = {best_fmax:.4f}")
        except Exception as e:
            print(f"⚠️ Loading failed; training will restart from scratch. Error: {e}")
    else:
        print("🆕 No saved model found or resume is disabled; training from scratch.")

    # 5. Training parameters
    max_epochs = 150   # Increase the total number of epochs to allow enough time
    patience = 8     # 🔥 [Modified] Increase patience to 20
    no_improve = 0

    # 6. Training loop
    for epoch in range(start_epoch, max_epochs):
        print(f"\n[Epoch {epoch}] -------- Training (Batch={BATCH_SIZE}) --------")

        model.train()
        train_meter = AverageMeter()

        for batched_graph, sample_idx, labels in tqdm(train_dataloader, desc=f"{ont_name} Train"):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)

            idx_np = sample_idx.cpu().numpy()
            row = train_interpro[idx_np]
            inter_features = (
                torch.from_numpy(row.indices).long().to(device),
                torch.from_numpy(row.indptr).long().to(device),
                torch.from_numpy(row.data).float().to(device)
            )

            feats = batched_graph.ndata["x"]

            # PPI Batching
            ppi_graph_list = [train_ppi_graph[i] for i in idx_np]
            batched_ppi_graph = dgl.batch(ppi_graph_list).to(device)

            logits = model(inter_features, batched_graph, feats, batched_ppi_graph)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_meter.update(loss.item())

        print(f"[{ont_name}] Epoch {epoch} | Train Loss = {train_meter.avg:.4f}")

        # ========== Validation ==========
        print(f"[Epoch {epoch}] -------- Validating --------")
        
        fmax, aupr, cutoff, df, val_loss = test_performance_gnn_inter(
            model,
            valid_dataloader,
            valid_pid_list,
            valid_interpro,
            valid_y,
            idx_goid,
            goid_idx,
            ont_name,
            device,
            loss_fn=loss_fn,
            valid_ppi_graph=valid_ppi_graph 
        )

        print(f"🔍 [{ont_name}] Valid Fmax={fmax:.4f}, AUPR={aupr:.4f}, Loss={val_loss:.4f}")

        # 🔥 Update the scheduler
        # Pass the current Fmax to the scheduler so it can decide whether to reduce the learning rate.
        scheduler.step(fmax)

        # ===== Early stopping and saving =====
        if fmax > best_fmax:
            change = fmax - best_fmax
            best_fmax = fmax
            best_epoch = epoch
            no_improve = 0
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"📁 Created directory automatically: {save_dir}")
            torch.save(model.state_dict(), save_path)
            print(f"💾 [New Best] Improved by {change:.4f} -> Model saved to {save_path}")
        else:
            no_improve += 1
            print(f"⚠️ Fmax did not improve: {no_improve}/{patience} (Best: {best_fmax:.4f})")
            
            if no_improve >= patience:
                print(f"⏹ Early stopping triggered: Fmax did not improve for {patience} consecutive epochs.")
                print("💡 Tip: If loss is still decreasing while Fmax is unchanged, try tuning the focal-loss parameters.")
                break

    print(f"\n[{ont_name}] Training complete! Final best model: Epoch {best_epoch}, Fmax={best_fmax:.4f}\n")

##########################################################
# Main function (automatically iterates over MF/BP/CC)
##########################################################
def main():
    device = torch.device("cuda")

    # Mapping: ontology name -> key in pdb2go
    ont2key = {
        "mf": "molecular_function",
        "bp": "biological_process",
        "cc": "cellular_component",
    }

    for ont in ["mf", "bp", "cc"]:
        print(f"\n############################")
        print(f"### Starting ontology: {ont} ###")
        print(f"############################\n")

        # ===============================
        # 🔥 1. Reload the PID list for this ontology
        # ===============================
        train_pid_list = load_pid_from_seq_file(f"data_split/{ont}_train_ids.txt")
        valid_pid_list = load_pid_from_seq_file(f"data_split/{ont}_valid_ids.txt")

        print(f"[{ont}] Training PID count: {len(train_pid_list)}")
        print(f"[{ont}] Validation PID count: {len(valid_pid_list)}")

        # ===============================
        # 2. Load structure graphs
        # ===============================
        train_graph = load_graphs_for_pid_list(train_pid_list, "graphs")
        valid_graph = load_graphs_for_pid_list(valid_pid_list, "graphs")

        # ===============================
        # 3. Load PPI graphs
        # ===============================
        train_ppi_graph = load_ppi_graph_for_pid_list(train_pid_list, "ppi_graphs")
        valid_ppi_graph = load_ppi_graph_for_pid_list(valid_pid_list, "ppi_graphs")

        # ===============================
        # 4. Load InterPro features
        # ===============================
        train_interpro = pkl.load(open(f"interpro_feature_{ont}_train.pkl", "rb"))
        valid_interpro = pkl.load(open(f"interpro_feature_{ont}_valid.pkl", "rb"))

        assert train_interpro.shape[1] == valid_interpro.shape[1]

        # ===============================
        # 5. Build GO labels
        # ===============================
        pdb2go = load_pdb2go("pdb2go_propagate.json")
        ont_key = ont2key[ont]

        train_go = build_go_lists_for_ontology(train_pid_list, pdb2go, ont_key)
        valid_go = build_go_lists_for_ontology(valid_pid_list, pdb2go, ont_key)

        mlb = MultiLabelBinarizer(sparse_output=False)
        train_y = mlb.fit_transform(train_go).astype(np.float32)
        valid_y = mlb.transform(valid_go).astype(np.float32)

        idx_goid = {i: go for i, go in enumerate(mlb.classes_)}
        goid_idx = {go: i for i, go in idx_goid.items()}

        # ===============================
        # 6. Train the model for this ontology
        # ===============================
        train_one_ontology(
            ont,
            train_y, valid_y,
            idx_goid, goid_idx,
            train_graph, valid_graph,
            train_ppi_graph, valid_ppi_graph,
            train_interpro, valid_interpro,
            train_pid_list, valid_pid_list,
            device
        )


if __name__ == "__main__":
    main()
