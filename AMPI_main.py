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
# 数据加载工具函数
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
    专门加载 PPI 图：
    ⚠️ 不要动自环和边权，保持构图脚本里的信息
    """
    graphs = []
    for pid in tqdm(pid_list, desc=f"Loading PPI graphs from {folder}"):
        bin_path = os.path.join(folder, pid + ".bin")
        g = dgl.load_graphs(bin_path)[0][0]
        graphs.append(g)
    return graphs

def load_pdb2go(path):
    """读取你之前生成的 pdb2go.json"""
    with open(path, "r") as f:
        return json.load(f)


def build_go_lists_for_ontology(pid_list, pdb2go, ont_key):
    """
    根据 pid_list 和 pdb2go.json，构造 DPFunc 风格的 GO 列表（list[set[go_id]]）

    ont_key 取值：'molecular_function' / 'biological_process' / 'cellular_component'
    """
    go_lists = []

    for pid in pid_list:
        term_set = set()

        if pid in pdb2go:
            # 例如 pdb2go[pid]["molecular_function"] 是一个 list，每个元素可能是 "GO:0001234,GO:0005678"
            raw_list = pdb2go[pid].get(ont_key, [])
            for go_str in raw_list:
                for go in go_str.split(","):
                    go = go.strip()
                    if go:
                        term_set.add(go)

        # 即使 term_set 为空，也照样 append，保证长度和 pid_list 对齐
        go_lists.append(term_set)

    return go_lists


#########################################################
# 训练一个本体（MF / BP / CC）
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
    resume=True  # 🔥 [新增参数] 是否尝试加载旧模型续练
):

    print(f"\n==============================")
    print(f" ⭐ 开始训练本体：{ont_name}")
    print(f"==============================\n")

    label_num = train_y.shape[1]
    
    # 1. DataLoader: Batch Size 改为 128
    BATCH_SIZE = 128
    train_data = [(train_graph[i], i, train_y[i]) for i in range(len(train_y))]
    valid_data = [(valid_graph[i], i, valid_y[i]) for i in range(len(valid_y))]

    train_dataloader = GraphDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = GraphDataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 模型初始化
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
        use_ppi=False,          # ❌ 关掉 PPI
        use_interpro=False, 
        fusion_type='concat',    # (此时这个参数无效)
        use_transformer=False
    ).to(device)

    # 3. 优化器与 Loss
    # 注意：如果 Batch Size 变大 (128)，建议初始 LR 稍微给大一点点，或者保持 1e-3 看情况
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    # 🔥 [新增] 学习率调度器：如果 10 轮 Fmax 不涨，LR 减半
    # patience=10 必须小于早停的 20，这样才有机会在停止前调整 LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    # 4. 续练逻辑 (Resume Logic)
    best_fmax = -1.0
    best_epoch = -1
    start_epoch = 0
    save_path = f"{ont_name}_best_model_wo_all.pt"

    if resume and os.path.exists(save_path):
        print(f"🔄 发现已保存的模型 {save_path}，正在加载权重以续练...")
        try:
            model.load_state_dict(torch.load(save_path))
            print("✅ 模型权重加载成功！")
            
            # 关键：先跑一遍验证集，看看当前的 Fmax 是多少
            print("📊 正在基准测试加载的模型性能...")
            curr_fmax, _, _, _, _ = test_performance_gnn_inter(
                model, valid_dataloader, valid_pid_list, valid_interpro, valid_y,
                idx_goid, goid_idx, ont_name, device,
                loss_fn=loss_fn, valid_ppi_graph=valid_ppi_graph
            )
            best_fmax = curr_fmax
            print(f"🚀以此为起点继续训练: Current Best Fmax = {best_fmax:.4f}")
        except Exception as e:
            print(f"⚠️ 加载失败，将重新开始训练。错误: {e}")
    else:
        print("🆕 未发现旧模型或未启用 Resume，从头开始训练。")

    # 5. 训练参数设置
    max_epochs = 150   # 总轮次增加，给足时间
    patience = 8     # 🔥 [修改] 耐心值增加到 20
    no_improve = 0

    # 6. 训练循环
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

        # ========== 验证 ==========
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

        # 🔥 更新 Scheduler
        # 告诉调度器当前的 Fmax，让它决定是否要降学习率
        scheduler.step(fmax)

        # ===== 早停 & 保存 =====
        if fmax > best_fmax:
            change = fmax - best_fmax
            best_fmax = fmax
            best_epoch = epoch
            no_improve = 0
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"📁 已自动创建文件夹: {save_dir}")
            torch.save(model.state_dict(), save_path)
            print(f"💾 [New Best] 提升 {change:.4f} -> 模型已保存至 {save_path}")
        else:
            no_improve += 1
            print(f"⚠️ Fmax 未提升: {no_improve}/{patience} (Best: {best_fmax:.4f})")
            
            if no_improve >= patience:
                print(f"⏹ 早停触发：Fmax 已连续 {patience} 轮未提升。")
                print("💡 提示：如果发现 Loss 还在降但 Fmax 不动，可以尝试微调 Focal Loss 参数。")
                break

    print(f"\n[{ont_name}] 训练结束！最终最佳模型 Epoch {best_epoch}, Fmax={best_fmax:.4f}\n")

##########################################################
# 主函数（自动循环 MF/BP/CC）
##########################################################
def main():
    device = torch.device("cuda")

    # 映射：ontology 名称 -> pdb2go 中的 key
    ont2key = {
        "mf": "molecular_function",
        "bp": "biological_process",
        "cc": "cellular_component",
    }

    for ont in ["mf", "bp", "cc"]:
        print(f"\n############################")
        print(f"### 开始处理本体：{ont} ###")
        print(f"############################\n")

        # ===============================
        # 🔥 1. 按本体重新加载 PID 列表
        # ===============================
        train_pid_list = load_pid_from_seq_file(f"data_split/{ont}_train_ids.txt")
        valid_pid_list = load_pid_from_seq_file(f"data_split/{ont}_valid_ids.txt")

        print(f"[{ont}] Train PID 数: {len(train_pid_list)}")
        print(f"[{ont}] Valid PID 数: {len(valid_pid_list)}")

        # ===============================
        # 2. 加载结构图
        # ===============================
        train_graph = load_graphs_for_pid_list(train_pid_list, "graphs")
        valid_graph = load_graphs_for_pid_list(valid_pid_list, "graphs")

        # ===============================
        # 3. 加载 PPI 图
        # ===============================
        train_ppi_graph = load_ppi_graph_for_pid_list(train_pid_list, "ppi_graphs")
        valid_ppi_graph = load_ppi_graph_for_pid_list(valid_pid_list, "ppi_graphs")

        # ===============================
        # 4. 加载 InterPro 特征
        # ===============================
        train_interpro = pkl.load(open(f"interpro_feature_{ont}_train.pkl", "rb"))
        valid_interpro = pkl.load(open(f"interpro_feature_{ont}_valid.pkl", "rb"))

        assert train_interpro.shape[1] == valid_interpro.shape[1]

        # ===============================
        # 5. 构造 GO 标签
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
        # 6. 训练该本体模型
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
