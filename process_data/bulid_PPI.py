import os
import torch
import pandas as pd
import dgl
from tqdm import tqdm
from collections import defaultdict

# ===== 1. 路径设置 =====
ppi_dir = "ppi_from_species_2hop_valid_100_10_supplement"   
embed_dir = [
    "extra_protT5_embeds_train",
    "extra_protT5_embeds_train_supplemem"
]            
save_dir = "ppi_graphs_supplemnt"                        
mapping_file = "pdb_uniprot_string_supplement.txt" 

os.makedirs(save_dir, exist_ok=True)

# ===== 2. 全局缓存 =====
# embed_cache 里混合了 "1MZH-A" (PDB) 和 "9606.ENSP..." (STRING) 两种 Key
embed_cache = {}       
string_to_pdbs = defaultdict(list) 

# ===== 3. 预加载 Embeddings =====
def preload_embeddings():
    print(f"🚀 开始加载 3个文件夹的 Embeddings...")
    
    total_files = 0
    
    for d in embed_dir:
        if not os.path.exists(d):
            print(f"⚠️ 警告: 路径不存在 {d}，跳过")
            continue
            
        files = [f for f in os.listdir(d) if f.endswith(".pt")]
        print(f"📂 正在加载 {d} ({len(files)} 个文件)...")
        
        for f in tqdm(files, desc="Loading", leave=False):
            node_id = os.path.splitext(f)[0] # 区分大小写 (WSL环境)
            
            # 避免重复加载 (如果不同文件夹有同名文件，优先保留先加载的)
            if node_id in embed_cache:
                continue
                
            try:
                path = os.path.join(d, f)
                emb = torch.load(path, map_location="cpu")
                if emb.ndim == 2: emb = emb.squeeze(0)
                embed_cache[node_id] = emb
            except: pass
            
        total_files += len(files)

    print(f"✅ 所有嵌入加载完成！内存中共有 {len(embed_cache)} 个唯一向量。")

# ===== 4. 加载映射表 =====
def load_mapping():
    print(f"⏳ [2/4] 正在加载映射表 {mapping_file} ...")
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                # 1. 长度检查 (防止空行或缺列)
                if len(parts) < 3: 
                    continue
                
                pdb_id = parts[0]   # 1MZH-A
                # string_id 是第三列
                string_id = parts[2] 

                # 2. 过滤 "NA" (字符串) 和其他无效值
                # 有些文件里可能写的是 "N/A", "-", "null" 等
                if string_id in ["NA", "N/A", "-", "nan", "None"]:
                    continue
                
                # 3. 存入字典
                string_to_pdbs[string_id].append(pdb_id)
                
    except Exception as e:
        print(f"❌ 映射表读取失败: {e}")

# ===== 5. 核心 ID 转换逻辑 (含兜底) =====
def get_best_node_id(raw_id, current_center_pdb):
    """
    raw_id: PPI文件里的 ID (通常是 STRING ID)
    current_center_pdb: 当前文件对应的 PDB ID
    """
    
    # 如果 raw_id 本身就在缓存里 (比如它已经是 PDB ID，或者是不需要转化的 STRING ID)
    # 先别急着返回，先看看能不能映射成 current_center_pdb (优先级最高)
    # 但通常 PPI 文件里全是 STRING ID，所以我们直接进映射逻辑
    
    if raw_id in string_to_pdbs:
        candidates = string_to_pdbs[raw_id] # 获取对应的 PDB 列表
        
        # 👑 优先级 1: 映射列表里正好包含“主角” (当前文件名)
        if current_center_pdb in candidates:
            return current_center_pdb
        
        # 🥈 优先级 2: 映射成其他有向量的 PDB
        for pdb in candidates:
            if pdb in embed_cache:
                return pdb
    
    # 🥉 优先级 3 (兜底): 
    # - 映射表里没这个 STRING ID
    # - 或者映射出来的 PDB 都没有向量
    # -> 那就直接返回原始的 STRING ID
    #    (后续构建特征时，会去 embed_cache 里查这个 STRING ID)
    return raw_id

# ===== 6. 构图逻辑 =====
def build_ppi_graph(ppi_file):
    filename = os.path.basename(ppi_file)
    center_pdb_id = os.path.splitext(filename)[0] 

    # 1. 读取文件
    try:
        sep = "\t" if not ppi_file.endswith(".csv") else ","
        df = pd.read_csv(ppi_file, sep=sep)
    except Exception as e:
        print(f"❌ [读取失败] {filename}: {e}")
        return None # 跳过损坏文件
    
    # 2. 重命名列 (适配 STRING 格式)
    rename_map = {
        "protein1": "PDB_A",
        "protein2": "PDB_B",
        "score": "score",
        "combined_score": "score" #以此类推
    }
    df = df.rename(columns=rename_map)

    # 3. 检查必要列是否存在
    if not {"PDB_A", "PDB_B", "score"}.issubset(df.columns):
        # 只有缺少关键列时才打印错误并跳过
        # print(f"❌ [列名错误] {filename} 只有列: {list(df.columns)}") 
        return None # 跳过格式不对的文件
    
    # 4. 清洗无效数据 (强制转数字，去空行)
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=["PDB_A", "PDB_B", "score"])

    # 5. 检查是否为空
    if df.empty: 
        # 这里的 return None 配合主程序的 if g，就会实现“跳过”
        # print(f"⚠️ [空数据] {filename} (已跳过)")
        return None 
    df["score"] = df["score"].clip(lower=0, upper=1000.0) / float(1000.0)
    # 获取所有涉及的原始 ID
    raw_nodes = set(df["PDB_A"]).union(set(df["PDB_B"]))
    
    # --- ID 转换 ---
    id_map = {} # Old -> New
    final_node_list = [center_pdb_id] # 强制 Index 0 为中心 PDB
    seen_final_ids = {center_pdb_id}

    for raw in raw_nodes:
        # 这里会执行我们的 3 级优先级逻辑
        best_id = get_best_node_id(raw, center_pdb_id)
        id_map[raw] = best_id
        
        if best_id not in seen_final_ids:
            final_node_list.append(best_id)
            seen_final_ids.add(best_id)

    node_to_idx = {n: i for i, n in enumerate(final_node_list)}
    num_nodes = len(final_node_list)

    # 6. 手动构建边列表 (核心修改：一次性搞定所有边)
    src_list = []
    dst_list = []
    weight_list = []

    # (A) 添加 PPI 边 (双向)
    for _, row in df.iterrows():
        # 获取转换后的 ID
        u_final = id_map.get(row["PDB_A"])
        v_final = id_map.get(row["PDB_B"])
        
        if u_final in node_to_idx and v_final in node_to_idx:
            u = node_to_idx[u_final]
            v = node_to_idx[v_final]
            w = float(row["score"])

            # 正向 u->v
            src_list.append(u)
            dst_list.append(v)
            weight_list.append(w)

            # 反向 v->u (防止自环重复添加)
            if u != v:
                src_list.append(v)
                dst_list.append(u)
                weight_list.append(w)

    # (B) 添加自环 (权重设为 1.0)
    for i in range(num_nodes):
        src_list.append(i)
        dst_list.append(i)
        weight_list.append(1.0) # 自环权重

    # 7. 构建 DGL 图
    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    w   = torch.tensor(weight_list, dtype=torch.float32)

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    
    # ✅ 这里赋值绝对安全，因为图结构已经固定了
    g.edata["weight"] = w 

    # --- 标记中心 ---
    mask = torch.zeros(len(final_node_list), dtype=torch.bool)
    mask[0] = True
    g.ndata["is_center"] = mask

    # --- 填充特征 ---
    feats = []
    missing = 0
    for node_id in final_node_list:
        # 这里就是见证兜底逻辑生效的地方：
        # 如果上面的 best_id 返回了 PDB ID，这里就查 PDB Embedding
        # 如果上面的 best_id 返回了 STRING ID，这里就查 STRING Embedding
        if node_id in embed_cache:
            feats.append(embed_cache[node_id])
        else:
            missing += 1
            feats.append(torch.zeros(1024)) # 实在没有就补零

    g.ndata["x"] = torch.stack(feats)
    
    return g

# ===== 7. 主程序 =====
if __name__ == "__main__":
    preload_embeddings()
    load_mapping()

    files = [f for f in os.listdir(ppi_dir) if f.endswith((".txt", ".tsv"))]
    print(f"🚀 [3/4] 开始构建 {len(files)} 个图...")

    success = 0
    for f in tqdm(files):
        g = build_ppi_graph(os.path.join(ppi_dir, f))
        if g:
            out = os.path.join(save_dir, os.path.splitext(f)[0] + ".bin")
            try:
                dgl.save_graphs(out, [g])
                success += 1
            except: pass

    print(f"🎉 [4/4] 全部完成！成功: {success}")




# import os
# import torch
# import pandas as pd
# import dgl
# from tqdm import tqdm
# from collections import defaultdict
#
# # ===== 1. 路径设置 =====
# ppi_dir = "ppi_from_species_2hop_100_10"
# embed_dir = [
#     "extra_protT5_embeds_train"
# ]
# save_dir = "ppi_graphs_big"
# mapping_file = "final_mapping_full_scan.tsv"
#
# # [新增] 指定只加载这些 ID 的嵌入
# target_id_file = "big_extracted_sequences_from_folder.txt"
#
# os.makedirs(save_dir, exist_ok=True)
#
# # ===== 2. 全局缓存 =====
# embed_cache = {}
# string_to_pdbs = defaultdict(list)
#
# # ===== [新增] 加载目标 ID 列表 =====
# def load_target_ids():
#     print(f"📋 正在读取目标 ID 列表: {target_id_file} ...")
#     valid_ids = set()
#     if not os.path.exists(target_id_file):
#         print(f"❌ 错误: 找不到文件 {target_id_file}，将无法加载任何嵌入！")
#         return valid_ids
#
#     with open(target_id_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             # 假设第一列是 ID (兼容 tab 或空格分隔)
#             parts = line.split()
#             if parts:
#                 valid_ids.add(parts[0])
#
#     print(f"✅ 目标 ID 加载完成，共有 {len(valid_ids)} 个唯一 ID 需要加载。")
#     return valid_ids
#
# # ===== 3. [修改] 按需加载 Embeddings =====
# def preload_embeddings(valid_ids):
#     print(f"🚀 开始根据 ID 列表加载 Embeddings...")
#
#     if not valid_ids:
#         print("⚠️ 警告: 目标 ID 列表为空，将不会加载任何嵌入！")
#         return
#
#     total_loaded = 0
#
#     for d in embed_dir:
#         if not os.path.exists(d):
#             print(f"⚠️ 警告: 路径不存在 {d}，跳过")
#             continue
#
#         # 获取文件夹下所有 .pt 文件
#         files = [f for f in os.listdir(d) if f.endswith(".pt")]
#         print(f"📂 正在扫描文件夹 {d} (共 {len(files)} 个文件)...")
#
#         # 使用 tqdm 显示进度
#         for f in tqdm(files, desc="Filtering & Loading", leave=False):
#             node_id = os.path.splitext(f)[0] # 去掉 .pt 后缀
#
#             # --- 核心修改：过滤逻辑 ---
#             # 只有当这个文件的文件名在我们的 valid_ids 列表里时，才加载
#             if node_id not in valid_ids:
#                 continue
#             # -----------------------
#
#             # 避免重复加载
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
#     print(f"✅ 嵌入加载完成！内存中实际加载了 {len(embed_cache)} 个向量 (命中率: {len(embed_cache)}/{len(valid_ids)})。")
#
# # ===== 4. 加载映射表 (保持不变) =====
# def load_mapping():
#     print(f"⏳ [2/4] 正在加载映射表 {mapping_file} ...")
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
#         print(f"❌ 映射表读取失败: {e}")
#
# # ===== 5. 核心 ID 转换逻辑 (保持不变) =====
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
# # ===== 6. 构图逻辑 (保持不变) =====
# def build_ppi_graph(ppi_file):
#     filename = os.path.basename(ppi_file)
#     center_pdb_id = os.path.splitext(filename)[0]
#
#     try:
#         sep = "\t" if not ppi_file.endswith(".csv") else ","
#         df = pd.read_csv(ppi_file, sep=sep)
#     except Exception as e:
#         print(f"❌ [读取失败] {filename}: {e}")
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
# # ===== 7. 主程序 =====
# if __name__ == "__main__":
#     # 1. 先加载 ID 列表
#     target_ids = load_target_ids()
#
#     # 2. 将 ID 列表传给加载函数
#     preload_embeddings(target_ids)
#
#     # 3. 继续后续流程
#     load_mapping()
#
#     files = [f for f in os.listdir(ppi_dir) if f.endswith((".txt", ".tsv"))]
#     print(f"🚀 [3/4] 开始构建 {len(files)} 个图...")
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
#     print(f"🎉 [4/4] 全部完成！成功: {success}")