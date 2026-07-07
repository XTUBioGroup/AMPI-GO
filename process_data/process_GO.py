# import json
# import os
# from collections import defaultdict
# from tqdm import tqdm

# # ================= 配置 =================
# OLD_JSON_FILE = "process_data/pdb2go.json"           # 提供 Key
# TSV_FILE = "F:\download\pdb_chain_go.tsv\pdb_chain_go.tsv"           # 提供 Value (GO数据)
# OBO_FILE = "process_data/go.obo"                     # 提供结构 (分类 + 传播)
# OUTPUT_FILE = "pdb2go_updated.json"     # 最终结果
# # =======================================

# def load_obo_structure(obo_path):
#     """
#     解析 OBO: 获取 namespace 和 父子关系
#     """
#     print(f"📖 解析 OBO: {obo_path} ...")
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
#     递归获取祖先节点 (传播的核心)
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
#         print("❌ 缺少必要文件 (go.obo 或 pdb2go.json)")
#         return

#     # 1. 解析 OBO
#     parents_map, ns_map = load_obo_structure(OBO_FILE)

#     # 2. 读取旧 JSON，只为了获取 Keys (白名单)
#     print(f"📖 读取旧 JSON 键值: {OLD_JSON_FILE} ...")
#     with open(OLD_JSON_FILE, 'r', encoding='utf-8') as f:
#         old_data = json.load(f)
    
#     # 建立 Key 的白名单集合，方便快速查找
#     # 假设 key 是 "155C-A" 这种格式
#     valid_keys = set(old_data.keys())
#     print(f"✅ 锁定 {len(valid_keys)} 个目标 PDB-Chain。")

#     # 3. 预处理 TSV 数据到内存
#     # 结构: tsv_data["155C-A"] = {"GO:001", "GO:002", ...} (混合 namespace)
#     tsv_data = defaultdict(set)
    
#     print(f"📖 正在扫描 TSV: {TSV_FILE} ...")
#     with open(TSV_FILE, 'r', encoding='utf-8') as f:
#         for line in tqdm(f, desc="Scanning TSV"):
#             if line.startswith("#") or not line.strip(): continue
#             parts = line.strip().split('\t')
#             if len(parts) <= 5: continue
            
#             # 构造 Key: 你的格式是大写 PDB + "-" + Chain
#             pdb = parts[0].upper()
#             chain = parts[1]
#             key = f"{pdb}-{chain}"
            
#             # 🔥 关键点：只提取在旧 JSON 里出现过的 Key
#             if key in valid_keys:
#                 go_id = parts[5]
#                 if go_id in ns_map: # 只处理有效的 GO
#                     tsv_data[key].add(go_id)

#     # 4. 构建新字典 + 传播 (Propagation)
#     print("🚀 开始构建新字典并执行父节点传播...")
#     final_dict = {}
#     ancestor_memo = {}
    
#     for key in tqdm(valid_keys, desc="Propagating"):
#         # 初始化空结构 (即使 TSV 里没有数据，也要保留 Key，防止代码报错)
#         entry = {
#             "molecular_function": set(),
#             "biological_process": set(),
#             "cellular_component": set()
#         }
        
#         # 获取 TSV 里该 Key 对应的叶子 GO
#         leaf_gos = tsv_data.get(key, set())
        
#         # 遍历每一个叶子 GO，做传播
#         all_gos = set(leaf_gos) # 包含自己
#         for go_id in leaf_gos:
#             ancestors = get_ancestors(go_id, parents_map, ancestor_memo)
#             all_gos.update(ancestors)
            
#         # 将传播后的所有 GO 分类填入 entry
#         for go_id in all_gos:
#             ns = ns_map.get(go_id)
#             if ns:
#                 entry[ns].add(go_id)
        
#         # 格式化保存：转为 list
#         final_dict[key] = {
#             "molecular_function": list(entry["molecular_function"]),
#             "biological_process": list(entry["biological_process"]),
#             "cellular_component": list(entry["cellular_component"])
#         }
        
#         # 如果你想兼容旧的字符串列表格式 ["GO:1,GO:2"]，请解开下面注释
#         # for ns_key in final_dict[key]:
#         #     if final_dict[key][ns_key]:
#         #         merged_str = ",".join(final_dict[key][ns_key])
#         #         final_dict[key][ns_key] = [merged_str]

#     # 5. 保存
#     print(f"💾 保存至 {OUTPUT_FILE} ...")
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         json.dump(final_dict, f, indent=2)
        
#     print(f"🎉 大功告成！")
#     print(f"保留 Key 数量: {len(final_dict)}")
#     print("数据已全部替换为 TSV 最新版，且已包含所有父节点。")

# if __name__ == "__main__":
#     main()


import json
import os

# ================= 配置路径 =================
JSON_FILE = "pdb2go_updated.json"        # 待清洗的 JSON
TRAIN_FILE = "protein_id_and_sequence_train.txt" # 训练集 ID 列表
VALID_FILE = "protein_id_and_sequence_valid.txt" # 验证集 ID 列表
OUTPUT_FILE = "pdb2go_final_clean.json"  # 输出文件
# ===========================================

def load_ids_from_file(filepath):
    """
    读取 ID 列表文件。
    假设文件格式是每行一个 ID，或者 "ID SEQUENCE" 格式。
    通常如果是 ID+Seq，第一列是 ID。
    """
    ids = set()
    if not os.path.exists(filepath):
        print(f"⚠️ 警告: 找不到文件 {filepath}，跳过。")
        return ids
        
    print(f"📖 读取 ID 列表: {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 根据你的文件格式调整这里
            # 如果是纯 ID 列表:
            # pid = line
            
            # 如果是 "ID 序列" (空格或Tab分隔):
            parts = line.split()
            pid = parts[0]
            
            # 去除可能存在的 ">" 符号 (fasta格式常见)
            if pid.startswith(">"):
                pid = pid[1:]
                
            ids.add(pid)
    
    print(f"   └─ 找到 {len(ids)} 个唯一 ID。")
    return ids

def main():
    # 1. 构建白名单 (Whitelist)
    train_ids = load_ids_from_file(TRAIN_FILE)
    valid_ids = load_ids_from_file(VALID_FILE)
    
    valid_keys = train_ids.union(valid_ids)
    
    print(f"✅ 白名单构建完成。总共允许 {len(valid_keys)} 个蛋白质 ID。")
    
    # 2. 读取 JSON
    if not os.path.exists(JSON_FILE):
        print(f"❌ 错误: 找不到 {JSON_FILE}")
        return

    print(f"📖 读取 JSON 文件: {JSON_FILE} ...")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    initial_count = len(data)
    print(f"   └─ 原始数据包含 {initial_count} 个条目。")
    
    # 3. 执行清洗
    print("🧹 开始清洗...")
    cleaned_data = {}
    removed_count = 0
    
    for pid, content in data.items():
        # 检查 Key 是否在白名单里
        # 注意：有时候 JSON Key 可能是 "155C-A"，而文件里是 "155C_A" 或者 "155C"，需要确认格式一致性
        # 这里假设是完全匹配
        if pid in valid_keys:
            cleaned_data[pid] = content
        else:
            removed_count += 1
            
    # 4. 保存
    print(f"💾 正在保存清洗后的文件至 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2)
        
    print(f"🎉 完成！")
    print(f"   原始数量: {initial_count}")
    print(f"   删除数量: {removed_count}")
    print(f"   保留数量: {len(cleaned_data)}")
    print(f"   结果文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()