# import json
# import os

# # ================= 配置 =================
# INPUT_JSON = "pdb2go_final_clean.json"  # 这里建议用你刚刚更新过、传播过的那个最好的版本
# OUTPUT_DIR = "dataset_splits"       # 输出文件夹
# # =======================================

# def main():
#     if not os.path.exists(INPUT_JSON):
#         print(f"❌ 错误：找不到 {INPUT_JSON}")
#         return

#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)

#     print(f"📖 正在读取 {INPUT_JSON} ...")
#     with open(INPUT_JSON, 'r') as f:
#         data = json.load(f)

#     print(f"🔹 原始总 ID 数: {len(data)}")

#     # 初始化三个列表
#     mf_ids = []
#     bp_ids = []
#     cc_ids = []

#     # 遍历数据，进行筛选
#     for pid, annotations in data.items():
#         # 1. 检查 Molecular Function
#         # 兼容两种格式：list of strings ["GO:1,GO:2"] 或 list of GOs ["GO:1", "GO:2"]
#         mf_list = annotations.get("molecular_function", [])
#         if mf_list and len(mf_list) > 0:
#             # 再次确认里面不是空字符串
#             if isinstance(mf_list[0], str) and len(mf_list[0].strip()) > 0:
#                 mf_ids.append(pid)
        
#         # 2. 检查 Biological Process
#         bp_list = annotations.get("biological_process", [])
#         if bp_list and len(bp_list) > 0:
#             if isinstance(bp_list[0], str) and len(bp_list[0].strip()) > 0:
#                 bp_ids.append(pid)

#         # 3. 检查 Cellular Component
#         cc_list = annotations.get("cellular_component", [])
#         if cc_list and len(cc_list) > 0:
#             if isinstance(cc_list[0], str) and len(cc_list[0].strip()) > 0:
#                 cc_ids.append(pid)

#     # 输出统计结果
#     print("\n📊 拆分统计结果:")
#     print(f"   🧬 Molecular Function (MF) 样本数: {len(mf_ids)}")
#     print(f"   🔄 Biological Process (BP) 样本数: {len(bp_ids)}")
#     print(f"   🏠 Cellular Component (CC) 样本数: {len(cc_ids)}")

#     # 保存 ID 列表
#     def save_ids(filename, id_list):
#         path = os.path.join(OUTPUT_DIR, filename)
#         with open(path, 'w') as f:
#             for pid in id_list:
#                 f.write(f"{pid}\n")
#         print(f"✅ 已保存: {path}")

#     save_ids("mf_ids_all.txt", mf_ids)
#     save_ids("bp_ids_all.txt", bp_ids)
#     save_ids("cc_ids_all.txt", cc_ids)

#     print("\n🎉 第一步完成！现在你有了三个纯净的 ID 列表，不再包含空标签样本。")
#     print("下一步我们将对这三个列表分别进行 训练/验证/测试 的划分。")

# if __name__ == "__main__":
#     main()


########_____________##################### 分割数据集
import os

# ================= 配置 =================
# 你的原始序列文件 (格式: ID \t Sequence 或 FASTA)
SOURCE_SEQ_FILE = "protein_id_and_sequence.txt" 
OUTPUT_DIR = "dataset_splits"
# =======================================

def load_sequences(seq_file):
    """读取 ID -> Sequence 映射"""
    seq_map = {}
    with open(seq_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # 确保 ID 格式一致 (比如都大写)
                pid = parts[0].strip()
                seq = parts[1].strip()
                seq_map[pid] = seq
    return seq_map

def generate_fasta_for_ontology(ont, seq_map):
    id_file = os.path.join(OUTPUT_DIR, f"{ont}_ids_all.txt")
    fasta_file = os.path.join(OUTPUT_DIR, f"{ont}_all.fasta")
    
    if not os.path.exists(id_file):
        print(f"⚠️ 跳过 {ont}，找不到 ID 列表。")
        return

    with open(id_file, 'r') as f_in, open(fasta_file, 'w') as f_out:
        count = 0
        for line in f_in:
            pid = line.strip()
            if pid in seq_map:
                # 写入 FASTA 格式
                f_out.write(f">{pid}\n{seq_map[pid]}\n")
                count += 1
            else:
                # 可能有的 ID 在序列文件里找不到，记录下来
                pass 
    print(f"✅ 生成 {fasta_file}: 包含 {count} 条序列")

def main():
    print("📖 加载原始序列库...")
    seq_map = load_sequences(SOURCE_SEQ_FILE)
    
    for ont in ['mf', 'bp', 'cc']:
        generate_fasta_for_ontology(ont, seq_map)

if __name__ == "__main__":
    main()