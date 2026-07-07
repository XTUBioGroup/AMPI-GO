

# #匹配pdb ID


# import os
# import glob
# import pandas as pd

# # ===================== 配置区 =====================

# # 1) 输入：包含 PPI 文件的两个文件夹路径
# folder1 = r"ppi_from_species_2hop_train_100_10"
# folder2 = r"ppi_from_species_2hop_valid_100_10"

# # 2) 输入：你自己的数据集 ID 文件（用来排除已有的序列）
# #    格式：至少三列，第三列是 STRING ID，可能为 NA
# dataset_id_file = r"pdb_uniprot_string.txt"

# # 3) 输入：STRING 的全量序列文件 (FASTA格式)
# fasta_file = r"F:\download\protein.sequences.v12.0.fa\protein.sequences.v12.0.fa"

# # 4) 输出：补充的序列文件路径
# output_txt = r"extra_protein_sequences_combined.txt"

# # =================================================


# def collect_ppi_proteins_from_folders(folders):
#     """
#     遍历指定文件夹列表，读取所有 .txt 文件中的 protein1 和 protein2
#     """
#     proteins = set()
#     total_files = 0

#     for folder in folders:
#         # 构造搜索路径：folder/*.txt
#         search_pattern = os.path.join(folder, "*.txt")
#         files = glob.glob(search_pattern)
        
#         print(f"\n📂 正在扫描文件夹: {folder}")
#         print(f"   找到 {len(files)} 个 txt 文件")
        
#         total_files += len(files)

#         # 逐个读取
#         for i, path in enumerate(files):
#             # 简单的进度打印，每100个文件打印一次
#             if (i + 1) % 500 == 0:
#                 print(f"   正在处理第 {i + 1} 个文件...")
            
#             try:
#                 # 只读取前两列，节省内存
#                 # STRING 格式通常是 tab 分隔
#                 df = pd.read_csv(path, sep="\t", usecols=["protein1", "protein2"])
                
#                 # 将两列转为集合并更新
#                 proteins.update(df["protein1"].astype(str))
#                 proteins.update(df["protein2"].astype(str))
                
#             except ValueError:
#                 # 可能是空文件或列名不对
#                 pass
#             except Exception as e:
#                 print(f"   ❌ 读取失败: {path}, 错误: {e}")

#     print(f"\n📊 统计:")
#     print(f"   总共处理文件数: {total_files}")
#     print(f"   PPI 中包含的唯一蛋白数量: {len(proteins)}")
#     return proteins


# def load_dataset_string_ids(filepath):
#     """
#     读取 pdb_uniprot_string.txt 的第三列
#     """
#     ids = set()
#     if not os.path.exists(filepath):
#         print(f"❌ 找不到数据集文件: {filepath}")
#         return ids

#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
            
#             parts = line.split()  # 默认空白符分割
#             if len(parts) < 3:
#                 continue
            
#             string_id = parts[2]
#             if string_id != "NA":
#                 ids.add(string_id)

#     print(f"📚 原始数据集中的 STRING ID 数量: {len(ids)}")
#     return ids


# def extract_sequences_from_fasta(fasta_path, wanted_ids):
#     """
#     流式读取 FASTA，只提取需要的 ID
#     """
#     seqs = {}
#     current_id = None
#     current_seq_lines = []

#     print(f"\n🧬 正在读取 FASTA 库: {fasta_path}")
#     print("   这可能需要几分钟，请稍候...")

#     def flush():
#         nonlocal current_id, current_seq_lines
#         if current_id is not None and current_id in wanted_ids:
#             seqs[current_id] = "".join(current_seq_lines)
#         current_id = None
#         current_seq_lines = []

#     try:
#         with open(fasta_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.rstrip("\n")
#                 if not line:
#                     continue

#                 if line.startswith(">"):
#                     # 保存上一条记录
#                     flush()
#                     # 解析新 ID: >9606.ENSP00000123456 描述...
#                     header = line[1:]
#                     # 取第一个空格前的部分作为 ID
#                     header_id = header.split()[0]
#                     current_id = header_id
#                 else:
#                     current_seq_lines.append(line)
            
#             # 处理最后一条
#             flush()

#     except FileNotFoundError:
#         print(f"❌ 找不到 FASTA 文件: {fasta_path}")
#         return {}

#     print(f"✅ 成功提取序列数量: {len(seqs)}")
#     missing = len(wanted_ids) - len(seqs)
#     if missing > 0:
#         print(f"⚠️  警告: 有 {missing} 个蛋白在 FASTA 中未找到（可能是旧版ID或物种库缺失）")
    
#     return seqs


# def main():
#     # 1. 收集两个文件夹下的所有 PPI 蛋白
#     #    你可以把 folder1, folder2 放入列表
#     target_folders = [folder1, folder2]
#     ppi_ids = collect_ppi_proteins_from_folders(target_folders)

#     if not ppi_ids:
#         print("❌ 未找到任何 PPI 蛋白，程序结束。")
#         return

#     # 2. 读取已有数据集的 ID
#     dataset_ids = load_dataset_string_ids(dataset_id_file)

#     # 3. 计算差集：需要补充的 ID = PPI出现的 - 数据集已有的
#     extra_ids = ppi_ids - dataset_ids
#     print(f"\n🔍 需要补充序列的蛋白数量 (差集): {len(extra_ids)}")

#     if not extra_ids:
#         print("✅ 所有 PPI 蛋白均已存在于数据集中，无需补充。")
#         return

#     # 4. 从 FASTA 提取序列
#     seq_dict = extract_sequences_from_fasta(fasta_file, extra_ids)

#     # 5. 保存结果
#     print(f"\n💾 正在保存到: {output_txt}")
#     with open(output_txt, "w", encoding="utf-8") as out:
#         # 写入表头（可选）
#         # out.write("string_id\tsequence\n") 
#         for pid, seq in seq_dict.items():
#             out.write(f"{pid}\t{seq}\n")

#     print("🎉 全部完成！")


# if __name__ == "__main__":
#     main()



import os
import glob
import pandas as pd

# ===================== 配置区 =====================

# 1) 输入：PPI 文件夹路径 (只填一个)
ppi_folder = r"ppi_from_species_2hop_valid_100_10_supplement"

# 2) 输入：已有的序列文件 (用于排除)
#    格式：第一列是 STRING ID (tab 分隔)
existing_seq_file = r"extra_protein_sequences_combined.txt"

# 3) 输入：数据集 ID 文件 (继续排除 dataset 里本来就有的)
#    (如果你确认 extra_protein_sequences_combined.txt 已经包含了 dataset 之外的所有补充，
#     这一步其实是双重保险，建议保留)
dataset_id_file = r"pdb_uniprot_string_supplement.txt"

# 4) 输入：STRING 全量序列库
fasta_file = r"F:\download\protein.sequences.v12.0.fa\protein.sequences.v12.0.fa"

# 5) 输出：新的增量补充文件
output_txt = r"extra_protein_sequences_combined_supplement.txt"

# =================================================


def collect_ppi_proteins_from_folder(folder):
    """
    扫描单个文件夹下的所有 .txt 文件，提取 protein1 和 protein2
    """
    proteins = set()
    search_pattern = os.path.join(folder, "*.txt")
    files = glob.glob(search_pattern)
    
    print(f"\n📂 正在扫描文件夹: {folder}")
    print(f"   找到 {len(files)} 个 txt 文件")

    for i, path in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"   已处理 {i + 1} 个文件...", end="\r")
        
        try:
            # 只读前两列
            df = pd.read_csv(path, sep="\t", usecols=["protein1", "protein2"])
            proteins.update(df["protein1"].astype(str))
            proteins.update(df["protein2"].astype(str))
        except Exception:
            pass # 忽略空文件或错误文件

    print(f"\n✅ 文件夹内 PPI 涉及的唯一蛋白总数: {len(proteins)}")
    return proteins


def load_existing_ids(filepath):
    """
    读取已有的序列文件第一列 (extra_protein_sequences_combined.txt)
    """
    ids = set()
    if not os.path.exists(filepath):
        print(f"⚠️ 文件不存在，将不进行排除: {filepath}")
        return ids

    print(f"📖 正在加载已有序列列表: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # 假设格式: ID \t Sequence
            parts = line.split("\t")
            ids.add(parts[0])
    
    print(f"   -> 已排除 {len(ids)} 个已知序列")
    return ids


def load_dataset_string_ids(filepath):
    """
    读取 pdb_uniprot_string.txt 的第三列
    """
    ids = set()
    if not os.path.exists(filepath):
        return ids

    print(f"📖 正在加载原始数据集 ID: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) >= 3:
                string_id = parts[2]
                if string_id != "NA":
                    ids.add(string_id)

    print(f"   -> 已排除 {len(ids)} 个原始数据集 ID")
    return ids


def extract_sequences_from_fasta(fasta_path, wanted_ids):
    """
    流式读取 FASTA，只提取 wanted_ids
    """
    seqs = {}
    current_id = None
    current_seq_lines = []
    
    # 转为 set 加速查找
    wanted_set = set(wanted_ids)

    print(f"\n🧬 正在扫描 FASTA 库提取 {len(wanted_set)} 条序列...")
    
    def flush():
        nonlocal current_id, current_seq_lines
        if current_id and current_id in wanted_set:
            seqs[current_id] = "".join(current_seq_lines)
        current_id = None
        current_seq_lines = []

    try:
        with open(fasta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line: continue

                if line.startswith(">"):
                    flush()
                    # 解析 ID: >9606.ENSP... 描述
                    header = line[1:]
                    current_id = header.split()[0]
                else:
                    current_seq_lines.append(line)
            flush() # 最后一条

    except FileNotFoundError:
        print(f"❌ 错误: 找不到 FASTA 文件 {fasta_path}")
        return {}

    print(f"✅ 成功提取: {len(seqs)} 条")
    return seqs


def main():
    # 1. 获取 PPI 里的所有 ID
    ppi_ids = collect_ppi_proteins_from_folder(ppi_folder)
    if not ppi_ids:
        return

    # 2. 获取需要排除的 ID (Combined文件 + 原始Dataset)
    existing_combined_ids = load_existing_ids(existing_seq_file)
    original_dataset_ids = load_dataset_string_ids(dataset_id_file)
    
    # 合并所有已知的 ID
    all_known_ids = existing_combined_ids.union(original_dataset_ids)

    # 3. 计算真正的增量 (Diff)
    ids_to_fetch = ppi_ids - all_known_ids
    
    print(f"\n📊 统计摘要:")
    print(f"   PPI 总 ID 数:       {len(ppi_ids)}")
    print(f"   已有文件 ID 数:     {len(existing_combined_ids)}")
    print(f"   原始数据 ID 数:     {len(original_dataset_ids)}")
    print(f"   ---------------------------")
    print(f"   🔥 真正需补充 ID 数: {len(ids_to_fetch)}")

    if not ids_to_fetch:
        print("\n🎉 没有新的 ID 需要补充，所有序列都已存在！")
        return

    # 4. 提取序列
    new_seqs = extract_sequences_from_fasta(fasta_file, ids_to_fetch)

    # 5. 保存补充文件
    if new_seqs:
        print(f"\n💾 正在保存增量文件: {output_txt}")
        with open(output_txt, "w", encoding="utf-8") as f:
            for pid, seq in new_seqs.items():
                f.write(f"{pid}\t{seq}\n")
        print("🎉 完成！")
    else:
        print("⚠️ 未能在 FASTA 中找到任何目标 ID 的序列。")

if __name__ == "__main__":
    main()