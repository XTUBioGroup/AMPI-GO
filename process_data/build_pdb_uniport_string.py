
import os
from tqdm import tqdm

# 输入文件
found_ids_file = "found_ids.txt"  # PDB Chain -> UniProt AC
aliases_file = r"F:\QLDownload\protein.aliases.v12.0.txt"  # STRING alias 文件
output_file = "pdb_uniprot_string.txt"
missing_file = "missing_uniprot.txt"

# 先统计文件行数，用于进度条
with open(aliases_file, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

# 读取 UniProt AC -> STRING ID 映射
uniprot2string = {}
with open(aliases_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="Loading UniProt→STRING"):
        if line.startswith("#"):  # 跳过注释行
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            string_id, alias, source = parts[0], parts[1], parts[2]
            if source == "UniProt_AC":  # 只用 UniProt_AC 作为映射
                uniprot2string[alias] = string_id

print(f"[INFO] 已加载 UniProt→STRING 映射数: {len(uniprot2string)}")

# 读取 found_ids 并生成三列
missing_count = 0
with open(found_ids_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout, \
     open(missing_file, "w", encoding="utf-8") as fmiss:

    for line in tqdm(fin, desc="Mapping PDB→UniProt→STRING"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        pdb_chain, uniprot_ac = parts[0], parts[1]
        string_id = uniprot2string.get(uniprot_ac, "NA")
        fout.write(f"{pdb_chain}\t{uniprot_ac}\t{string_id}\n")
        if string_id == "NA":
            missing_count += 1
            fmiss.write(f"{pdb_chain}\t{uniprot_ac}\n")

print(f"[INFO] 已生成 {output_file}")
print(f"[INFO] 未找到 STRING ID 的 UniProt 数量: {missing_count}")
print(f"[INFO] 未找到的 UniProt 已写入 {missing_file}")
