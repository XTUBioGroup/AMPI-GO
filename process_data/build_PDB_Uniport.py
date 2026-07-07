import csv
import pickle



##建立pdb id-uniport映射表


# import csv
# import pickle
#
# # 输入文件 (解压后的 tsv)
# sifts_file = r"F:\QLDownload\pdb_chain_uniprot.tsv"
#
# # 输出映射表
# out_pkl = "pdb2uniprot.pkl"
# out_txt = "pdb2uniprot.txt"
#
# pdb2uniprot = {}
#
# with open(sifts_file, "r", encoding="utf-8") as f:
#     # 跳过以 # 开头的注释行
#     for line in f:
#         if line.startswith("#"):
#             continue
#         header = line.strip().split("\t")
#         break  # 读到真正表头就停
#
#     reader = csv.DictReader(f, delimiter="\t", fieldnames=header)
#
#     for row in reader:
#         pdb_id = row["PDB"]
#         chain_id = row["CHAIN"]
#         uniprot_ac = row["SP_PRIMARY"]
#
#         pdb_chain = f"{pdb_id.upper()}-{chain_id}"
#         pdb2uniprot[pdb_chain] = uniprot_ac
#
# print(f"[INFO] 映射数量: {len(pdb2uniprot)}")
#
# # 保存为 pickle
# with open(out_pkl, "wb") as fw:
#     pickle.dump(pdb2uniprot, fw)
#
# # 保存为 txt（每行: PDB-CHAIN \t UniProt AC）
# with open(out_txt, "w", encoding="utf-8") as fw:
#     for pdb_chain, uniprot_ac in pdb2uniprot.items():
#         fw.write(f"{pdb_chain}\t{uniprot_ac}\n")
#
# print(f"[INFO] 已保存映射表到 {out_pkl} 和 {out_txt}")
#
# # 测试
#
# # 测试
# test_id = "2V3B-A"
# if test_id in pdb2uniprot:
#     print(f"{test_id} → UniProt AC: {pdb2uniprot[test_id]}")
# else:
#     print(f"{test_id} 不在映射表里")
import requests

# ##查找缺失ID
# # 文件路径
# # 文件路径
train_file = "protein_id_and_sequence_train.txt"
valid_file = "protein_id_and_sequence_valid.txt"
mapping_file = "pdb2uniprot.txt"

# 读取已有的映射
pdb2uniprot = {}
with open(mapping_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            pdb2uniprot[parts[0]] = parts[1]

print(f"[INFO] 已加载映射数: {len(pdb2uniprot)}")

# 收集需要检查的 PDB IDs
def load_pdb_ids(file_path):
    pdb_ids = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            pdb_id = line.strip().split()[0]  # 取 4MID-A
            pdb_ids.append(pdb_id)
    return pdb_ids

all_ids = set(load_pdb_ids(train_file) + load_pdb_ids(valid_file))

# 分类
missing_ids = [pid for pid in all_ids if pid not in pdb2uniprot]
found_ids = [pid for pid in all_ids if pid in pdb2uniprot]

print(f"[INFO] 总共 {len(all_ids)} 个 ID，其中缺失 {len(missing_ids)} 个，已找到 {len(found_ids)} 个")

# 写缺失 ID
with open("missing_ids.txt", "w") as fw:
    for pid in missing_ids:
        fw.write(pid + "\n")
print(f"[INFO] 已将缺失的 {len(missing_ids)} 个 ID 写入 missing_ids.txt")

# 写已找到 ID
with open("found_ids.txt", "w") as fw:
    for pid in found_ids:
        fw.write(f"{pid}\t{pdb2uniprot[pid]}\n")
print(f"[INFO] 已将找到的 {len(found_ids)} 个 ID 写入 found_ids.txt")

