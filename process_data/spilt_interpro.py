# import os
# from collections import defaultdict

# input_file = "all_ids_seq_final.fasta.tsv"
# output_dir = "interpro_big"
# col_idx = 0  

# os.makedirs(output_dir, exist_ok=True)

# # ==== 优化配置 ====
# # 如果你的内存很大（比如 > 16GB），可以设得更大，或者干脆一次性读完
# BATCH_SIZE = 200_000  # 每处理 20 万行才写入一次硬盘

# def flush_buffer(buffer_dict):
#     """将缓冲区的数据批量写入硬盘"""
#     print(f"   [IO] 正在写入 {len(buffer_dict)} 个蛋白质的数据...", end="\r")
#     for pid, lines in buffer_dict.items():
#         out_path = os.path.join(output_dir, f"{pid}.tsv")
#         # 使用 'a' 模式追加
#         with open(out_path, "a", encoding="utf-8") as fout:
#             fout.write("\n".join(lines) + "\n")
#     buffer_dict.clear() # 清空缓冲区

# def main():
#     buffer = defaultdict(list)
#     line_count = 0
#     total_written = 0

#     print(f"开始读取: {input_file} ...")
    
#     with open(input_file, "r", encoding="utf-8") as fin:
#         for line in fin:
#             line = line.rstrip("\n")
#             if not line: continue
            
#             parts = line.split("\t")
#             if len(parts) <= col_idx: continue
            
#             pid = parts[col_idx].strip()
#             if not pid: continue

#             # 1. 存入内存缓冲区，而不是直接写硬盘
#             buffer[pid].append(line)
#             line_count += 1

#             # 2. 如果积攒到了阈值，就批量写入一次
#             if line_count >= BATCH_SIZE:
#                 flush_buffer(buffer)
#                 total_written += line_count
#                 line_count = 0
    
#     # 3. 循环结束后，把剩下的也写入
#     if buffer:
#         flush_buffer(buffer)
#         total_written += line_count

#     print(f"\n[INFO] 拆分完成！共处理 {total_written} 行数据。")

# if __name__ == "__main__":
#     # 为了保险，先清空一下旧的 output_dir (或者手动删除)，否则 'a' 模式会一直追加
#     # import shutil
#     # if os.path.exists(output_dir): shutil.rmtree(output_dir)
#     # os.makedirs(output_dir, exist_ok=True)
    
#     main()

import os
from tqdm import tqdm

# ==================== 路径配置 ====================
# 刚才生成的终极 ID 列表文件
IDS_FILE = "all_ids_seq_final.txt"
# 存放单个蛋白质 TSV 文件的文件夹
TARGET_DIR = "interpro_big"

print("================ 📝 开始补全空白 InterPro 特征文件 ================")

# 1. 读取所有需要的蛋白质 ID
all_pids = []
if os.path.exists(IDS_FILE):
    with open(IDS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                all_pids.append(parts[0]) # 提取第一列 ID
    print(f"📜 总表中共有 {len(all_pids)} 个蛋白质 ID。")
else:
    print(f"❌ 找不到 ID 文件: {IDS_FILE}")
    exit()

# 2. 确保目标文件夹存在
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print(f"📁 已创建文件夹: {TARGET_DIR}")

# 3. 开始检查并创建空白文件
created_count = 0
exists_count = 0

for pid in tqdm(all_pids, desc="检查对齐情况"):
    tsv_filename = f"{pid}.tsv"
    tsv_path = os.path.join(TARGET_DIR, tsv_filename)
    
    # 如果文件不存在，就创建一个空白文件
    if not os.path.exists(tsv_path):
        try:
            # 使用 'w' 模式打开并立即关闭，即创建一个空文件
            with open(tsv_path, 'w', encoding='utf-8') as f:
                pass 
            created_count += 1
        except Exception as e:
            print(f"⚠️ 为 {pid} 创建空白文件失败: {e}")
    else:
        exists_count += 1

print("\n" + "="*50)
print(f"🎉 任务处理完成！")
print(f"✅ 已存在的有效文件: {exists_count} 个")
print(f"🆕 新创建的空白占位文件: {created_count} 个")
print(f"📦 文件夹内目前总计: {exists_count + created_count} 个文件")
print(f"这下所有的蛋白质都在 {TARGET_DIR} 里有对应的身份证了！")






