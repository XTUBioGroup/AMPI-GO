# import os
# from collections import defaultdict

# input_file = "all_ids_seq_final.fasta.tsv"
# output_dir = "interpro_big"
# col_idx = 0  

# os.makedirs(output_dir, exist_ok=True)

# # ==== Optimization settings ====
# # If ample memory is available (for example, >16 GB), increase this value or read everything at once
# BATCH_SIZE = 200_000  # Write to disk after every 200,000 processed rows

# def flush_buffer(buffer_dict):
#     """Write buffered data to disk in batches."""
#     print(f"   [IO] Writing data for {len(buffer_dict)} proteins...", end="\r")
#     for pid, lines in buffer_dict.items():
#         out_path = os.path.join(output_dir, f"{pid}.tsv")
#         # Append using 'a' mode
#         with open(out_path, "a", encoding="utf-8") as fout:
#             fout.write("\n".join(lines) + "\n")
#     buffer_dict.clear() # Clear the buffer

# def main():
#     buffer = defaultdict(list)
#     line_count = 0
#     total_written = 0

#     print(f"Starting to read: {input_file} ...")
    
#     with open(input_file, "r", encoding="utf-8") as fin:
#         for line in fin:
#             line = line.rstrip("\n")
#             if not line: continue
            
#             parts = line.split("\t")
#             if len(parts) <= col_idx: continue
            
#             pid = parts[col_idx].strip()
#             if not pid: continue

#             # 1. Store in the memory buffer instead of writing directly to disk
#             buffer[pid].append(line)
#             line_count += 1

#             # 2. Write a batch when the threshold is reached
#             if line_count >= BATCH_SIZE:
#                 flush_buffer(buffer)
#                 total_written += line_count
#                 line_count = 0
    
#     # 3. Write any remaining data after the loop
#     if buffer:
#         flush_buffer(buffer)
#         total_written += line_count

#     print(f"\n[INFO] Splitting complete! Processed {total_written} rows.")

# if __name__ == "__main__":
#     # For safety, clear the old output_dir first (or delete it manually), or 'a' mode will keep appending
#     # import shutil
#     # if os.path.exists(output_dir): shutil.rmtree(output_dir)
#     # os.makedirs(output_dir, exist_ok=True)
    
#     main()

import os
from tqdm import tqdm

# ==================== Path configuration ====================
# Final ID list generated above
IDS_FILE = "all_ids_seq_final.txt"
# Directory containing individual protein TSV files
TARGET_DIR = "interpro_big"

print("================ 📝 Starting creation of missing empty InterPro feature files ================")

# 1. Read all required protein IDs
all_pids = []
if os.path.exists(IDS_FILE):
    with open(IDS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                all_pids.append(parts[0]) # Extract the ID from the first column
    print(f"📜 The master list contains {len(all_pids)} protein IDs.")
else:
    print(f"❌ ID file not found: {IDS_FILE}")
    exit()

# 2. Ensure the target directory exists
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print(f"📁 Created directory: {TARGET_DIR}")

# 3. Check and create empty files
created_count = 0
exists_count = 0

for pid in tqdm(all_pids, desc="Checking alignment"):
    tsv_filename = f"{pid}.tsv"
    tsv_path = os.path.join(TARGET_DIR, tsv_filename)
    
    # Create an empty file if it does not exist
    if not os.path.exists(tsv_path):
        try:
            # Open in 'w' mode and close immediately to create an empty file
            with open(tsv_path, 'w', encoding='utf-8') as f:
                pass 
            created_count += 1
        except Exception as e:
            print(f"⚠️ Failed to create an empty file for {pid}: {e}")
    else:
        exists_count += 1

print("\n" + "="*50)
print(f"🎉 Task complete!")
print(f"✅ Existing valid files: {exists_count}")
print(f"🆕 Newly created empty placeholder files: {created_count}")
print(f"📦 Current total files in the directory: {exists_count + created_count}")
print(f"Every protein now has a corresponding file in {TARGET_DIR}!")






