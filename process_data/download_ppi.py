# import os
# import time
# import requests
# import pandas as pd
# from io import StringIO
# from tqdm import tqdm

# # ================== 这里改成你的实际路径 ==================

# # 映射表：第1列 PDB CHAIN ID，第3列 STRING ID
# MAPPING_FILE = r"pdb_uniprot_string_valid.txt"

# # 输出目录：每个 PDB chain 一个文件，文件名类似 5CED-A.txt
# OUTPUT_DIR = r"F:\PythonProject1\ppi_2hop_per_pdb"

# # =======================================================

# API_BASE = "https://string-db.org/api/tsv/interaction_partners"
# HEADERS = {"User-Agent": "Mozilla/5.0"}

# # 为了稍微温柔一点，防止打爆 STRING，两个请求之间 sleep 一下
# SLEEP_SECONDS = 0.01  # 你嫌慢可以改小点，但不建议改成 0


# def load_pdb_string_mapping(mapping_file):
#     """
#     读取 pdb_uniprot_string_valid.txt
#     假设格式为：PDBCHAIN  UniProt  STRINGID
#     例如：
#       5CED-A  Q6MHT0  264462.Bd3459
#     返回：list[(pdb_chain_id, string_id)]
#     """
#     mapping = []
#     with open(mapping_file, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
#             parts = line.split("\t")
#             if len(parts) < 3:
#                 continue
#             pdb_chain_id = parts[0]
#             string_id = parts[2]
#             mapping.append((pdb_chain_id, string_id))
#     print(f"从映射表中读取到 {len(mapping)} 条 PDB-STRING 映射")
#     return mapping


# def fetch_interaction_partners(string_id):
#     """
#     使用 /api/tsv/interaction_partners 获取一个 STRING ID 的所有 partners。
#     返回 list[(a, b, score)]，其中 a=stringId_A, b=stringId_B。
#     """
#     # 从 STRING ID 中解析物种 ID，例如 9606.ENSPXXX -> 9606
#     species = None
#     if "." in string_id:
#         prefix = string_id.split(".")[0]
#         if prefix.isdigit():
#             species = int(prefix)

#     params = {
#         "identifiers": string_id,
#         # 如需要可以加过滤，如：
#         # "required_score": 400,
#         # "limit": 0,  # 部分版本中 0 表示不限制；具体看 STRING 文档
#     }
#     if species is not None:
#         params["species"] = species

#     try:
#         resp = requests.get(API_BASE, params=params, headers=HEADERS, timeout=60)
#         resp.raise_for_status()
#     except Exception as e:
#         print(f"[ERROR] interaction_partners 请求失败: STRING={string_id}, error={e}")
#         return []

#     text = resp.text.strip()
#     if not text:
#         return []

#     try:
#         df = pd.read_csv(StringIO(text), sep="\t")
#     except Exception as e:
#         print(f"[ERROR] 解析 TSV 失败 STRING={string_id}, error={e}")
#         return []

#     required_cols = ["stringId_A", "stringId_B", "score"]
#     if not all(col in df.columns for col in required_cols):
#         print(f"[WARN] STRING={string_id} 返回的列不包含 {required_cols}")
#         return []

#     edges = []
#     for _, row in df.iterrows():
#         a = str(row["stringId_A"])
#         b = str(row["stringId_B"])
#         s = float(row["score"])
#         edges.append((a, b, s))

#     return edges


# def build_2hop_for_one_pdb(pdb_chain_id, string_id):
#     """
#     为某一个 PDB chain 构建以 string_id 为中心的 2-hop 局部 PPI 网络：
#       1. 第一轮：center -> 1-hop 邻居
#       2. 第二轮：对所有 1-hop 邻居再扩一圈
#     返回：DataFrame(columns=['protein1','protein2','score'])
#     """
#     all_edges = set()   # 存无向边 (u, v, score)
#     neighbors = set()   # 第一轮得到的一跳邻居

#     # ---------- 第一轮：中心点的 1-hop ----------
#     edges1 = fetch_interaction_partners(string_id)
#     time.sleep(SLEEP_SECONDS)

#     for a, b, s in edges1:
#         # 无向去重：sort 之后存
#         u, v = sorted([a, b])
#         all_edges.add((u, v, s))

#         # 这个中心点的 “一跳邻居” 是与 string_id 相连的另一个节点
#         if a == string_id:
#             neighbors.add(b)
#         elif b == string_id:
#             neighbors.add(a)
#         else:
#             # 理论上 API 中 stringId_A 一般是输入的 ID，但为了保险，这里也可以都算邻居
#             neighbors.add(a)
#             neighbors.add(b)

#     # ---------- 第二轮：对所有一跳邻居再扩 ----------
#     for nei in neighbors:
#         edges2 = fetch_interaction_partners(nei)
#         time.sleep(SLEEP_SECONDS)
#         for a, b, s in edges2:
#             u, v = sorted([a, b])
#             all_edges.add((u, v, s))

#     # 转成 DataFrame，列名按你之前习惯：protein1 protein2 score
#     if not all_edges:
#         return pd.DataFrame(columns=["protein1", "protein2", "score"])

#     records = [{"protein1": u, "protein2": v, "score": s} for (u, v, s) in all_edges]
#     df = pd.DataFrame(records)
#     return df


# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     mapping = load_pdb_string_mapping(MAPPING_FILE)

#     for pdb_chain_id, string_id in tqdm(mapping, desc="构建每个 PDB 的 2-hop PPI"):
#         out_path = os.path.join(OUTPUT_DIR, f"{pdb_chain_id}.txt")

#         # 如果你想跳过已存在的结果，可以加一个判断：
#         if os.path.exists(out_path):
#          continue

#         print(f"\n[INFO] 处理 PDB={pdb_chain_id}, STRING={string_id}")
#         df = build_2hop_for_one_pdb(pdb_chain_id, string_id)

#         # 保存为 tab 分隔
#         df.to_csv(out_path, sep="\t", index=False)
#         print(f"[INFO] 保存 {pdb_chain_id} 的 2-hop PPI 到 {out_path}，边数={len(df)}")


# if __name__ == "__main__":
#     main()

# import os
# import requests
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ====================== 配置区 ======================

# # 输入文件：之前生成的缺失记录文件 (格式: PDB_CHAIN  STRING_ID  TAXID)
# SPECIES_FILE = "missing_species_files.txt"   

# OUTPUT_DIR = "string_species_links_v12"
# BASE_URL = "https://stringdb-downloads.org/download/protein.links.v12.0"

# MAX_WORKERS = 6    # 并发数
# TIMEOUT = 120       # 单个文件超时时间（秒）
# MAX_RETRIES = 3     # 失败重试次数

# FAILED_LOG = "failed_taxids_retry.txt"    # 失败记录文件

# # ===================================================


# def load_taxids_from_third_column(species_file):
#     """
#     读取 missing_species_files.txt 的第三列 (TAXID)
#     并去重，只保留唯一的 taxid 用于下载
#     """
#     taxids = set()
#     try:
#         with open(species_file, "r") as f:
#             header = next(f, None) # 跳过标题行 (PDB_CHAIN STRING_ID TAXID)
            
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 parts = line.split("\t")
#                 if len(parts) >= 3:
#                     # 获取第三列，并去除可能存在的 .0 后缀 (例如 9606.0 -> 9606)
#                     taxid = parts[2].split(".")[0] 
#                     taxids.add(taxid)
                    
#         print(f"✅ 从文件读取并去重后，共需下载 {len(taxids)} 个物种文件")
#         return list(taxids)
        
#     except FileNotFoundError:
#         print(f"❌ 找不到文件: {species_file}")
#         return []


# def download_one(taxid):
#     url = f"{BASE_URL}/{taxid}.protein.links.v12.0.txt.gz"
#     out_path = os.path.join(OUTPUT_DIR, f"{taxid}.protein.links.v12.0.txt.gz")
#     temp_path = out_path + ".tmp"

#     # ✅ 1. 检查已存在文件
#     if os.path.exists(out_path):
#         # 简单检查：如果文件大小 > 1KB 认为有效，跳过 (STRING文件通常都很大)
#         if os.path.getsize(out_path) > 1024:
#             return ("SKIP", taxid, "file exists")
#         else:
#             # 如果是空文件或极小，可能是上次下载失败残留，删除重下
#             try:
#                 os.remove(out_path)
#             except:
#                 pass

#     # ✅ 2. 尝试下载 (带重试机制)
#     for attempt in range(MAX_RETRIES):
#         try:
#             with requests.get(url, stream=True, timeout=TIMEOUT) as r:
#                 if r.status_code == 404:
#                     return ("FAIL", taxid, "HTTP 404 Not Found (TaxID可能错误)")
#                 if r.status_code != 200:
#                     raise Exception(f"HTTP {r.status_code}")

#                 # 写入临时文件，防止中断导致文件损坏
#                 with open(temp_path, "wb") as f:
#                     for chunk in r.iter_content(chunk_size=1024 * 1024):
#                         if chunk:
#                             f.write(chunk)
            
#             # 下载完成，重命名
#             os.replace(temp_path, out_path)
#             return ("OK", taxid, "downloaded")

#         except Exception as e:
#             if attempt < MAX_RETRIES - 1:
#                 continue # 重试
#             else:
#                 # 删除可能残留的临时文件
#                 if os.path.exists(temp_path):
#                     try:
#                         os.remove(temp_path)
#                     except:
#                         pass
#                 return ("ERROR", taxid, str(e))

#     return ("ERROR", taxid, "Unknown error")


# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # 读取 TAXID
#     taxids = load_taxids_from_third_column(SPECIES_FILE)
    
#     if not taxids:
#         print("没有需要下载的任务。")
#         return

#     failed = []

#     print(f"🚀 开始并发下载 (Workers={MAX_WORKERS})...")

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(download_one, taxid): taxid for taxid in taxids}

#         for future in as_completed(futures):
#             status, taxid, msg = future.result()

#             if status == "OK":
#                 print(f"[OK]   {taxid}")
#             elif status == "SKIP":
#                 print(f"[SKIP] {taxid}")
#             else:
#                 print(f"[FAIL] {taxid} -> {msg}")
#                 failed.append(taxid)

#     # 写入失败日志
#     if failed:
#         with open(FAILED_LOG, "w") as f:
#             for t in failed:
#                 f.write(f"{t}\n")
#         print(f"\n⚠️ 共有 {len(failed)} 个文件下载失败，已记录到: {FAILED_LOG}")
#         print("建议：检查网络连接后，再次运行此脚本即可重试下载。")
#     else:
#         print("\n✅ 所有需要的物种文件均已就绪！")


# if __name__ == "__main__":
#     main()






import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ===================== 配置 =====================

SPECIES_DIR = r"string_species_links_v12"     # 每个物种的 PPI 文件所在目录
MAPPING_FILE = r"pdb_uniprot_string_supplement.txt"
OUTPUT_DIR = r"ppi_from_species_2hop_valid_100_10_supplement" 

MISSING_LOG = "missing_species_files.txt"

# ✅ 核心配置：双层不同的 Top-K
TOP_K_1HOP = 100  # 第一跳：每个 Seed 取 100 个最强邻居
TOP_K_2HOP = 10   # 第二跳：每个邻居取 10 个最强邻居

# ===============================================


def load_mapping_grouped_by_taxid(mapping_file):
    """
    读取 PDB-STRING 映射表，按 TaxID 分组
    """
    taxid_to_string_to_pdbs = defaultdict(lambda: defaultdict(list))
    records_all = []

    print(f"📖 正在加载映射文件: {mapping_file} ...")
    with open(mapping_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            pdb_chain = parts[0]
            string_id = parts[2]
            
            if string_id == "NA": continue
            if "." not in string_id: continue

            taxid = string_id.split(".")[0]

            taxid_to_string_to_pdbs[taxid][string_id].append(pdb_chain)
            records_all.append((pdb_chain, string_id, taxid))

    print(f"✅ 映射总条数: {len(records_all)}")
    print(f"✅ 涉及物种数: {len(taxid_to_string_to_pdbs)}")
    return taxid_to_string_to_pdbs, records_all


def build_1hop_2hop_for_species_mixed_k(df, seeds_this_species, k1=100, k2=10):
    """
    混合 Top-K 构建逻辑：
    1. 1-hop 使用 k1
    2. 2-hop 使用 k2
    """
    edges_1hop = defaultdict(list)
    edges_2hop = defaultdict(list)
    valid_neighbors_map = defaultdict(set)

    # -------------------------------------------------------
    # 1. 处理 1-hop (Seed -> Neighbor, Top K1)
    # -------------------------------------------------------
    # 筛选与 Seed 相关的边
    mask1 = df["protein1"].isin(seeds_this_species) | df["protein2"].isin(seeds_this_species)
    df_1hop_all = df[mask1]

    # 双向扩展：确保 Seed 在 protein1 位置，方便分组
    s1 = df_1hop_all[df_1hop_all["protein1"].isin(seeds_this_species)].copy()
    s1.columns = ["seed", "neighbor", "score"]
    
    s2 = df_1hop_all[df_1hop_all["protein2"].isin(seeds_this_species)].copy()
    s2.columns = ["neighbor", "seed", "score"]
    s2 = s2[["seed", "neighbor", "score"]] # 重排顺序
    
    df_1hop_unified = pd.concat([s1, s2], ignore_index=True)
    
    # 按 seed 分组，取 score 最高的 K1 个
    df_1hop_topk = (
        df_1hop_unified.sort_values(["seed", "score"], ascending=[True, False])
        .groupby("seed")
        .head(k1)
    )

    # 记录结果
    for row in df_1hop_topk.itertuples(index=False):
        # 排除自环（虽然 STRING 一般没有自环，但为了保险）
        if row.seed == row.neighbor: continue
        
        edges_1hop[row.seed].append((row.seed, row.neighbor, row.score))
        valid_neighbors_map[row.neighbor].add(row.seed) # 记录谁连到了这个邻居

    all_valid_neighbors = set(valid_neighbors_map.keys())
    
    if not all_valid_neighbors:
        return edges_1hop, edges_2hop

    # -------------------------------------------------------
    # 2. 处理 2-hop (Neighbor -> Next_Neighbor, Top K2)
    # -------------------------------------------------------
    
    # 筛选与 1-hop 邻居相关的边
    mask2 = df["protein1"].isin(all_valid_neighbors) | df["protein2"].isin(all_valid_neighbors)
    df_2hop_all = df[mask2]
    
    n1 = df_2hop_all[df_2hop_all["protein1"].isin(all_valid_neighbors)].copy()
    n1.columns = ["neighbor", "next_neighbor", "score"]
    
    n2 = df_2hop_all[df_2hop_all["protein2"].isin(all_valid_neighbors)].copy()
    n2.columns = ["next_neighbor", "neighbor", "score"]
    n2 = n2[["neighbor", "next_neighbor", "score"]]
    
    df_2hop_unified = pd.concat([n1, n2], ignore_index=True)
    
    # 按 Neighbor 分组，取 score 最高的 K2 个
    df_2hop_topk = (
        df_2hop_unified.sort_values(["neighbor", "score"], ascending=[True, False])
        .groupby("neighbor")
        .head(k2)
    )
    
    # 将 2-hop 边回溯给原始的 Seed
    for row in df_2hop_topk.itertuples(index=False):
        neighbor = row.neighbor
        next_neighbor = row.next_neighbor
        
        # 找到所有连接到这个 neighbor 的原始 seed
        parent_seeds = valid_neighbors_map.get(neighbor, set())
        
        for seed in parent_seeds:
            # 防止回溯（A->B->A）
            if next_neighbor == seed:
                continue
            # 这里记录的是 (B, C, score)，归属于 Seed A 的子图
            edges_2hop[seed].append((neighbor, next_neighbor, row.score))

    return edges_1hop, edges_2hop


def write_pdb_files_with_resume(edges_1hop, edges_2hop, string_to_pdbs, outdir):
    """
    写入文件
    """
    os.makedirs(outdir, exist_ok=True)
    written_count = 0

    for seed, pdb_list in string_to_pdbs.items():
        # 检查该 seed 下的所有 PDB 文件是否都已存在
        # 如果都存在，直接跳过计算和写入
        targets = [os.path.join(outdir, f"{p}.txt") for p in pdb_list]
        if all(os.path.exists(t) for t in targets):
            continue

        # --- 合并数据 ---
        rec = []
        rec.extend(edges_1hop.get(seed, []))
        rec.extend(edges_2hop.get(seed, []))

        if not rec:
            df_seed = pd.DataFrame(columns=["protein1", "protein2", "score"])
        else:
            df_seed = pd.DataFrame(rec, columns=["p1_temp", "p2_temp", "score"])
            
            # 排序逻辑：保证无向图边的一致性 (min, max)
            a = df_seed["p1_temp"]
            b = df_seed["p2_temp"]
            df_seed["protein1"] = list(map(min, zip(a, b)))
            df_seed["protein2"] = list(map(max, zip(a, b)))
            
            # 去重：保留分数最高的
            df_seed = (
                df_seed[["protein1", "protein2", "score"]]
                .sort_values(["protein1", "protein2", "score"], ascending=[True, True, False])
                .drop_duplicates(subset=["protein1", "protein2"], keep="first")
            )

        # 逐个写入 PDB 文件
        for pdb_chain in pdb_list:
            out_path = os.path.join(outdir, f"{pdb_chain}.txt")
            # 再次检查，防止多线程或其他情况（虽然这里是单线程）
            if os.path.exists(out_path):
                continue
            
            df_seed.to_csv(out_path, sep="\t", index=False)
            written_count += 1
            
    return written_count


def main():
    taxid_to_string_to_pdbs, records_all = load_mapping_grouped_by_taxid(MAPPING_FILE)
    missing_lines = []

    # 直接遍历所有 TaxID，进度条显示
    # 使用 list() 包装 keys，确保顺序固定
    all_taxids = list(taxid_to_string_to_pdbs.keys())
    
    print(f"🚀 开始处理 {len(all_taxids)} 个物种的任务...")
    
    pbar = tqdm(all_taxids, desc="Processing Species")
    
    for taxid in pbar:
        string_to_pdbs = taxid_to_string_to_pdbs[taxid]
        
        # ----------------------------------------------------
        # ⭐ 优化的断点续传检查 ⭐
        # 在加载巨大 CSV 之前，检查该物种下 *所有* PDB 是否都已经有结果了
        # 如果全部都存在，直接 continue，不再读取 CSV，秒过
        # ----------------------------------------------------
        all_done = True
        # 抽样检查：如果物种PDB很多，检查所有文件可能会有IO耗时，
        # 但相比读几十GB CSV，这点IO耗时是可以接受的。
        for pdb_list in string_to_pdbs.values():
            for p in pdb_list:
                if not os.path.exists(os.path.join(OUTPUT_DIR, f"{p}.txt")):
                    all_done = False
                    break
            if not all_done: break
        
        if all_done:
            # pbar.write(f"[{taxid}] 跳过 (已完成)")
            continue
        # ----------------------------------------------------

        pbar.set_postfix({"TaxID": taxid, "Status": "Init"})
        species_file = os.path.join(SPECIES_DIR, f"{taxid}.protein.links.v12.0.txt.gz")

        if not os.path.exists(species_file):
            # pbar.write(f"⚠️ 缺失物种文件: {taxid}")
            for string_id, pdb_list in string_to_pdbs.items():
                for pdb_chain in pdb_list:
                    missing_lines.append(f"{pdb_chain}\t{string_id}\t{taxid}")
            continue

        try:
            # Step 1: 读取 CSV (C引擎加速 + 内存优化)
            pbar.set_postfix({"TaxID": taxid, "Status": "Reading CSV"})
            
            df = pd.read_csv(
                species_file,
                sep=" ",  # STRING v12 是空格分隔
                compression="gzip",
                usecols=["protein1", "protein2", "combined_score"],
                dtype={"protein1": "string", "protein2": "string", "combined_score": "int32"}
            )
            df = df.rename(columns={"combined_score": "score"})
            
            # Step 2: 计算 Top-K 子图
            pbar.set_postfix({"TaxID": taxid, "Status": "Graph Build"})
            seeds_this_species = set(string_to_pdbs.keys())
            
            edges_1hop, edges_2hop = build_1hop_2hop_for_species_mixed_k(
                df, seeds_this_species, k1=TOP_K_1HOP, k2=TOP_K_2HOP
            )

            # Step 3: 写入文件
            pbar.set_postfix({"TaxID": taxid, "Status": "Writing"})
            count = write_pdb_files_with_resume(edges_1hop, edges_2hop, string_to_pdbs, OUTPUT_DIR)
            
            # pbar.write(f"[{taxid}] 生成了 {count} 个文件")

        except Exception as e:
            pbar.write(f"❌ Error processing species {taxid}: {e}")
            continue

    if missing_lines:
        with open(MISSING_LOG, "w") as f:
            for line in missing_lines:
                f.write(line + "\n")
        print(f"\n⚠️ 缺失记录已写入: {MISSING_LOG}")
    else:
        print("\n✅ 所有任务完成！")

if __name__ == "__main__":
    main()




