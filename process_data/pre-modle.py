# import torch
# from transformers import T5EncoderModel, T5Tokenizer
# import re
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
# # ================== 配置区 ==================
# MODEL_PATH = "huggingface_cache/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
# FASTA_FILE = "supplement_sequences.fasta"
# CORRUPTED_ID_FILE = "split_missing_ids.txt"
# OUTPUT_DIR = "supplement_embeddings_pt"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # ============================================


# def load_corrupted_ids(path):
#     with open(path) as f:
#         return set(line.strip() for line in f if line.strip())


# def read_fasta_selective(fasta_path, target_ids):
#     """
#     只读取 corrupted_ids.txt 中的蛋白
#     """
#     results = {}
#     current_id = None
#     current_seq = []

#     with open(fasta_path) as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             if line.startswith(">"):
#                 if current_id in target_ids:
#                     results[current_id] = "".join(current_seq)
#                 header = line[1:]
#                 current_id = header.split("|")[1] if "|" in header else header.split()[0]
#                 current_seq = []
#             else:
#                 current_seq.append(line)

#         if current_id in target_ids:
#             results[current_id] = "".join(current_seq)

#     return results


# def get_model():
#     tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, do_lower_case=False)
#     model = T5EncoderModel.from_pretrained(MODEL_PATH)
#     model.to(DEVICE)

#     if DEVICE.type == "cuda":
#         model.half()

#     model.eval()
#     return tokenizer, model


# def safe_save_embedding(tensor, path):
#     """
#     原子写入，防止写一半就中断 → corrupted pt
#     """
#     tmp = path + ".tmp"
#     torch.save(tensor, tmp)
#     os.replace(tmp, path)


# def run_inference(seq, tokenizer, model):
#     seq = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
#     ids = tokenizer(seq, return_tensors="pt")
#     input_ids = ids["input_ids"].to(DEVICE)
#     attention_mask = ids["attention_mask"].to(DEVICE)

#     with torch.no_grad():
#         out = model(input_ids=input_ids, attention_mask=attention_mask)

#     emb = out.last_hidden_state[0][:len(seq.replace(" ", "")), :].cpu()

#     if emb.dtype == torch.float32:
#         emb = emb.half()

#     return emb


# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     corrupted_ids = load_corrupted_ids(CORRUPTED_ID_FILE)
#     print(f"🔁 需要重算的蛋白数量: {len(corrupted_ids)}")

#     seq_dict = read_fasta_selective(FASTA_FILE, corrupted_ids)
#     print(f"📖 从 FASTA 中成功提取: {len(seq_dict)} 条序列")

#     tokenizer, model = get_model()

#     for i, (pid, seq) in enumerate(seq_dict.items(), 1):
#         try:
#             emb = run_inference(seq, tokenizer, model)
#             save_path = os.path.join(OUTPUT_DIR, f"{pid}.pt")
#             safe_save_embedding(emb, save_path)
#             print(f"[{i}/{len(seq_dict)}] ✅ 重算成功: {pid}", end="\r")

#         except RuntimeError as e:
#             print(f"\n❌ GPU 失败 {pid}: {e}")
#             torch.cuda.empty_cache()

#     print("\n🎉 所有 corrupted embedding 已处理完成")


# if __name__ == "__main__":
#     main()


##########蛋白质级序列嵌入

import os
import re
import math
import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

# ===== 用户配置区 =====

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

MODEL_PATH = r"huggingface_cache/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
SEQ_FILE = r"supplement_ids_seq.txt"
SAVE_DIR = r"extra_protT5_embeds_train_supplemem"

# 分块策略配置
MIN_CHUNK_SIZE = 3000   # 二分查找的下限
DEFAULT_OVERLAP = 1024   # 必须保证 chunk_size > 2 * overlap

# =================================================


def load_id_sequences(path):
    ids, seqs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ids.append(parts[0])
            seqs.append(parts[1])
    return ids, seqs


def preprocess_seq_for_t5(seq: str) -> str:
    """ProtT5 标准预处理"""
    seq = seq.strip().upper().replace(" ", "")
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


def run_model_forward(model, tokenizer, seq, device):
    """
    基础的前向传播函数，返回 hidden_states 和 attention_mask
    """
    proc = preprocess_seq_for_t5(seq)
    inputs = tokenizer([proc], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        hidden = outputs.last_hidden_state
        
    return hidden, attn_mask


def encode_whole_sequence(model, tokenizer, seq, device):
    """
    尝试整条编码。如果成功返回 embedding，失败抛出 OOM 异常。
    """
    hidden, attn_mask = run_model_forward(model, tokenizer, seq, device)
    
    # Masked Mean Pooling
    mask = attn_mask.unsqueeze(-1).expand_as(hidden).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    emb = (summed / counts).squeeze(0).cpu()
    
    return emb


def encode_with_sliding_window(model, tokenizer, seq, device, chunk_size, overlap=DEFAULT_OVERLAP):
    """
    当整条失败时，使用滑动窗口策略进行编码。
    策略：重叠切分，只取中间部分求和，最后除以总长。
    """
    L = len(seq)
    sum_embeds = torch.zeros(1024, dtype=torch.float32)
    
    # 步长 = 窗口大小 - 重叠
    stride = chunk_size - overlap
    if stride <= 0:
        stride = chunk_size // 2 # 保底防止死循环

    for i in range(0, L, stride):
        # 1. 确定当前窗口 [start : end]
        start = i
        end = min(i + chunk_size, L)
        
        # 如果是最后一块且完全包含在前一块的有效区内，可以跳过（简化处理，这里不跳过以保证覆盖）
        chunk_seq = seq[start:end]
        current_chunk_len = len(chunk_seq)
        
        # 2. 运行模型
        hidden, _ = run_model_forward(model, tokenizer, chunk_seq, device)
        token_embeds = hidden.squeeze(0).cpu() # [chunk_len, 1024]
        
        # 3. 截取有效区域（剔除 overlap 带来的边缘效应）
        # 规则：除了第一块和最后一块，中间的块都掐头去尾 overlap/2
        valid_start = 0
        valid_end = current_chunk_len
        
        if i > 0: # 不是第一块，去掉开头的 overlap/2
            valid_start = overlap // 2
        
        if end < L: # 不是最后一块，去掉结尾的 overlap/2
            valid_end = current_chunk_len - (overlap // 2)
        
        # 4. 累加有效部分的 Sum
        # 注意：这里我们做的是 Global Mean Pooling 的拆解版 -> Sum(所有token) / L
        # 所以只需要把所有 token 的 embedding 加起来即可
        valid_tokens = token_embeds[valid_start:valid_end, :]
        sum_embeds += valid_tokens.sum(dim=0)
        
        # 显存清理
        del hidden, token_embeds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 5. 计算全局平均
    final_emb = sum_embeds / L
    return final_emb


def find_max_chunk_size(model, tokenizer, seq, device):
    """
    二分查找：在 [MIN_CHUNK, Len-1] 之间找到最大的可行 chunk_size
    """
    L = len(seq)
    low = MIN_CHUNK_SIZE
    high = L - 1
    best_chunk = MIN_CHUNK_SIZE # 默认保底

    # 简单优化：如果上次找到的 chunk_size 能用，可以先试一下，这里省略
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # 试跑一下前 mid 个字符
            # 不需要跑完整个 encode 流程，只要 forward 不爆就行
            test_seq = seq[:mid]
            _ = run_model_forward(model, tokenizer, test_seq, device)
            
            # 成功了，说明 mid 可行，尝试更大的
            best_chunk = mid
            low = mid + 1
            
            # 清理
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                # 失败了，说明 mid 太大
                high = mid - 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise e # 其他错误直接抛出

    return best_chunk


def smart_encode(model, tokenizer, seq, device):
    """
    智能编码入口：
    1. 尝试整条 -> 成功返回
    2. 失败 -> 二分查找最大 Chunk -> 滑动窗口编码
    """
    L = len(seq)
    
    # --- 策略 1: 优先整条 ---
    try:
        return encode_whole_sequence(model, tokenizer, seq, device)
    
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise e # 非 OOM 错误直接报错
        
        # OOM 发生，清理显存
        print(f"  [Len={L}] 整条 OOM，触发分块策略...")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- 策略 2: 二分查找 + 滑动窗口 ---
    # 确定最大安全分块大小
    max_chunk = find_max_chunk_size(model, tokenizer, seq, device)
    print(f"  [Len={L}] 测得最大分块: {max_chunk} (Overlap={DEFAULT_OVERLAP})")
    
    # 使用该分块大小进行平滑编码
    return encode_with_sliding_window(model, tokenizer, seq, device, max_chunk, overlap=DEFAULT_OVERLAP)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠ 未检测到 CUDA，将使用 CPU")

    print("加载 ProtT5 模型...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, do_lower_case=False, local_files_only=True)
    model = T5EncoderModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    model.eval()

    ids, seqs = load_id_sequences(SEQ_FILE)
    print(f"任务总数: {len(ids)}")

    # 1. 过滤已完成的任务 (断点续传)
    todo_list = []
    for pid, seq in zip(ids, seqs):
        save_path = os.path.join(SAVE_DIR, f"{pid}.pt")
        if not os.path.exists(save_path):
            todo_list.append((pid, seq))
            
    print(f"已跳过 {len(ids) - len(todo_list)} 个，剩余 {len(todo_list)} 个待处理。")

    # 2. 处理剩余任务
    with torch.no_grad():
        for pid, seq in tqdm(todo_list, desc="Embedding"):
            seq = seq.strip()
            if not seq:
                continue

            save_path = os.path.join(SAVE_DIR, f"{pid}.pt")
            
            try:
                # 调用智能编码逻辑
                emb = smart_encode(model, tokenizer, seq, device)
                torch.save(emb, save_path)
            
            except Exception as e:
                print(f"\n❌ Error on {pid} (Len={len(seq)}): {e}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print("✅ 所有任务完成。")


if __name__ == "__main__":
    main()