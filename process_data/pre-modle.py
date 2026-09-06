# import torch
# from transformers import T5EncoderModel, T5Tokenizer
# import re
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
# # ================== Configuration ==================
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
#     Read only proteins listed in corrupted_ids.txt
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
#     Write atomically to prevent interruption from producing a corrupted .pt file
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
#     print(f"🔁 Proteins requiring recomputation: {len(corrupted_ids)}")

#     seq_dict = read_fasta_selective(FASTA_FILE, corrupted_ids)
#     print(f"📖 Successfully extracted {len(seq_dict)} sequences from FASTA")

#     tokenizer, model = get_model()

#     for i, (pid, seq) in enumerate(seq_dict.items(), 1):
#         try:
#             emb = run_inference(seq, tokenizer, model)
#             save_path = os.path.join(OUTPUT_DIR, f"{pid}.pt")
#             safe_save_embedding(emb, save_path)
#             print(f"[{i}/{len(seq_dict)}] ✅ Recomputed successfully: {pid}", end="\r")

#         except RuntimeError as e:
#             print(f"\n❌ GPU failure for {pid}: {e}")
#             torch.cuda.empty_cache()

#     print("\n🎉 All corrupted embeddings have been processed")


# if __name__ == "__main__":
#     main()


########## Protein-level sequence embeddings

import os
import re
import math
import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

# ===== User configuration =====

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

MODEL_PATH = r"huggingface_cache/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
SEQ_FILE = r"supplement_ids_seq.txt"
SAVE_DIR = r"extra_protT5_embeds_train_supplemem"

# Chunking-strategy configuration
MIN_CHUNK_SIZE = 3000   # Lower bound for binary search
DEFAULT_OVERLAP = 1024   # Must ensure chunk_size > 2 * overlap

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
    """Standard ProtT5 preprocessing."""
    seq = seq.strip().upper().replace(" ", "")
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


def run_model_forward(model, tokenizer, seq, device):
    """
    Basic forward-pass function that returns hidden_states and attention_mask.
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
    Attempt to encode the full sequence. Return the embedding on success; raise an OOM exception on failure.
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
    Use a sliding-window strategy when full-sequence encoding fails.
    Strategy: Split with overlap, sum only the middle sections, then divide by the total length.
    """
    L = len(seq)
    sum_embeds = torch.zeros(1024, dtype=torch.float32)
    
    # Stride = window size - overlap
    stride = chunk_size - overlap
    if stride <= 0:
        stride = chunk_size // 2 # Fallback to prevent an infinite loop

    for i in range(0, L, stride):
        # 1. Determine the current window [start : end]
        start = i
        end = min(i + chunk_size, L)
        
        # The final chunk could be skipped if fully covered by the previous valid region; keep it here to ensure coverage
        chunk_seq = seq[start:end]
        current_chunk_len = len(chunk_seq)
        
        # 2. Run the model
        hidden, _ = run_model_forward(model, tokenizer, chunk_seq, device)
        token_embeds = hidden.squeeze(0).cpu() # [chunk_len, 1024]
        
        # 3. Extract the valid region (remove boundary effects introduced by overlap)
        # Rule: Trim overlap/2 from both ends of intermediate chunks, excluding the first and last chunks
        valid_start = 0
        valid_end = current_chunk_len
        
        if i > 0: # Not the first chunk: remove overlap/2 from the start
            valid_start = overlap // 2
        
        if end < L: # Not the last chunk: remove overlap/2 from the end
            valid_end = current_chunk_len - (overlap // 2)
        
        # 4. Accumulate the sum of valid regions
        # This decomposes global mean pooling as Sum(all tokens) / L
        # Therefore, simply add the embeddings of all tokens
        valid_tokens = token_embeds[valid_start:valid_end, :]
        sum_embeds += valid_tokens.sum(dim=0)
        
        # Release GPU memory
        del hidden, token_embeds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 5. Compute the global mean
    final_emb = sum_embeds / L
    return final_emb


def find_max_chunk_size(model, tokenizer, seq, device):
    """
    Binary search for the largest feasible chunk_size in [MIN_CHUNK, Len-1].
    """
    L = len(seq)
    low = MIN_CHUNK_SIZE
    high = L - 1
    best_chunk = MIN_CHUNK_SIZE # Default fallback

    # Simple optimization: Try the last successful chunk_size first; omitted here
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # Test the first mid characters
            # A full encode is unnecessary; only verify that the forward pass does not fail
            test_seq = seq[:mid]
            _ = run_model_forward(model, tokenizer, test_seq, device)
            
            # Success means mid is feasible; try a larger value
            best_chunk = mid
            low = mid + 1
            
            # Clean up
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Failure means mid is too large
                high = mid - 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise e # Re-raise other errors directly

    return best_chunk


def smart_encode(model, tokenizer, seq, device):
    """
    Adaptive encoding entry point:
    1. Try the full sequence -> return on success
    2. On failure -> binary-search for the largest chunk -> encode with a sliding window
    """
    L = len(seq)
    
    # --- Strategy 1: Prefer the full sequence ---
    try:
        return encode_whole_sequence(model, tokenizer, seq, device)
    
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise e # Re-raise non-OOM errors directly
        
        # OOM occurred; release GPU memory
        print(f"  [Len={L}] Full-sequence OOM; activating the chunking strategy...")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Strategy 2: Binary search + sliding window ---
    # Determine the largest safe chunk size
    max_chunk = find_max_chunk_size(model, tokenizer, seq, device)
    print(f"  [Len={L}] Largest tested chunk: {max_chunk} (Overlap={DEFAULT_OVERLAP})")
    
    # Perform smooth encoding with this chunk size
    return encode_with_sliding_window(model, tokenizer, seq, device, max_chunk, overlap=DEFAULT_OVERLAP)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠ CUDA not detected; using CPU")

    print("Loading the ProtT5 model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, do_lower_case=False, local_files_only=True)
    model = T5EncoderModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    model.eval()

    ids, seqs = load_id_sequences(SEQ_FILE)
    print(f"Total tasks: {len(ids)}")

    # 1. Filter completed tasks (resume support)
    todo_list = []
    for pid, seq in zip(ids, seqs):
        save_path = os.path.join(SAVE_DIR, f"{pid}.pt")
        if not os.path.exists(save_path):
            todo_list.append((pid, seq))
            
    print(f"Skipped {len(ids) - len(todo_list)}; {len(todo_list)} remain.")

    # 2. Process remaining tasks
    with torch.no_grad():
        for pid, seq in tqdm(todo_list, desc="Embedding"):
            seq = seq.strip()
            if not seq:
                continue

            save_path = os.path.join(SAVE_DIR, f"{pid}.pt")
            
            try:
                # Invoke adaptive encoding logic
                emb = smart_encode(model, tokenizer, seq, device)
                torch.save(emb, save_path)
            
            except Exception as e:
                print(f"\n❌ Error on {pid} (Len={len(seq)}): {e}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print("✅ All tasks complete.")


if __name__ == "__main__":
    main()
