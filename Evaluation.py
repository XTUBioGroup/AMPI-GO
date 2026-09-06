import warnings
import numpy as np
import scipy.sparse as ssp
import torch
import numpy as np
import pandas as pd


__all__ = ['fmax', 'ROOT_GO_TERMS', 'compute_performance_deepgoplus', 'read_pkl', 'save_pkl']
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}


def fmax(targets, scores):
    """Compute the Fmax metric."""
    targets = ssp.csr_matrix(targets)
    
    fmax_ = 0.0, 0.0
    precisions = []
    recalls = []
    
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
        
        if np.isnan(p):
            precisions.append(0.0)
            recalls.append(r)
            continue
        else: 
            precisions.append(p)
            recalls.append(r)
        
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
        except ZeroDivisionError: 
            pass
    
    return fmax_[0], fmax_[1], precisions, recalls


import pandas as pd
from collections import OrderedDict,deque,Counter
from sklearn.metrics import average_precision_score, roc_curve, auc, matthews_corrcoef, precision_recall_curve
import math
import re
import pickle as pkl
from tqdm.auto import tqdm, trange

def read_pkl(pklfile):
    with open(pklfile,'rb') as fr:
        data=pkl.load(fr)
    return data

def save_pkl(pklfile, data):
    with open(pklfile,'wb') as fw:
        pkl.dump(data, fw)


BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',])
#    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])
CAFA_TARGETS = set([
    '10090', '223283', '273057', '559292', '85962',
    '10116',  '224308', '284812', '7227', '9606',
    '160488', '237561', '321314', '7955', '99287',
    '170187', '243232', '3702', '83333', '208963',
    '243273', '44689', '8355'])

def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES

class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def calculate_ic(self, annots):
        # print(annots[:10])
        # print(type(annots[0]))
        # sys.exit(0)
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        if it[0] == 'part_of':
                            obj['is_a'].append(it[1])
                            
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
        if obj is not None:
            ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont


    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set


import numpy as np

def new_compute_performance_deepgoplus(
    test_df, 
    go_file, 
    ont, 
    idx_goid,
    goid_idx,
    with_relations=True
):
    go = Ontology(go_file, with_rels=with_relations)
    num_labels = len(idx_goid)
    
    # -------------------------------------------------------
    # 1. Pre-cache ancestors (including the term itself)
    # -------------------------------------------------------
    print("📦 Caching ancestor information...")
    ancestor_cache = {}
    valid_goids = set(goid_idx.keys())
    
    for go_id in valid_goids: 
        if go. has_term(go_id):
            try:
                ancestors = set(go.get_anchestors(go_id))
            except AttributeError: 
                ancestors = set(go.get_ancestors(go_id))
            
            # Add the term itself to the ancestor set
            ancestors.add(go_id)
            
            # Keep only terms in the label space and store their indices directly
            ancestor_cache[go_id] = [goid_idx[a] for a in ancestors if a in valid_goids]
        else:
            # ✅ Fix: If the GO term is absent from the OBO file, at least retain its own index
            ancestor_cache[go_id] = [goid_idx[go_id]]
            
    print(f"✅ Caching complete: {len(ancestor_cache)} GO terms")
    
    # -------------------------------------------------------
    # 2. Build matrices
    # -------------------------------------------------------
    num_samples = len(test_df)
    pred_scores = np. zeros((num_samples, num_labels), dtype=np.float32)
    true_scores = np. zeros((num_samples, num_labels), dtype=np.float32)
    
    print("🔄 Starting score propagation...")
    
    for i, row in enumerate(test_df. itertuples()):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{num_samples}", end='\r')

        # ===== True labels =====
        for go_id in row.gos:
            if go_id in ancestor_cache: 
                indices = ancestor_cache[go_id]
                true_scores[i, indices] = 1.0
            elif go_id in goid_idx: 
                # ✅ Fallback: If absent from the cache but present in the label space, mark the term itself
                true_scores[i, goid_idx[go_id]] = 1.0

        # ===== Predictions =====
        for go_id, score in row.predictions.items():
            if go_id in ancestor_cache: 
                indices = ancestor_cache[go_id]
                pred_scores[i, indices] = np.maximum(pred_scores[i, indices], score)
            elif go_id in goid_idx:
                # ✅ Fallback: If absent from the cache but present in the label space, assign the score to itself
                pred_scores[i, goid_idx[go_id]] = max(pred_scores[i, goid_idx[go_id]], score)
    
    print(f"\n📊 [Eval Debug] pred_scores shape: {pred_scores.shape}")
    print(f"📊 [Eval Debug] true_scores shape: {true_scores.shape}")
    print(f"📊 [Eval Debug] pred range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
    print(f"📊 [Eval Debug] true positives:  {int(true_scores.sum())}")
    
    # ✅ Add an overlap check
    pred_binary = (pred_scores > 0.5).astype(int)
    overlap = (pred_binary * true_scores).sum()
    print(f"📊 [Eval Debug] Predictions > 0.5 overlapping ground truth: {int(overlap)}")
    print(f"📊 [Eval Debug] Total predictions > 0.5: {int(pred_binary.sum())}")

    # -------------------------------------------------------
    # 3. Compute metrics
    # -------------------------------------------------------
    result_fmax, result_t, precisions, recalls = fmax(true_scores, pred_scores)
    
    precisions = np.array(precisions)
    recalls = np. array(recalls)
    sorted_idx = np.argsort(recalls)
    result_aupr = np.trapz(precisions[sorted_idx], recalls[sorted_idx])

    return result_fmax, result_aupr, result_t




# def new_compute_performance_deepgoplus(
#     test_df, 
#     go_file, 
#     ont, 
#     idx_goid,
#     goid_idx,
#     with_relations=True,
#     device='cuda' if torch.cuda.is_available() else 'cpu'
# ):
#     print(f"⚙️ Starting high-performance evaluation mode (Device: {device})")
    
#     # -------------------------------------------------------
#     # 1. Build the ancestor mapping matrix (map tensor)
#     # -------------------------------------------------------
#     print("📦 [Step 1] Building ancestor-index mapping...")
#     go = Ontology(go_file, with_rels=with_relations)
#     valid_goids = set(goid_idx.keys())
    
#     # Build two lists for the mapping tensors: term_idx -> ancestor_idx
#     # Example: If the ancestors of term 5 are [5, 2, 1], generate:
#     # src_indices (term): [5, 5, 5]
#     # dst_indices (anc):  [5, 2, 1]
#     src_list = []
#     dst_list = []
    
#     for go_id, current_idx in goid_idx.items():
#         if go.has_term(go_id):
#             try:
#                 # Support spelling differences across libraries
#                 ancestors = set(getattr(go, 'get_anchestors', getattr(go, 'get_ancestors', None))(go_id))
#             except:
#                 ancestors = set()
#             ancestors.add(go_id) # Include the term itself
            
#             # Filter and convert to indices
#             anc_indices = [goid_idx[a] for a in ancestors if a in valid_goids]
#         else:
#             anc_indices = [current_idx] # Only the term itself
            
#         # Record the mapping
#         src_list.extend([current_idx] * len(anc_indices))
#         dst_list.extend(anc_indices)
        
#     # Convert to tensors and move them to the GPU
#     # map_src: Term IDs from the original predictions
#     # map_dst: Ancestor IDs to which scores should propagate
#     map_src = torch.tensor(src_list, dtype=torch.long, device=device)
#     map_dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    
#     print(f"✅ Mapping complete: {len(map_src)} propagation paths")

#     # -------------------------------------------------------
#     # 2. Vectorize predictions (Flatten -> Expand -> Scatter Max)
#     # -------------------------------------------------------
#     print("🔄 [Step 2] Propagating predictions with vectorized operations...")
#     num_samples = len(test_df)
#     num_labels = len(goid_idx)
    
#     # 2.1 Flatten dictionaries from the DataFrame into lists
#     # This requires one unavoidable Python pass, but it is linear O(N), much faster than O(N*Depth)
#     batch_indices = []
#     term_indices = []
#     scores = []
    
#     # pandas.explode could also optimize this, but a list comprehension is usually fast enough here
#     for i, row in enumerate(test_df.itertuples()):
#         preds = row.predictions # dict {goid: score}
#         if not preds: continue
        
#         # Pre-filter existing keys for speed
#         valid_preds = [(goid_idx[k], v) for k, v in preds.items() if k in goid_idx]
#         if not valid_preds: continue
            
#         t_idxs, vals = zip(*valid_preds)
#         batch_indices.extend([i] * len(t_idxs))
#         term_indices.extend(t_idxs)
#         scores.extend(vals)
        
#     # Convert to tensors
#     t_batch = torch.tensor(batch_indices, dtype=torch.long, device=device)
#     t_term = torch.tensor(term_indices, dtype=torch.long, device=device)
#     t_score = torch.tensor(scores, dtype=torch.float32, device=device)
    
#     # 2.2 Core optimization: use the ancestor map for "broadcasting"
#     # t_term now contains predicted leaf nodes; find all their ancestors
#     # This resembles an SQL join
    
#     # A direct join is not possible because the mapping is many-to-many.
#     # One option is a large sparse matrix multiplication; another is the clearer method below.
#     # The map_src/map_dst arrays above are global.
#     # Batch data requires a more efficient approach.
    
#     # === Method A: Global sparse matrix multiplication (fastest and most memory-efficient) ===
#     # Build propagation matrix P (Label x Label), where P[i, j]=1 means j is an ancestor of i
#     # Build prediction matrix Y (Batch x Label)
#     # Result = Y @ P (the issue is that we need Max-Product, whereas matrix multiplication is Sum-Product)
#     # Therefore, use scatter-reduce instead.
    
#     # === Method B: Precompute expanded indices (PyTorch gather) ===
#     # This requires a lookup table, which is difficult because terms have different numbers of ancestors.
    
#     # === Method C: Sparse alignment (recommended) ===
#     # If the dataset is not extremely large, this approach can be used:
#     # A padded (num_labels, max_ancestors) tensor is too GPU-memory intensive.
    
#     # Step back and use the most robust full-expansion strategy:
#     # 1. Convert the ancestor map to CSR or another efficient lookup format
#     # In practice, Python loops for map lookup are slow.
#     # The ancestor map can be stored as an edge index and processed with torch_sparse (if installed).
#     # Without torch_sparse, use the vectorized-expansion technique below:
    
#     # ------------------------------------------------------------------
#     # Core optimization: Expand indices using map_src and map_dst
#     # ------------------------------------------------------------------
    
#     # To avoid a complex join, sort map_src and use searchsorted
#     # An even simpler option is a large sparse Boolean adjacency matrix (Num_Labels x Num_Labels)
#     # A[u, v] = 1 means v is an ancestor of u
#     indices = torch.stack([map_src, map_dst])
#     values = torch.ones(len(map_src), device=device)
#     # A_mat: Rows are child nodes and columns are ancestor nodes
#     A_mat = torch.sparse_coo_tensor(indices, values, (num_labels, num_labels)).coalesce()
    
#     # Build the raw prediction matrix (Batch x Num_Labels)
#     # This is a sparse matrix
#     pred_indices = torch.stack([t_batch, t_term])
#     pred_sparse = torch.sparse_coo_tensor(pred_indices, t_score, (num_samples, num_labels)).coalesce()
    
#     # Challenge: Matrix multiplication computes sums, but we need maxima.
#     # Solution: If GPU memory is sufficient (num_labels ~ 4000), convert to dense, though this may cause OOM.
#     # Alternative: iterative propagation by hierarchy level. The most general option is PyTorch scatter_reduce_ (requires torch >= 1.12).
    
#     # --- Final approach: Efficient PyTorch implementation (dense matrix with masking) ---
#     # If GPU memory permits (Batch 10000 x Label 4000 * 4 bytes = 160 MB), dense operations are fastest.
    
#     # 1. Initialize the result matrix
#     final_pred_scores = torch.zeros((num_samples, num_labels), dtype=torch.float32, device=device)
    
#     # 2. Fill in the original scores
#     # final_pred_scores[t_batch, t_term] = t_score # Multiple values at one position would overwrite each other
#     # Use scatter_reduce to take the maximum (handling duplicates)
#     final_pred_scores.index_put_((t_batch, t_term), t_score, accumulate=False) 
#     # If terms are unique in the original predictions, direct assignment is sufficient; otherwise use max.
    
#     # 3. Propagation
#     # Use the nonzero structure of matrix multiplication for max propagation
#     # A @ B computes sums, so it cannot be used directly.
#     # Use a loop that propagates by DAG level, or...
#     # Expanding once at the Python level is reasonably fast when tensor operations are used.
    
#     # Use index expansion, the standard approach for irregular data:
#     # 1. Find all nonzero prediction positions (sample_id, term_id, score)
#     # 2. Find every ancestor_id for each term_id
#     # 3. Generate new (sample_id, ancestor_id, score) entries
#     # 4. Scatter Max
    
#     # Fast ancestor_id lookup requires a broadcastable structure
#     # Because ancestor counts vary, use CSR-style arrays
#     # For simplicity, use pandas explode for expansion on the CPU and aggregation on the GPU
    
#     # ---> Hybrid mode: pandas for expansion, PyTorch for aggregation <---
#     # This is 100 times faster than pure Python loops
    
#     # A. Prepare data
#     flat_data = pd.DataFrame({
#         'sid': batch_indices,
#         'tid': term_indices,
#         'score': scores
#     })
    
#     # B. Prepare the ancestor mapping table (DataFrame)
#     anc_map_df = pd.DataFrame({
#         'tid': src_list,
#         'aid': dst_list
#     })
    
#     # C. SQL-style join (expansion) -> This step is very fast
#     merged = flat_data.merge(anc_map_df, on='tid', how='inner')
    
#     # D. Convert back to tensors and perform max aggregation on the GPU
#     m_sid = torch.tensor(merged['sid'].values, dtype=torch.long, device=device)
#     m_aid = torch.tensor(merged['aid'].values, dtype=torch.long, device=device)
#     m_scr = torch.tensor(merged['score'].values, dtype=torch.float32, device=device)
    
#     # Scatter Max
#     # Initialize to zero
#     pred_matrix = torch.zeros((num_samples, num_labels), dtype=torch.float32, device=device)
    
#     # Use scatter_reduce_ (PyTorch 1.12+) or scatter_max
#     # For older versions, use the index_put_ + sort trick; a newer version is assumed here
#     try:
#         # linear_index = sid * num_labels + aid
#         linear_idx = m_sid * num_labels + m_aid
#         flat_pred = pred_matrix.view(-1)
        
#         # reduce="amax" is essential
#         flat_pred.scatter_reduce_(0, linear_idx, m_scr, reduce="amax", include_self=False)
#         pred_matrix = flat_pred.view(num_samples, num_labels)
        
#     except AttributeError:
#         # Support older PyTorch versions with loops or scatter_max (requires torch_scatter)
#         # Simplest fallback: Convert back to pandas groupby
#         print("⚠️ PyTorch is outdated; falling back to pandas groupby aggregation...")
#         grp = merged.groupby(['sid', 'aid'])['score'].max().reset_index()
#         pred_matrix = torch.zeros((num_samples, num_labels), dtype=torch.float32, device=device)
#         pred_matrix[grp.sid.values, grp.aid.values] = torch.tensor(grp.score.values, dtype=torch.float32, device=device)

#     # -------------------------------------------------------
#     # 3. Process true labels (using sparse matrix multiplication directly)
#     # -------------------------------------------------------
#     print("🔄 [Step 3] Processing true labels with vectorized operations...")
#     # Collect ground-truth labels
#     t_batch_idxs = []
#     t_term_idxs = []
#     for i, row in enumerate(test_df.itertuples()):
#         for go_id in row.gos:
#             if go_id in goid_idx: # Process only terms in the label space
#                 t_batch_idxs.append(i)
#                 t_term_idxs.append(goid_idx[go_id])
                
#     # This is a binary matrix, so matrix multiplication can be used directly
#     # Y_true = Y_raw @ A_mat
#     # Y_raw: (Samples x Labels), 1 if labeled
#     # A_mat: (Labels x Labels), 1 if ancestor
    
#     # Build sparse matrix Y_raw
#     idx_t = torch.tensor([t_batch_idxs, t_term_idxs], device=device)
#     val_t = torch.ones(len(t_batch_idxs), device=device)
#     Y_raw = torch.sparse_coo_tensor(idx_t, val_t, (num_samples, num_labels))
    
#     # Matrix multiplication (Sparse @ Sparse -> Sparse)
#     # Nonzero elements in the result are 1 (for true labels, any connection means true)
#     # Note: A_mat was defined above as (src -> dst). Multiplication requires (Label x Ancestor)
#     # Y(NxL) @ A(LxL) -> Result(NxL)
    
#     # Here A_mat[i, j]=1 means j is an ancestor of i, matching the multiplication logic.
#     True_matrix_sparse = torch.sparse.mm(Y_raw, A_mat)
    
#     # Convert to dense and binarize (>0 becomes 1)
#     # If GPU memory is insufficient, keep it sparse, though Fmax computation requires dense data
#     true_matrix = True_matrix_sparse.to_dense()
#     true_matrix = (true_matrix > 0).float()
    
#     # -------------------------------------------------------
#     # 4. Compute metrics (move back to the CPU)
#     # -------------------------------------------------------
#     print("📊 Computing Fmax/AUPR...")
#     pred_np = pred_matrix.cpu().numpy()
#     true_np = true_matrix.cpu().numpy()
    
#     result_fmax, result_t, precisions, recalls = fmax(true_np, pred_np)
    
#     precisions = np.array(precisions)
#     recalls = np.array(recalls)
#     sorted_idx = np.argsort(recalls)
#     result_aupr = np.trapz(precisions[sorted_idx], recalls[sorted_idx])
    
#     # Release GPU memory
#     del map_src, map_dst, A_mat, pred_matrix, true_matrix
#     torch.cuda.empty_cache()

#     return result_fmax, result_aupr, result_t
