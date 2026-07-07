import warnings
import numpy as np
import scipy.sparse as ssp
import torch
import numpy as np
import pandas as pd


__all__ = ['fmax', 'ROOT_GO_TERMS', 'compute_performance_deepgoplus', 'read_pkl', 'save_pkl']
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}


def fmax(targets, scores):
    """计算 Fmax 指标"""
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
    # 1. 预缓存祖先 (包含自身)
    # -------------------------------------------------------
    print("📦 正在缓存祖先信息...")
    ancestor_cache = {}
    valid_goids = set(goid_idx.keys())
    
    for go_id in valid_goids: 
        if go. has_term(go_id):
            try:
                ancestors = set(go.get_anchestors(go_id))
            except AttributeError: 
                ancestors = set(go.get_ancestors(go_id))
            
            # 将自身加入祖先集合
            ancestors.add(go_id)
            
            # 只保留在标签空间里的 term，直接存索引
            ancestor_cache[go_id] = [goid_idx[a] for a in ancestors if a in valid_goids]
        else:
            # ✅ 修复：GO 不在 obo 里，但至少保留自身索引
            ancestor_cache[go_id] = [goid_idx[go_id]]
            
    print(f"✅ 缓存完成，共 {len(ancestor_cache)} 个 GO term")
    
    # -------------------------------------------------------
    # 2. 构建矩阵
    # -------------------------------------------------------
    num_samples = len(test_df)
    pred_scores = np. zeros((num_samples, num_labels), dtype=np.float32)
    true_scores = np. zeros((num_samples, num_labels), dtype=np.float32)
    
    print("🔄 开始传播分数...")
    
    for i, row in enumerate(test_df. itertuples()):
        if i % 1000 == 0:
            print(f"   处理进度:  {i}/{num_samples}", end='\r')

        # ===== True labels =====
        for go_id in row.gos:
            if go_id in ancestor_cache: 
                indices = ancestor_cache[go_id]
                true_scores[i, indices] = 1.0
            elif go_id in goid_idx: 
                # ✅ 兜底：不在缓存但在标签空间，标记自身
                true_scores[i, goid_idx[go_id]] = 1.0

        # ===== Predictions =====
        for go_id, score in row.predictions.items():
            if go_id in ancestor_cache: 
                indices = ancestor_cache[go_id]
                pred_scores[i, indices] = np.maximum(pred_scores[i, indices], score)
            elif go_id in goid_idx:
                # ✅ 兜底：不在缓存但在标签空间，赋值自身
                pred_scores[i, goid_idx[go_id]] = max(pred_scores[i, goid_idx[go_id]], score)
    
    print(f"\n📊 [Eval Debug] pred_scores shape: {pred_scores.shape}")
    print(f"📊 [Eval Debug] true_scores shape: {true_scores.shape}")
    print(f"📊 [Eval Debug] pred range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
    print(f"📊 [Eval Debug] true positives:  {int(true_scores.sum())}")
    
    # ✅ 添加重叠检查
    pred_binary = (pred_scores > 0.5).astype(int)
    overlap = (pred_binary * true_scores).sum()
    print(f"📊 [Eval Debug] 预测>0.5 与真实重叠: {int(overlap)}")
    print(f"📊 [Eval Debug] 预测>0.5 总数: {int(pred_binary.sum())}")

    # -------------------------------------------------------
    # 3. 计算指标
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
#     print(f"⚙️ 启动高性能评估模式 (Device: {device})")
    
#     # -------------------------------------------------------
#     # 1. 构建祖先映射矩阵 (Map Tensor)
#     # -------------------------------------------------------
#     print("📦 [Step 1] 构建祖先索引映射...")
#     go = Ontology(go_file, with_rels=with_relations)
#     valid_goids = set(goid_idx.keys())
    
#     # 构建两个列表，用于创建映射张量: term_idx -> ancestor_idx
#     # 例如：如果 term 5 的祖先是 [5, 2, 1]，则生成：
#     # src_indices (term): [5, 5, 5]
#     # dst_indices (anc):  [5, 2, 1]
#     src_list = []
#     dst_list = []
    
#     for go_id, current_idx in goid_idx.items():
#         if go.has_term(go_id):
#             try:
#                 # 兼容不同库的拼写
#                 ancestors = set(getattr(go, 'get_anchestors', getattr(go, 'get_ancestors', None))(go_id))
#             except:
#                 ancestors = set()
#             ancestors.add(go_id) # 加上自身
            
#             # 过滤并转为索引
#             anc_indices = [goid_idx[a] for a in ancestors if a in valid_goids]
#         else:
#             anc_indices = [current_idx] # 只有自身
            
#         # 记录映射关系
#         src_list.extend([current_idx] * len(anc_indices))
#         dst_list.extend(anc_indices)
        
#     # 转为 Tensor 并移至 GPU
#     # map_src: 原始预测的词 ID
#     # map_dst: 应该把分数值传导到的祖先 ID
#     map_src = torch.tensor(src_list, dtype=torch.long, device=device)
#     map_dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    
#     print(f"✅ 映射构建完成，共包含 {len(map_src)} 条传播路径")

#     # -------------------------------------------------------
#     # 2. 向量化处理 Predictions (Flatten -> Expand -> Scatter Max)
#     # -------------------------------------------------------
#     print("🔄 [Step 2] 向量化传播 Predictions...")
#     num_samples = len(test_df)
#     num_labels = len(goid_idx)
    
#     # 2.1 将 DataFrame 中的字典展平为列表
#     # 这一步不可避免需要一次 Python 遍历，但它是线性的 O(N)，比 O(N*Depth) 快得多
#     batch_indices = []
#     term_indices = []
#     scores = []
    
#     # 也可以用 pandas 的 explode 优化，但这里用列表推导式通常足够快
#     for i, row in enumerate(test_df.itertuples()):
#         preds = row.predictions # dict {goid: score}
#         if not preds: continue
        
#         # 预先筛选只存在的 key，加速
#         valid_preds = [(goid_idx[k], v) for k, v in preds.items() if k in goid_idx]
#         if not valid_preds: continue
            
#         t_idxs, vals = zip(*valid_preds)
#         batch_indices.extend([i] * len(t_idxs))
#         term_indices.extend(t_idxs)
#         scores.extend(vals)
        
#     # 转为 Tensor
#     t_batch = torch.tensor(batch_indices, dtype=torch.long, device=device)
#     t_term = torch.tensor(term_indices, dtype=torch.long, device=device)
#     t_score = torch.tensor(scores, dtype=torch.float32, device=device)
    
#     # 2.2 核心加速：利用 Ancestor Map 进行“广播”
#     # 现在的 t_term 是预测的叶子节点，我们需要找到它所有的祖先
#     # 这是一个类似于 SQL Join 的操作
    
#     # 这里我们不能直接 join，因为 map 是多对多的。
#     # 技巧：创建一个巨大的稀疏矩阵乘法，或者使用这种更直观的方法：
#     # 实际上，上面的 map_src/map_dst 是全局的。
#     # 为了处理 Batch 数据，我们需要更高效的方法。
    
#     # === 方法 A: 全局稀疏矩阵乘法 (最快，最省显存) ===
#     # 构建传播矩阵 P (Label x Label)，P[i, j]=1 表示 j 是 i 的祖先
#     # 构建预测矩阵 Y (Batch x Label)
#     # 结果 = Y @ P (这里稍微有点问题是 Max-Product，矩阵乘法是 Sum-Product)
#     # 所以我们还是用 Scatter Reduce 方法。
    
#     # === 方法 B: 预计算 Expanded Indices (PyTorch Gather) ===
#     # 我们需要构建一个 lookup table，这比较难，因为每个 term 祖先数量不同。
    
#     # === 方法 C: 稀疏对齐 (推荐) ===
#     # 如果数据量不是特别巨大，我们可以用这种方式：
#     # 创建一个 (num_labels, max_ancestors) 的 padded tensor? 不行，太费显存。
    
#     # 让我们回退一步，使用最稳健的【全展开策略】：
#     # 1. 将 ancestor map 转为 CSR 格式或者类似的高效查询
#     # 但实际上，Python 循环处理 "map lookup" 是慢的。
#     # 我们可以把 ancestor map 存成 Edge Index 形式，然后用 torch_sparse (如果安装了)
#     # 如果没有 torch_sparse，我们可以用下面的 "Vectorized Expand" 技巧：
    
#     # ------------------------------------------------------------------
#     # 【优化核心】：利用 map_src 和 map_dst 进行索引扩展
#     # ------------------------------------------------------------------
    
#     # 为了避免复杂的 Join，我们直接把 map_src 排序，然后用 searchsorted
#     # 但更简单的是：构建一个巨大的稀疏布尔矩阵 Adjacency Matrix (Num_Labels x Num_Labels)
#     # A[u, v] = 1 表示 v 是 u 的祖先
#     indices = torch.stack([map_src, map_dst])
#     values = torch.ones(len(map_src), device=device)
#     # A_mat: 行是子节点，列是祖先节点
#     A_mat = torch.sparse_coo_tensor(indices, values, (num_labels, num_labels)).coalesce()
    
#     # 构建 Raw Prediction Matrix (Batch x Num_Labels)
#     # 这是一个稀疏矩阵
#     pred_indices = torch.stack([t_batch, t_term])
#     pred_sparse = torch.sparse_coo_tensor(pred_indices, t_score, (num_samples, num_labels)).coalesce()
    
#     # !!! 难点：矩阵乘法是 Sum，我们要 Max。
#     # 解决：如果显存够大（num_labels ~ 4000），可以转 Dense 算，但这可能 OOM。
#     # 替代方案：Iterative Propagation (按层级)。但最通用的是利用 PyTorch 的 scatter_reduce_ (需要 torch >= 1.12)
    
#     # --- 最终方案：基于 PyTorch 的高效实现 (Dense Matrix with Masking) ---
#     # 如果显存允许 (Batch 10000 x Label 4000 * 4 bytes = 160MB)，直接上 Dense 最快。
    
#     # 1. 初始化结果矩阵
#     final_pred_scores = torch.zeros((num_samples, num_labels), dtype=torch.float32, device=device)
    
#     # 2. 填充原始分数
#     # final_pred_scores[t_batch, t_term] = t_score # 这样如果同一个位置有多个值会覆盖
#     # 使用 scatter_reduce 取最大 (防止重复)
#     final_pred_scores.index_put_((t_batch, t_term), t_score, accumulate=False) 
#     # 注意：如果原始预测里同一个词没重复，直接赋值即可。如果有重复用 max。
    
#     # 3. 传播 (Propagation)
#     # 利用矩阵乘法的非零结构来做 Max 传播
#     # 由于 A @ B 是 Sum，我们不能直接用。
#     # 我们用一个巧妙的循环：按 DAG 层级传播，或者...
#     # 直接在 Python 层面做一次“展开”其实并不慢，如果利用了 tensor 操作。
    
#     # 让我们使用【索引扩展法】，这是处理不规则数据的标准做法：
#     # 1. 找到所有非零预测的位置 (sample_id, term_id, score)
#     # 2. 根据 term_id 找到所有 ancestor_id
#     # 3. 生成新的 (sample_id, ancestor_id, score)
#     # 4. Scatter Max
    
#     # 为了快速找到 ancestor_id，我们需要一个能够广播的结构
#     # 由于 ancestors 数量不一，我们使用 "CSR 风格" 的数组
#     # 但为了代码简洁，这里使用【Pandas Explode】辅助 (CPU上做扩展，GPU上做聚合)
    
#     # ---> 混合模式：Pandas 做扩展，PyTorch 做聚合 <---
#     # 这比纯 Python 循环快 100 倍
    
#     # A. 准备数据
#     flat_data = pd.DataFrame({
#         'sid': batch_indices,
#         'tid': term_indices,
#         'score': scores
#     })
    
#     # B. 准备祖先映射表 (DataFrame)
#     anc_map_df = pd.DataFrame({
#         'tid': src_list,
#         'aid': dst_list
#     })
    
#     # C. SQL 风格 Join (扩展) -> 这一步极快
#     merged = flat_data.merge(anc_map_df, on='tid', how='inner')
    
#     # D. 转回 Tensor 并在 GPU 上做 Max 聚合
#     m_sid = torch.tensor(merged['sid'].values, dtype=torch.long, device=device)
#     m_aid = torch.tensor(merged['aid'].values, dtype=torch.long, device=device)
#     m_scr = torch.tensor(merged['score'].values, dtype=torch.float32, device=device)
    
#     # Scatter Max
#     # 初始化为 0
#     pred_matrix = torch.zeros((num_samples, num_labels), dtype=torch.float32, device=device)
    
#     # 使用 scatter_reduce_ (PyTorch 1.12+) 或 scatter_max
#     # 如果版本旧，用 index_put_ + sort 的 trick，但这里假设较新版本
#     try:
#         # linear_index = sid * num_labels + aid
#         linear_idx = m_sid * num_labels + m_aid
#         flat_pred = pred_matrix.view(-1)
        
#         # reduce="amax" 是关键
#         flat_pred.scatter_reduce_(0, linear_idx, m_scr, reduce="amax", include_self=False)
#         pred_matrix = flat_pred.view(num_samples, num_labels)
        
#     except AttributeError:
#         # 兼容旧版本 PyTorch: 使用循环或者 scatter_max (需 torch_scatter)
#         # 这里写一个最简单的 fallback：转回 Pandas Groupby
#         print("⚠️ PyTorch 版本较旧，降级使用 Pandas Groupby Aggregation...")
#         grp = merged.groupby(['sid', 'aid'])['score'].max().reset_index()
#         pred_matrix = torch.zeros((num_samples, num_labels), dtype=torch.float32, device=device)
#         pred_matrix[grp.sid.values, grp.aid.values] = torch.tensor(grp.score.values, dtype=torch.float32, device=device)

#     # -------------------------------------------------------
#     # 3. 处理 True Labels (直接用 Sparse Matrix Mult)
#     # -------------------------------------------------------
#     print("🔄 [Step 3] 向量化处理 True Labels...")
#     # 收集真实标签
#     t_batch_idxs = []
#     t_term_idxs = []
#     for i, row in enumerate(test_df.itertuples()):
#         for go_id in row.gos:
#             if go_id in goid_idx: # 只处理在标签空间的
#                 t_batch_idxs.append(i)
#                 t_term_idxs.append(goid_idx[go_id])
                
#     # 这是一个 Binary 矩阵，我们可以直接用矩阵乘法！
#     # Y_true = Y_raw @ A_mat
#     # Y_raw: (Samples x Labels), 1 if labeled
#     # A_mat: (Labels x Labels), 1 if ancestor
    
#     # 构建稀疏矩阵 Y_raw
#     idx_t = torch.tensor([t_batch_idxs, t_term_idxs], device=device)
#     val_t = torch.ones(len(t_batch_idxs), device=device)
#     Y_raw = torch.sparse_coo_tensor(idx_t, val_t, (num_samples, num_labels))
    
#     # 矩阵乘法 (Sparse @ Sparse -> Sparse)
#     # 结果矩阵中，非零元素即为 1 (因为是 True Label，只要有连接就是 True)
#     # 注意：A_mat 之前定义过，是 (src -> dst)。矩阵乘法需要 (Label x Ancestor)
#     # Y(NxL) @ A(LxL) -> Result(NxL)
    
#     # 此时 A_mat[i, j]=1 表示 j 是 i 的祖先。符合乘法逻辑。
#     True_matrix_sparse = torch.sparse.mm(Y_raw, A_mat)
    
#     # 转 Dense 并二值化 (>0 即为 1)
#     # 如果显存不够，可以 keep sparse，但 fmax 计算需要 dense
#     true_matrix = True_matrix_sparse.to_dense()
#     true_matrix = (true_matrix > 0).float()
    
#     # -------------------------------------------------------
#     # 4. 计算指标 (移回 CPU)
#     # -------------------------------------------------------
#     print("📊 计算 Fmax/AUPR...")
#     pred_np = pred_matrix.cpu().numpy()
#     true_np = true_matrix.cpu().numpy()
    
#     result_fmax, result_t, precisions, recalls = fmax(true_np, pred_np)
    
#     precisions = np.array(precisions)
#     recalls = np.array(recalls)
#     sorted_idx = np.argsort(recalls)
#     result_aupr = np.trapz(precisions[sorted_idx], recalls[sorted_idx])
    
#     # 释放显存
#     del map_src, map_dst, A_mat, pred_matrix, true_matrix
#     torch.cuda.empty_cache()

#     return result_fmax, result_aupr, result_t