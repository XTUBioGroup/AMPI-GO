import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# tfrecord_dir = '.\PDB-GO-valid'        # 你的 .tfrecords 文件所在路径
# save_dir = "./ca_matrices_valid"                # 保存 .npy 的目录
# def extract_all_ca_from_tfrecords(tfrecord_dir, save_dir=None):
#     """
#     批量从指定目录下的所有 .tfrecords 文件中提取 Cα 距离矩阵

#     参数:
#         tfrecord_dir: 包含 .tfrecords 文件的目录
#         save_dir: 如果不为 None，提取结果会保存为 .npy 文件

#     返回:
#         提取出的 dict，格式为 {prot_id: ca_dist_matrix}
#     """
#     os.makedirs(save_dir, exist_ok=True) if save_dir else None
#     tfrecord_files = sorted(
#         [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecords")])

#     result_dict = {}

#     for tf_file in tqdm(tfrecord_files, desc="Processing TFRecords"):
#         raw_dataset = tf.data.TFRecordDataset(tf_file)

#         feature_description = {
#             'prot_id': tf.io.FixedLenFeature([], tf.string),
#             'L': tf.io.FixedLenFeature([], tf.int64),
#             'ca_dist_matrix': tf.io.VarLenFeature(tf.float32),
#         }

#         for raw_record in raw_dataset:
#             example = tf.io.parse_single_example(raw_record, feature_description)

#             prot_id = example['prot_id'].numpy().decode()
#             L = example['L'].numpy()
#             ca_matrix_flat = tf.sparse.to_dense(example['ca_dist_matrix']).numpy()
#             ca_matrix = ca_matrix_flat.reshape((L, L))

#             result_dict[prot_id] = ca_matrix

#             if save_dir:
#                 np.save(os.path.join(save_dir, f"{prot_id}_ca.npy"), ca_matrix)

#     return result_dict


# extract_all_ca_from_tfrecords(tfrecord_dir, save_dir)


import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import torch

def extract_all_ca_from_tfrecords_with_t5match(
    tfrecord_dir,
    feat_dir,
    invalid_pid_list,
    save_dir
):
    """
    修复大小写问题：根据 ProtT5 embedding 的序列长度匹配 TFRecord 中的 CA 矩阵。

    参数:
        tfrecord_dir: .tfrecords 文件目录（Linux 原始）
        feat_dir: Windows 下产生的 ProtT5 特征目录（大小写不敏感）
        invalid_pid_list: 无效蛋白质列表（因 mismatch 产生）
        save_dir: 要保存 CA 距离矩阵的目录

    返回:
        None（仅保存 .npy）
    """
    os.makedirs(save_dir, exist_ok=True)

    # ================
    # 1) 从 feat_dir 构建: {lowercase_prot_id → (true_name, seq_len)}
    # ================
    feat_map = {}  # {lowercase_pid: (real_pid, L)}
    for fname in os.listdir(feat_dir):
        if fname.endswith(".pt"):
            real_pid = fname[:-3]  # 去掉 .pt
            lower_pid = real_pid.lower()

            feat = torch.load(os.path.join(feat_dir, fname))
            L = feat.shape[0]  # ProtT5 embedding 序列长度

            feat_map[lower_pid] = (real_pid, L)

    print(f"[INFO] Loaded {len(feat_map)} T5 embeddings for length matching")

    # ================
    # 2) TFRecord 解析器
    # ================
    feature_description = {
        'prot_id': tf.io.FixedLenFeature([], tf.string),
        'L': tf.io.FixedLenFeature([], tf.int64),
        'ca_dist_matrix': tf.io.VarLenFeature(tf.float32),
    }

    # ================
    # 3) 遍历 TFRecords
    # ================
    tf_files = sorted(
        [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecords")]
    )

    for tf_file in tqdm(tf_files, desc="Processing TFRecords"):
        raw_dataset = tf.data.TFRecordDataset(tf_file)

        for raw_record in raw_dataset:
            example = tf.io.parse_single_example(raw_record, feature_description)

            linux_pid = example['prot_id'].numpy().decode()
            linux_pid_lower = linux_pid.lower()
            L = int(example['L'].numpy())

            # 仅处理 invalid 列表中的蛋白
            if linux_pid not in invalid_pid_list and linux_pid_lower not in invalid_pid_list:
                continue

            # ================
            # 大小写匹配：根据长度匹配 Windows PID
            # ================
            if linux_pid_lower not in feat_map:
                print(f"[SKIP] {linux_pid} 找不到对应的 T5 embedding，跳过")
                continue

            real_pid, t5_len = feat_map[linux_pid_lower]

            if t5_len != L:
                print(f"[WARN] 长度不一致: TFRecord={L}, T5={t5_len}, pid={real_pid}")
                continue

            # ================
            # 提取 CA
            # ================
            ca_flat = tf.sparse.to_dense(example['ca_dist_matrix']).numpy()
            ca_matrix = ca_flat.reshape((L, L))

            save_path = os.path.join(save_dir, f"{real_pid}_ca.npy")
            np.save(save_path, ca_matrix)
            print(f"[SAVE] {real_pid} → {save_path}")


# =============================
# 使用示例
# =============================

# 你的参数
tfrecord_dir = "./PDB-GO-train"
feat_dir = "protT5_embeddings_train"
save_dir = "./ca_matrices"

# 加载 invalid 列表
invalid_file = "graphs_train/invalid_proteins.txt"
invalid_pid_list = [line.strip() for line in open(invalid_file, "r") if line.strip()]

# 执行
extract_all_ca_from_tfrecords_with_t5match(
    tfrecord_dir=tfrecord_dir,
    feat_dir=feat_dir,
    invalid_pid_list=invalid_pid_list,
    save_dir=save_dir
)
