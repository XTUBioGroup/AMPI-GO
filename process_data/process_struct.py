import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# tfrecord_dir = '.\PDB-GO-valid'        # Directory containing your .tfrecords files
# save_dir = "./ca_matrices_valid"                # Directory for saving .npy files
# def extract_all_ca_from_tfrecords(tfrecord_dir, save_dir=None):
#     """
#     Extract Cα distance matrices in batches from all .tfrecords files in a directory.

#     Parameters:
#         tfrecord_dir: Directory containing .tfrecords files
#         save_dir: If not None, save extracted results as .npy files

#     Returns:
#         Extracted dictionary in the format {prot_id: ca_dist_matrix}
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
    Resolve case mismatches by matching TFRecord CA matrices to ProtT5 embedding sequence lengths.

    Parameters:
        tfrecord_dir: Directory of original Linux .tfrecords files
        feat_dir: ProtT5 feature directory generated on Windows (case-insensitive)
        invalid_pid_list: List of invalid proteins produced by mismatches
        save_dir: Directory in which to save CA distance matrices

    Returns:
        None (only saves .npy files)
    """
    os.makedirs(save_dir, exist_ok=True)

    # ================
    # 1) Build from feat_dir: {lowercase_prot_id -> (true_name, seq_len)}
    # ================
    feat_map = {}  # {lowercase_pid: (real_pid, L)}
    for fname in os.listdir(feat_dir):
        if fname.endswith(".pt"):
            real_pid = fname[:-3]  # Remove .pt
            lower_pid = real_pid.lower()

            feat = torch.load(os.path.join(feat_dir, fname))
            L = feat.shape[0]  # ProtT5 embedding sequence length

            feat_map[lower_pid] = (real_pid, L)

    print(f"[INFO] Loaded {len(feat_map)} T5 embeddings for length matching")

    # ================
    # 2) TFRecord parser
    # ================
    feature_description = {
        'prot_id': tf.io.FixedLenFeature([], tf.string),
        'L': tf.io.FixedLenFeature([], tf.int64),
        'ca_dist_matrix': tf.io.VarLenFeature(tf.float32),
    }

    # ================
    # 3) Iterate over TFRecords
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

            # Process only proteins in the invalid list
            if linux_pid not in invalid_pid_list and linux_pid_lower not in invalid_pid_list:
                continue

            # ================
            # Case-insensitive matching: Match Windows PIDs by length
            # ================
            if linux_pid_lower not in feat_map:
                print(f"[SKIP] No matching T5 embedding found for {linux_pid}; skipping")
                continue

            real_pid, t5_len = feat_map[linux_pid_lower]

            if t5_len != L:
                print(f"[WARN] Length mismatch: TFRecord={L}, T5={t5_len}, pid={real_pid}")
                continue

            # ================
            # Extract CA
            # ================
            ca_flat = tf.sparse.to_dense(example['ca_dist_matrix']).numpy()
            ca_matrix = ca_flat.reshape((L, L))

            save_path = os.path.join(save_dir, f"{real_pid}_ca.npy")
            np.save(save_path, ca_matrix)
            print(f"[SAVE] {real_pid} → {save_path}")


# =============================
# Usage example
# =============================

# Parameters
tfrecord_dir = "./PDB-GO-train"
feat_dir = "protT5_embeddings_train"
save_dir = "./ca_matrices"

# Load the invalid list
invalid_file = "graphs_train/invalid_proteins.txt"
invalid_pid_list = [line.strip() for line in open(invalid_file, "r") if line.strip()]

# Run
extract_all_ca_from_tfrecords_with_t5match(
    tfrecord_dir=tfrecord_dir,
    feat_dir=feat_dir,
    invalid_pid_list=invalid_pid_list,
    save_dir=save_dir
)
