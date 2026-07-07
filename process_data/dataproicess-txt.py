import tensorflow as tf
import numpy as np
import os

# 蛋白质字符表（与你的 one-hot 编码一致）
chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
         'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']

# TFRecord feature解析格式
def parse_example(example_proto):
    feature_description = {
        'prot_id': tf.io.FixedLenFeature([], tf.string),
        'L': tf.io.FixedLenFeature([], tf.int64),
        'seq_1hot': tf.io.VarLenFeature(tf.float32)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

# 输入目录 & 输出文件
tfrecord_dir = '.\PDB-GO-valid'  # 替换为你的路径
output_file = 'protein_id_and_sequence_valid.txt'

# 收集所有 tfrecords 文件
tfrecord_files = sorted([os.path.join(tfrecord_dir, f)
                         for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecords')])

# 打开写文件
with open(output_file, 'w') as out_f:
    for tfrecord_path in tfrecord_files:
        print(f"Processing: {tfrecord_path}")
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed = dataset.map(parse_example)

        for example in parsed:
            prot_id = example['prot_id'].numpy().decode()
            L = example['L'].numpy()
            onehot_flat = tf.sparse.to_dense(example['seq_1hot']).numpy()
            onehot_matrix = onehot_flat.reshape((L, 26))
            sequence = ''.join([chars[np.argmax(vec)] for vec in onehot_matrix])
            out_f.write(f"{prot_id}\t{sequence}\n")