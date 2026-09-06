import tensorflow as tf
import numpy as np
import os

# Protein alphabet (consistent with the one-hot encoding)
chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
         'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']

# TFRecord feature parsing schema
def parse_example(example_proto):
    feature_description = {
        'prot_id': tf.io.FixedLenFeature([], tf.string),
        'L': tf.io.FixedLenFeature([], tf.int64),
        'seq_1hot': tf.io.VarLenFeature(tf.float32)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

# Input directory and output file
tfrecord_dir = '.\PDB-GO-valid'  # Replace with your path
output_file = 'protein_id_and_sequence_valid.txt'

# Collect all TFRecord files
tfrecord_files = sorted([os.path.join(tfrecord_dir, f)
                         for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecords')])

# Open the output file for writing
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
