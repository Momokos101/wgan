# scripts/vcu/test_vcu_data_count.py

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sequence.vcu_data_process import load_vcu_data

if __name__ == "__main__":
    train_data, test_data, max_seq_len = load_vcu_data()
    train_count = sum(1 for _ in train_data)
    test_count = sum(1 for _ in test_data)
    print("train batches:", train_count)
    print("test batches:", test_count)
    print("max_seq_len:", max_seq_len)
