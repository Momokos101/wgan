from generate_vcu import generate_sequence, build_condition

print("=== Test 1: Basic generation ===")
out = generate_sequence(abnormal_type=2, sample_times=1)
print(out)

print("\n=== Test 2: Check dimensions ===")
cond = build_condition()
print("condition:", cond.shape)

from configs.config_vcu import Z_DIM
import numpy as np

z = np.random.normal(size=(1, Z_DIM)).astype("float32")
print("z:", z.shape)

print("\n=== Test 3: Multiple samples ===")
out = generate_sequence(abnormal_type=3, sample_times=3)
for i, r in enumerate(out):
    print(f"[{i}] hex={r['hex']}")
