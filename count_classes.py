import os
for split in ['train', 'validation', 'test']:
    root = f'dataset/{split}'
    if not os.path.exists(root):
        continue
    print(f"\n=== {split.upper()} ===")
    total = 0
    for c in sorted(os.listdir(root)):
        p = os.path.join(root, c)
        if os.path.isdir(p):
            n = len(os.listdir(p))
            total += n
            print(f"  {c:12s}: {n:5d}")
    print(f"  {'TOTAL':12s}: {total:5d}")
