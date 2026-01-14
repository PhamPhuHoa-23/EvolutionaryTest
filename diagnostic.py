import os
import sys
import time

print("Starting diagnostic script...", flush=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

print(f"Current dir: {current_dir}", flush=True)

try:
    print("Importing ArgoverseV2Dataset...", flush=True)
    from datasets.argoverse_v2_dataset import ArgoverseV2Dataset
    print("Import successful!", flush=True)
except Exception as e:
    print(f"Import failed: {e}", flush=True)

root_path = r'C:\Users\admin\Downloads\argoverse_v2'
val_raw = os.path.join(root_path, 'val', 'raw')
print(f"Checking {val_raw}...", flush=True)
if os.path.exists(val_raw):
    print(f"Exists. Items in raw: {len(os.listdir(val_raw))}", flush=True)
else:
    print("Does not exist!", flush=True)
