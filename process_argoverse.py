import os
import sys

# Add the current directory to sys.path to ensure we can import 'datasets'
# if running from the root of the project.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from datasets.argoverse_v2_dataset import ArgoverseV2Dataset

def process_dataset():
    root_path = r'C:\Users\admin\Downloads\argoverse_v2'
    splits = ['val', 'test', 'train']
    
    for split in splits:
        print(f"Processing split: {split}")
        try:
            # initializing the dataset triggers the downloading/processing logic
            dataset = ArgoverseV2Dataset(root=root_path, split=split)
            print(f"Successfully processed {split} split. {len(dataset)} samples found.")
        except Exception as e:
            print(f"Error processing {split} split: {e}")

if __name__ == "__main__":
    process_dataset()
