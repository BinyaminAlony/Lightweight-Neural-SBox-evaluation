import os
import re
import shutil
from pathlib import Path

# Define the two folders
folder1 = r"C:\Users\USER\Documents\tommy_binyamin_proj\git-updated-version\Lightweight-Neural-SBox-evaluation\dataset_chunks_5bit_153450000size_225000cs_28_04_2026"
folder2 = r"C:\Users\USER\Documents\tommy_binyamin_proj\git-updated-version\Lightweight-Neural-SBox-evaluation\dataset_chunks_5bit_225000000size_225000cs_12_04_2026"

def get_max_chunk_index(folder):
    """Find the highest chunk index in a folder"""
    if not os.path.exists(folder):
        print(f"Warning: Folder does not exist: {folder}")
        return -1
    
    max_index = -1
    for filename in os.listdir(folder):
        match = re.match(r'chunk_(\d+)\.pt$', filename)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    
    return max_index

def rename_and_move_chunks(source_folder, target_folder, start_index):
    """Rename chunks in source folder starting from start_index and move to target folder"""
    chunks = []
    
    # Get all chunk files and their current indices
    for filename in os.listdir(source_folder):
        match = re.match(r'chunk_(\d+)\.pt$', filename)
        if match:
            index = int(match.group(1))
            chunks.append((index, filename))
    
    # Sort by index to maintain order
    chunks.sort(key=lambda x: x[0])
    
    # Rename and move files
    for old_index, filename in chunks:
        new_index = start_index + old_index
        new_filename = f"chunk_{new_index}.pt"
        
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, new_filename)
        
        print(f"Moving {filename} -> {new_filename}")
        shutil.move(source_path, target_path)
    
    print(f"\nMoved {len(chunks)} chunks from {source_folder} to {target_folder}")

# Main script
print("Finding max chunk indices...")
max_index_1 = get_max_chunk_index(folder1)
max_index_2 = get_max_chunk_index(folder2)

print(f"Folder 1 max index: {max_index_1}")
print(f"Folder 2 max index: {max_index_2}")

if max_index_1 == -1 and max_index_2 == -1:
    print("No chunk files found in either folder!")
elif max_index_1 == -1:
    print("No chunk files found in folder 1. Skipping consolidation.")
elif max_index_2 == -1:
    print("No chunk files found in folder 2. Skipping consolidation.")
else:
    # Determine which folder has more chunks and consolidate
    if max_index_1 > max_index_2:
        print(f"\nFolder 2 has lower max index. Renaming and moving folder 2 chunks to folder 1...")
        print(f"Starting new index from: {max_index_1 + 1}")
        rename_and_move_chunks(folder2, folder1, max_index_1 + 1)
        print(f"\nConsolidation complete! All chunks are now in folder 1.")
    elif max_index_2 > max_index_1:
        print(f"\nFolder 1 has lower max index. Renaming and moving folder 1 chunks to folder 2...")
        print(f"Starting new index from: {max_index_2 + 1}")
        rename_and_move_chunks(folder1, folder2, max_index_2 + 1)
        print(f"\nConsolidation complete! All chunks are now in folder 2.")
    else:
        print("Both folders have the same max index. No consolidation needed.")
