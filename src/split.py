from typing import List
import os
import shutil
import json

def split(
    img_paths: List[str], 
    label_file_path: str, 
    fileState_file_path: str, 
    folder_names: List[str]
):
    """
    Split the label and fileState file (as well as corresponding imgs) into subfolders according to folder_names.
    Each folder will get an (almost) equal part of the data from the input files and img_paths.
    """
    k = len(folder_names)
    n = len(img_paths)
    print(f"k={k}, n = {n}")
    assert k > 0, "folder_names should not be empty"
    assert n > 0, "img_paths is empty"

    # Read label_file
    with open(label_file_path, "r", encoding="utf-8") as f:
        label_lines = f.readlines()
    # Read fileState_file
    with open(fileState_file_path, "r", encoding="utf-8") as f:
        fileState_lines = f.readlines()
    assert len(label_lines) == len(img_paths), "label_file and img_paths size mismatch"
    assert len(fileState_lines) == len(img_paths), "fileState_file and img_paths size mismatch"

    # Create subfolders and prepare content
    split_img_paths = [[] for _ in range(k)]
    split_label_lines = [[] for _ in range(k)]
    split_fileState_lines = [[] for _ in range(k)]

    # Assign each example to a split
    for idx in range(n):
        folder_idx = idx * k // n
        split_img_paths[folder_idx].append(img_paths[idx])

        # Update image path in label file line
        orig_label_line = label_lines[idx]
        try:
            old_path, rest = orig_label_line.split('\t', 1)
        except ValueError:
            raise ValueError(f"Line format error in Label.txt at line {idx}: {orig_label_line}")
        img_filename = os.path.basename(old_path)
        new_img_path = f"{folder_names[folder_idx]}/{img_filename}"
        new_label_line = f"{new_img_path}\t{rest}"
        split_label_lines[folder_idx].append(new_label_line)
        
        # Update image path in fileState file line
        orig_fileState_line = fileState_lines[idx]
        try:
            _, rest2 = orig_fileState_line.split('\t', 1)
        except ValueError:
            raise ValueError(f"Line format error in fileState.txt at line {idx}: {orig_fileState_line}")
        new_fileState_line = f"{new_img_path}\t{rest2}"
        split_fileState_lines[folder_idx].append(new_fileState_line)

    for idx, folder in enumerate(folder_names):
        os.makedirs(folder, exist_ok=True)
        # Write label file
        with open(os.path.join(folder, 'Label.txt'), 'w', encoding='utf-8') as f:
            f.writelines(split_label_lines[idx])
        # Write fileState file
        with open(os.path.join(folder, 'fileState.txt'), 'w', encoding='utf-8') as f:
            f.writelines(split_fileState_lines[idx])

        # Copy corresponding imgs in the order as in the label/fileState file
        for label_line in split_label_lines[idx]:
            try:
                new_img_path_in_folder, _ = label_line.split('\t', 1)
                img_filename = os.path.basename(new_img_path_in_folder)
                # Find the original img_path with this filename in img_paths
                orig_img_path = None
                for path in img_paths:
                    if os.path.basename(path) == img_filename:
                        orig_img_path = path
                        break
                if orig_img_path is not None and os.path.exists(orig_img_path):
                    shutil.copy(orig_img_path, os.path.join(folder, img_filename))
                else:
                    print(f"[WARNING] Image {img_filename} ({orig_img_path}) does not exist and will not be copied.")
            except Exception as e:
                print(f"[ERROR] Could not parse label line '{label_line.strip()}': {e}")
