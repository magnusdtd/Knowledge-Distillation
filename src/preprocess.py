from typing import List
import fitz
from pdf2image import convert_from_path
import os
from tqdm import tqdm
import re

def get_pdf_name(pdf_path: str) -> str:
    base = os.path.basename(pdf_path)
    name, _ = os.path.splitext(base)
    return name

def pdf_to_images(pdf_path: str, output_folder: str) -> List[str]:
    filename_prefix = get_pdf_name(pdf_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = convert_from_path(pdf_path)
    img_paths = []
    for i, img in tqdm(enumerate(images, 1), total=len(images), desc=f"Converting {filename_prefix} PDF pages to images"):
        filename = f"{filename_prefix}_page_{i}.png"
        img_path = os.path.join(output_folder, filename)
        img.save(img_path, "PNG")
        img_paths.append(img_path)
    return img_paths

def extract_text_pymupdf(pdf_path: str):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in tqdm(doc, desc="Extracting text from PDF pages", total=len(doc)):
            text += page.get_text()
    return text

def create_fileState(img_paths: List[str], output_file_path: str):
    with open(output_file_path, "w", encoding="utf-8") as f:
        for img_path in img_paths:
            f.write(f"{img_path}\t1\n")

def page_num_from_filename(fname):
    match = re.match(r".*page_(\d+)\.(png|jpg|jpeg)$", fname, re.IGNORECASE)
    return int(match.group(1)) if match else float('inf')

def find_images_with_prefix(folder: str, pdf_path: str) -> List[str]:
    prefix = get_pdf_name(pdf_path)
    if not (os.path.exists(folder) and os.path.isdir(folder)):
        return []
    prefix_pattern = f"{prefix}_page_"
    image_files = [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and fname.startswith(prefix_pattern)
    ]
    return sorted(image_files, key=lambda path: page_num_from_filename(os.path.basename(path)))

def preprocess(
    chapter4pdf_path: str,
    chapter12pdf_path: str,
    chapter4img_path: str,
    chapter12img_path: str,
    chapter4text_path: str,
    chapter12text_path: str,
    fileState_file_path: str
) -> List[str]:
    dataset_configs = [
        {
            "pdf_path": chapter4pdf_path,
            "output_folder": chapter4img_path,
            "text_path": chapter4text_path
        },
        {
            "pdf_path": chapter12pdf_path,
            "output_folder": chapter12img_path,
            "text_path": chapter12text_path
        }
    ]

    need_preprocess = False
    for config in dataset_configs:
        folder_exists = os.path.exists(config["output_folder"]) and os.path.isdir(config["output_folder"])
        images_exist = bool(find_images_with_prefix(config["output_folder"], config["pdf_path"]))
        if not (folder_exists and images_exist):
            need_preprocess = True
        if not os.path.isfile(config["text_path"]):
            need_preprocess = True

    if not need_preprocess:
        print("All image files and text files already exist, skipping preprocessing.")
        all_img_paths = []
        for config in dataset_configs:
            all_img_paths.extend(find_images_with_prefix(config["output_folder"], config["pdf_path"]))
        return all_img_paths

    all_img_paths = []
    for config in dataset_configs:
        img_paths = pdf_to_images(
            pdf_path=config["pdf_path"],
            output_folder=config["output_folder"]
        )
        all_img_paths.extend(img_paths)

    create_fileState(all_img_paths, fileState_file_path)

    for config in dataset_configs:
        extracted_text = extract_text_pymupdf(config["pdf_path"])
        with open(config["text_path"], 'w', encoding='utf-8') as f:
            f.write(extracted_text)

    return all_img_paths