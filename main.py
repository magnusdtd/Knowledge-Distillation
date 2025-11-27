from src.ocr import OCR
from src.preprocess import preprocess
import argparse
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run OCR pipeline Script')
    parser.add_argument('--chapter4pdf', type=str, default='data/chapter4.pdf', help='Path to chapter 4 pdf file')
    parser.add_argument('--chapter12pdf', type=str, default='data/chapter12.pdf', help='Path to chapter 12 pdf file')
    parser.add_argument('--chapter4img', type=str, default='raw', help='Path to output folder for chapter 4 images')
    parser.add_argument('--chapter12img', type=str, default='raw', help='Path to output folder for chapter 12 images')
    parser.add_argument('--chapter4txt', type=str, default='txt/chapter4.txt', help='Path to output text file for chapter 4')
    parser.add_argument('--chapter12txt', type=str, default='txt/chapter12.txt', help='Path to output text file for chapter 12')
    parser.add_argument('--label_file', type=str, default='raw/Label.txt', help='Path to Label file')
    parser.add_argument('--fileState_file', type=str, default='raw/fileState.txt', help='Path to fileState file')
    parser.add_argument('--rec_gt_file', type=str, default='raw/rec_gt.txt', help='Path to rec_gt file')
    parser.add_argument('--crop_img_folder', type=str, default='crop_img', help='Path to cropped image folder')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for OCR processing')
    parser.add_argument('--test', action='store_true', help='Run in testing mode, process only a small subset')
    return parser.parse_args()

def main(args):

    chapter4pdf_path = args.chapter4pdf
    chapter12pdf_path = args.chapter12pdf
    chapter4img_path = args.chapter4img
    chapter12img_path = args.chapter12img
    chapter4text_path = args.chapter4txt
    chapter12text_path = args.chapter12txt
    label_file_path = args.label_file
    fileState_file_path = args.fileState_file
    rec_gt_file_path = args.rec_gt_file
    cropped_img_folder_path = args.crop_img_folder
    is_testing = args.test
    os.makedirs(chapter4img_path, exist_ok=True)
    os.makedirs(chapter12img_path, exist_ok=True)
    os.makedirs(os.path.dirname(chapter4text_path), exist_ok=True)
    os.makedirs(os.path.dirname(chapter12text_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(fileState_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(rec_gt_file_path), exist_ok=True)
    os.makedirs(cropped_img_folder_path, exist_ok=True)

    all_img_paths = preprocess(
        chapter4pdf_path,
        chapter12pdf_path,
        chapter4img_path,
        chapter12img_path,
        chapter4text_path,
        chapter12text_path,
        fileState_file_path
    )

    ocr = OCR(
        label_file_path,
        rec_gt_file_path,
        cropped_img_folder_path
    )
    batch_size = args.batch_size
    for start_idx in tqdm(range(0, len(all_img_paths), batch_size), desc="OCR Batches"):
        ocr.predict(all_img_paths[start_idx: start_idx + batch_size])
        if is_testing and (start_idx + batch_size) >= 8:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
