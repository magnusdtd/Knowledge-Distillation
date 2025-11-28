import torch
from paddleocr import TextDetection
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import numpy as np
import json
import os
import re

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    return obj

def extract_prefix(img_path):
    fname = os.path.basename(img_path)
    match = re.match(r"([a-zA-Z0-9]+)_page_\d+\.(png|jpg|jpeg)$", fname, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return "unknownprefix"

class OCR:
    def __init__(
        self: str,
        label_file_path: str,
        rec_gt_file_path: str,
        cropped_img_folder_path: str
    ) -> None:

        self.label_file_path = label_file_path
        self.rec_gt_file_path = rec_gt_file_path
        self.cropped_img_folder_path = cropped_img_folder_path
        os.makedirs(cropped_img_folder_path, exist_ok=True)

        use_gpu = True if torch.cuda.is_available() else False
        if use_gpu:
            print("GPU is available. Using CUDA for computation.")
        else:
            print("GPU not available. Using CPU for computation.")
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = True
        config['predictor']['beamsearch'] = True
        config['device'] = 'cuda:0' if use_gpu else 'cpu'
        self.recognitor = Predictor(config)
        self.detector = TextDetection()

    def __save_to_label_file(self, all_page_entries: dict):
        mode = "a" if os.path.exists(self.label_file_path) else "w"
        with open(self.label_file_path, mode, encoding="utf-8") as f:
            for rel_img_path, entries in all_page_entries.items():
                entries_serializable = convert_ndarray(entries)
                f.write(f"{rel_img_path}\t{json.dumps(entries_serializable, ensure_ascii=False)}\n")

    def __save_to_rec_gt_file(self, all_page_entries: dict):
        mode = "a" if os.path.exists(self.rec_gt_file_path) else "w"
        with open(self.rec_gt_file_path, mode, encoding="utf-8") as f:
            for img_idx, (img_path, entries) in enumerate(all_page_entries.items()):
                prefix = extract_prefix(img_path)
                for poly_idx, entry in enumerate(entries):
                    crop_filename = f"{prefix}_page_{img_idx+1}_crop_{poly_idx}.jpg"
                    rel_crop_path = os.path.join(self.cropped_img_folder_path, crop_filename)
                    text = entry["transcription"]
                    f.write(f"{rel_crop_path}\t{text}\n")

    def predict(self, img_paths: list):
        """
        This function uses batch inference and writes results in PPOCRLabel format.
        """
        print(f"[DEBUG] img_paths={img_paths}")

        # PaddleOCR - batch detection
        dec_result = self.detector.predict(img_paths)
        all_page_entries = {}  # Store entries for each image for the PPOCRLabel format
        crop_imgs = []
        crops_per_image = []  # List of how many crops per page

        for img_idx, (img_path, page_result) in enumerate(zip(img_paths, dec_result)):
            boxes = page_result['dt_polys']

            pil_img = Image.open(img_path).convert("RGB")
            crops_this_page = []
            crop_file_names_this_page = []

            prefix = extract_prefix(img_path)

            for poly_idx, poly in enumerate(boxes):
                crop_filename = f"{prefix}_page_{img_idx+1}_crop_{poly_idx}.jpg"
                crop_img_path = os.path.join(self.cropped_img_folder_path, crop_filename)
                poly_np = np.array(poly, dtype=np.int32)
                min_x = np.min(poly_np[:, 0])
                max_x = np.max(poly_np[:, 0])
                min_y = np.min(poly_np[:, 1])
                max_y = np.max(poly_np[:, 1])

                min_x, min_y = max(min_x, 0), max(min_y, 0)
                max_x, max_y = min(max_x, pil_img.width), min(max_y, pil_img.height)

                crop = pil_img.crop((min_x, min_y, max_x, max_y))
                crop.save(crop_img_path, "JPEG")
                crop_imgs.append(crop)
                crops_this_page.append(poly)
                crop_file_names_this_page.append(crop_filename)

            crops_per_image.append(len(crops_this_page))

        # VietOCR batch inference on all cropped images, order matches crop_imgs
        rec_result = self.recognitor.predict_batch(crop_imgs)

        # Group rec results by page/image according to crops_per_image
        rec_result_pointer = 0
        for img_path, page_result, n in zip(img_paths, dec_result, crops_per_image):
            boxes = page_result['dt_polys']
            recs = rec_result[rec_result_pointer:rec_result_pointer + n]
            rec_result_pointer += n

            entries = []
            for poly, text in zip(boxes, recs):
                entry = {
                    "transcription": text,
                    "points": poly,
                    "difficult": False
                }
                entries.append(entry)
            all_page_entries[img_path] = entries

        # Write to output in PPOCRLabel format
        self.__save_to_label_file(all_page_entries)
        self.__save_to_rec_gt_file(all_page_entries)

        return all_page_entries