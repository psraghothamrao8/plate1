import os
import copy
import time
import torch
import csv
import logging
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import optuna
from backend.process_logger import process_logger
from backend.config import (LOGS_DIR, get_data_paths, DATASET_ROOT, BASE_DIR)
from backend.logger_config import setup_logger
from backend.services.data_service import  train_transform, val_transform
import subprocess
import json
import re
import glob
import shutil
from PIL import Image as PILImage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

json_path = os.path.join(os.getcwd(), 'backend/config.json')
with open(json_path, 'r',encoding="utf-8") as json_file:
    json_data = json.load(json_file)

DL_PROCESS_WRAPPER_PATH = json_data['paths']['dl_path']
_DEFAULT_HPARAMS = json_data.get("default_hyperparameters", {})
# print("DL PATH: ", DL_PROCESS_WRAPPER_PATH)

logger = setup_logger("pluto_trainer", os.path.join(LOGS_DIR, "startup.log"), mode='a')

# Training type flag: "classification" or "segmentation"
TRAINING_TYPE = "classification"

# DL mode strings per training type
DL_MODES = {
    "classification":  {"Train": "Train", "Evaluate": "Evaluate", "Test": "Test"},
    "segmentation":    {"Train": "segTrainBinary", "Evaluate": "segEvaluateBinary", "Test": "segTestBinary"},
    "objectdetection": {"Train": "ODMTrain", "Evaluate": "ODMEvaluate", "Test": "ODMTest"},
}

# Template directories: JSON configs (and, for segmentation, model files) live here.
# Classification templates are in req_files/<classification_subfolder>/
# Segmentation templates (JSON + ModelConfig) are in req_files/segmentation/
# Object detection templates (YOLO OBB) are in req_files/objectdetection/Model/Train|Test/
# Use __file__ so the path is always relative to this file (backend/services/),
# not BASE_DIR which resolves to the project root (one level above backend/).
_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR  = os.path.dirname(_SERVICES_DIR)
CLASSIFICATION_TEMPLATE_DIR = os.path.join(_BACKEND_DIR, "req_files", "classification")
SEGMENTATION_TEMPLATE_DIR   = os.path.join(_BACKEND_DIR, "req_files", "segmentation")
OBJECTDETECTION_TEMPLATE_DIR       = os.path.join(_BACKEND_DIR, "req_files", "objectdetection")
OBJECTDETECTION_TRAIN_TEMPLATE_DIR = os.path.join(OBJECTDETECTION_TEMPLATE_DIR, "Model", "Train")
OBJECTDETECTION_TEST_TEMPLATE_DIR  = os.path.join(OBJECTDETECTION_TEMPLATE_DIR, "Model", "Test")


# ----------------------------------------------------------------------
# Model path utilities
# ----------------------------------------------------------------------
def get_latest_model_name(storage_root):
    """Finds the highest Model_X folder in storage_root/Train/."""
    train_dir = os.path.join(storage_root, "Train")
    if not os.path.exists(train_dir):
        return "Model_1"

    max_num = 0
    for d in os.listdir(train_dir):
        if d.startswith("Model_") and os.path.isdir(os.path.join(train_dir, d)):
            try:
                num = int(d.split("_")[1])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return f"Model_{max_num}" if max_num > 0 else "Model_1"


def get_model_paths(storage_root, model_name):
    train_model_dir = os.path.join(storage_root, "Train", model_name)
    test_model_dir  = os.path.join(storage_root, "Test",  model_name)
    # os.makedirs(train_model_dir, exist_ok=True)
    # os.makedirs(test_model_dir,  exist_ok=True)
    return {
        "model_name":      model_name,
        "train_dir":       train_model_dir,
        "test_dir":        test_model_dir,
        "training_json":   os.path.join(train_model_dir, "Training.json"),
        "evaluation_json": os.path.join(train_model_dir, "Evaluation.json"),
        "testing_json":    os.path.join(test_model_dir,  "Testing.json"),
        "export_json":     os.path.join(train_model_dir, "Export_TF.json"),
        "status_file":     os.path.join(train_model_dir, "Status.txt"),
    }


# ----------------------------------------------------------------------
# JSON I/O utilities
# ----------------------------------------------------------------------
def load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error(f"Failed to load JSON from {path}: {exc}")
        return None


def dump_json(data: Dict, path: str) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        logger.error(f"Failed to write JSON to {path}: {exc}")
        return False


def ensure_json_from_template(target_path: str, template_name: str, base_dir: str) -> bool:
    """Copy template to target_path if target doesn't exist. Returns True when ready."""
    if os.path.exists(target_path):
        return True
    candidates = [
        os.path.join(base_dir, template_name.lower()),
        os.path.join(base_dir, template_name),
    ]
    src = next((c for c in candidates if os.path.exists(c)), None)
    if src is None:
        logger.error(f"Template {template_name} not found in {base_dir}")
        return False
    try:
        shutil.copy(src, target_path)
        logger.info(f"Created {target_path} from template {src}")
        return True
    except Exception as exc:
        logger.error(f"Copying template failed: {exc}")
        return False


# ----------------------------------------------------------------------
# Classification: dataset scanning helpers
# ----------------------------------------------------------------------
def resolve_dataset_paths(custom_dataset_path: Optional[str] = None) -> Tuple[str, str, str]:
    if custom_dataset_path and os.path.isdir(custom_dataset_path):
        return (
            os.path.join(custom_dataset_path, "train"),
            os.path.join(custom_dataset_path, "val"),
            os.path.join(custom_dataset_path, "test"),
        )
    paths = get_data_paths()
    return paths["train"], paths["val"], paths["test"]


def discover_classes(train_dir: str) -> List[str]:
    if not os.path.isdir(train_dir):
        logger.warning(f"Train folder {train_dir} does not exist.")
        return []
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes.sort()
    return classes


def load_user_meta(custom_dataset_path: Optional[str]) -> Tuple[Set[str], Set[str]]:
    meta_path = None
    if custom_dataset_path:
        cand = os.path.join(custom_dataset_path, "dataset_meta.json")
        if os.path.isfile(cand):
            meta_path = cand
    if not meta_path:
        cand = os.path.join(DATASET_ROOT, "dataset_meta.json")
        if os.path.isfile(cand):
            meta_path = cand
    ok_set, ng_set = set(), set()
    if meta_path:
        meta = load_json(meta_path)
        if meta:
            ok_set = {c.lower() for c in meta.get("ok_classes", [])}
            ng_set = {c.lower() for c in meta.get("ng_classes", [])}
    return ok_set, ng_set


# def build_class_lists(classes: List[str]) -> Tuple[List[Dict], List[Dict]]:
#     non_ng = json_data.get("non_ng_class", classes[0]).lower()
#     class_lst, class_bin = [], []
#     for cls_name in classes:
#         class_lst.append({"ClassName": cls_name, "iClassWeights": 1})
#         bin_name = "OK" if cls_name.lower() == non_ng else "NG"
#         class_bin.append({"classBinName": bin_name})
#     return class_lst, class_bin

def build_class_lists(classes: List[str],ok_class: Optional[str]=None) -> Tuple[List[Dict], List[Dict]]:
    class_lst, class_bin = [], []
    for cls_name in classes:
        class_lst.append({
            "ClassName": cls_name,
            "iClassWeights": 1
        })

        bin_name = "OK" if cls_name.lower() == ok_class.lower() else "NG"

        class_bin.append({
            "classBinName": bin_name
        })

    return class_lst, class_bin

def scan_images(root_dir: str, classes: List[str], class_to_idx: Dict[str, int]) -> List[Dict]:
    """Walk root_dir and collect every image inside class subfolders."""
    img_list = []
    if not os.path.isdir(root_dir):
        return img_list
    
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        label_idx = class_to_idx.get(cls_name, 0)
        for f_path in sorted(glob.glob(os.path.join(cls_path, "**", "*"), recursive=True)):
            if f_path.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                img_list.append({
                    "ImageName":       os.path.abspath(f_path),
                    "Label":           label_idx,
                    "MaskPath":        None,
                    "Captions":        None,
                    "DatasetId":       1,
                    "CaptionsSentence": None,
                    "TabularInfo":     None,
                })
    return img_list



# ----------------------------------------------------------------------
# Classification: JSON update helpers
# ----------------------------------------------------------------------
def update_training_json(
    path: str, model_name: str, storage_root: str,
    class_lst: List[Dict], class_bin: List[Dict], class_prbwt: List[Dict],
    train_imgs: List[Dict], val_imgs: List[Dict], n_classes: int,
    training_type: str = "classification",
    rect_dims: Optional[Tuple[int, int]] = None,
    roi_type: Optional[str] = None,
) -> bool:
    data = load_json(path)
    if data is None:
        return False
    if training_type != "segmentation":
        data["classlst"] = class_lst
        data["classBin"]  = class_bin
        data["classPrbwt"] = class_prbwt
    data["trainImglst"] = train_imgs
    data["ValImgList"]  = val_imgs
    if "Model" in data:
        data["Model"]["iTrainImgCount"]      = len(train_imgs)
        data["Model"]["iValidationImgCount"] = len(val_imgs)
        if training_type != "segmentation":
            data["Model"]["iTotalClasses"]   = n_classes
        data["Model"]["SolutionDir"]         = str(storage_root) + os.sep
        data["Model"]["ModelDir"]            = "Train"
        data["Model"]["name"]                = model_name
        if rect_dims:
            data["Model"]["iRectWidth"]  = rect_dims[0]
            data["Model"]["iRectHeight"] = rect_dims[1]
            data["Model"]["iWidth"]      = rect_dims[0]
            data["Model"]["iHeight"]     = rect_dims[1]
        elif train_imgs:
            width, height = _read_image_dimensions(train_imgs[0]["ImageName"])
            if width and height:
                data["Model"]["iRectWidth"]  = width
                data["Model"]["iRectHeight"] = height
                data["Model"]["iWidth"]      = width
                data["Model"]["iHeight"]     = height
        if roi_type:
            data["Model"]["strROIType"] = roi_type
    return dump_json(data, path)


def update_testing_json(
    path: str, model_name: str,
    class_lst: List[Dict], class_bin: List[Dict], class_prbwt: List[Dict],
    test_imgs: List[Dict], n_classes: int,
    training_type: str = "classification",
    rect_dims: Optional[Tuple[int, int]] = None,
    roi_type: Optional[str] = None,
) -> bool:
    data = load_json(path)
    if data is None:
        return False
    if training_type != "segmentation":
        data["classlst"] = class_lst
        data["classBin"]  = []
        data["classPrbwt"] = class_prbwt
    data["testImglst"]  = test_imgs
    data["trainImglst"] = []
    data["ValImgList"]  = []
    if "Model" in data:
        data["Model"]["iTestImgCount"]       = len(test_imgs)
        if training_type != "segmentation":
            data["Model"]["iTotalClasses"]   = n_classes
        data["Model"]["iTrainImgCount"]      = 0
        data["Model"]["iValidationImgCount"] = 0
        data["Model"]["name"]                = model_name
        if rect_dims:
            data["Model"]["iRectWidth"]  = rect_dims[0]
            data["Model"]["iRectHeight"] = rect_dims[1]
            data["Model"]["iWidth"]      = rect_dims[0]
            data["Model"]["iHeight"]     = rect_dims[1]
        elif test_imgs:
            width, height = _read_image_dimensions(test_imgs[0]["ImageName"])
            if width and height:
                data["Model"]["iRectWidth"]  = width
                data["Model"]["iRectHeight"] = height
                data["Model"]["iWidth"]      = width
                data["Model"]["iHeight"]     = height
        if roi_type:
            data["Model"]["strROIType"] = roi_type
    return dump_json(data, path)


def update_evaluation_json(
    path: str, model_name: str,
    class_lst: List[Dict], class_bin: List[Dict], class_prbwt: List[Dict],
    eval_imgs: List[Dict], n_classes: int,
    training_type: str = "classification",
    rect_dims: Optional[Tuple[int, int]] = None,
    roi_type: Optional[str] = None,
) -> bool:
    data = load_json(path)
    if data is None:
        return False
    if training_type != "segmentation":
        data["classlst"] = class_lst
        data["classBin"]  = []
        data["classPrbwt"] = class_prbwt
    data["testImglst"]  = eval_imgs
    data["trainImglst"] = []
    data["ValImgList"]  = []
    data["valImgList"]  = []
    if "Model" in data:
        data["Model"]["iTrainImgCount"]      = len(eval_imgs)
        data["Model"]["iTestImgCount"]       = 0
        if training_type != "segmentation":
            data["Model"]["iTotalClasses"]   = n_classes
        data["Model"]["iValidationImgCount"] = 0
        data["Model"]["name"]                = model_name
        if rect_dims:
            data["Model"]["iRectWidth"]  = rect_dims[0]
            data["Model"]["iRectHeight"] = rect_dims[1]
            data["Model"]["iWidth"]      = rect_dims[0]
            data["Model"]["iHeight"]     = rect_dims[1]
        elif eval_imgs:
            width, height = _read_image_dimensions(eval_imgs[0]["ImageName"])
            if width and height:
                data["Model"]["iRectWidth"]  = width
                data["Model"]["iRectHeight"] = height
                data["Model"]["iWidth"]      = width
                data["Model"]["iHeight"]     = height
        if roi_type:
            data["Model"]["strROIType"] = roi_type
    return dump_json(data, path)

def csv_scan_images(
    csv_path:str,
    class_to_idx: Dict[str, int],
    label:str
) -> List[Dict]:
    df=pd.read_csv(csv_path)
    base_train_img_list = []
    base_val_img_list = []
    base_test_img_list = []

    new_train_img_list = []
    new_val_img_list = []
    new_test_img_list = []

    for row in df.itertuples(index=True):
        if "cycle" not in label:
            if row.Data_type in ["Base", "GAN"]:
                label_idx = class_to_idx.get(row.class_label, 0)
                if row.set=="Train":  
                    base_train_img_list.append(
                    {
                        "ImageName":row.image_url, 
                        "Label": label_idx,
                        "MaskPath": None,
                        "Captions": None,
                        "DatasetId": 1,
                        "CaptionsSentence": None,
                        "TabularInfo": None,
                    }
                                )
                elif row.set=="Val":
                    base_val_img_list.append(
                    {
                        "ImageName":row.image_url, 
                        "Label": label_idx,
                        "MaskPath": None,
                        "Captions": None,
                        "DatasetId": 1,
                        "CaptionsSentence": None,
                        "TabularInfo": None,
                    }
                    )
                else:
                    base_test_img_list.append(
                    {
                        "ImageName":row.image_url, 
                        "Label": label_idx,
                        "MaskPath": None,
                        "Captions": None,
                        "DatasetId": 1,
                        "CaptionsSentence": None,
                        "TabularInfo": None,
                    }
                    )
        else:
            if row.Data_type =="Base":
                label_idx = class_to_idx.get(row.class_label, 0)
                if row.set=="Train":  
                    base_train_img_list.append(
                    {
                        "ImageName":row.image_url, 
                        "Label": label_idx,
                        "MaskPath": None,
                        "Captions": None,
                        "DatasetId": 1,
                        "CaptionsSentence": None,
                        "TabularInfo": None,
                    }
                                )
                elif row.set=="Val":
                    base_val_img_list.append(
                    {
                        "ImageName":row.image_url, 
                        "Label": label_idx,
                        "MaskPath": None,
                        "Captions": None,
                        "DatasetId": 1,
                        "CaptionsSentence": None,
                        "TabularInfo": None,
                    }
                    )
                else:
                    base_test_img_list.append(
                    {
                        "ImageName":row.image_url, 
                        "Label": label_idx,
                        "MaskPath": None,
                        "Captions": None,
                        "DatasetId": 1,
                        "CaptionsSentence": None,
                        "TabularInfo": None,
                    }
                    )

        if row.Data_type=="New":
            label_idx = class_to_idx.get(row.class_label, 0)
            if row.set=="Train":  
                new_train_img_list.append(
                {
                    "ImageName":row.image_url, 
                    "Label": label_idx,
                    "MaskPath": None,
                    "Captions": None,
                    "DatasetId": 1,
                    "CaptionsSentence": None,
                    "TabularInfo": None,
                }
                            )
            elif row.set=="Val":
                new_val_img_list.append(
                {
                    "ImageName":row.image_url, 
                    "Label": label_idx,
                    "MaskPath": None,
                    "Captions": None,
                    "DatasetId": 1,
                    "CaptionsSentence": None,
                    "TabularInfo": None,
                }
                )
            else:
                new_test_img_list.append(
                {
                    "ImageName":row.image_url, 
                    "Label": label_idx,
                    "MaskPath": None,
                    "Captions": None,
                    "DatasetId": 1,
                    "CaptionsSentence": None,
                    "TabularInfo": None,
                }
                )

    return base_train_img_list, base_val_img_list, base_test_img_list, new_train_img_list, new_val_img_list, new_test_img_list


def csv_scan_dataset_and_update_configs(
    dataset_csv_path:str,
    storage_root: Optional[str] = None,
    label=None
) -> bool:

    logger.info("Scanning dataset and updating JSON configurations...")
    df= pd.read_csv(dataset_csv_path)
    # train_dir, val_dir, test_dir = resolve_dataset_paths(custom_dataset_path)
    # classes = df['class_label'].unique().tolist()
    # classes.sort()
    # # classes = discover_classes(train_dir)
    # if not classes:
    #     logger.warning("No classes found in Train directory. Skipping JSON update.")
    #     return False
    imported_json = os.path.join(storage_root, "Imported_model", "Training.json")
    if os.path.exists(imported_json):
        with open(imported_json) as f:
            source_data = json.load(f)
        
        classes = [item['ClassName'] for item in source_data['classlst']]
    else:
        classes = sorted(df['class_label'].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    ok_class = df.loc[df["class_type"] == "OK", "class_label"].iloc[0]
    class_lst, class_bin = build_class_lists(classes, ok_class)
    class_prbwt = [{"ClassName": item['ClassName'],"ClassProb": 0.0} for item in class_lst]

    rect_dims = _read_imported_rect_dims(storage_root)
    roi_type = _read_imported_roi_type(storage_root)

    print(class_to_idx)
    base_train_img_list, base_val_img_list, base_test_img_list, new_train_img_list, new_val_img_list, new_test_img_list = csv_scan_images(dataset_csv_path, class_to_idx, label)
    train_imgs = base_train_img_list+new_train_img_list
    val_imgs = base_val_img_list + new_val_img_list
    test_imgs = base_test_img_list + new_test_img_list


    logger.info(
        f"Found {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images."
    )
    logger.info(f"Classes identified: {classes}")

    model_name = get_latest_model_name(storage_root)
    paths_dict = get_model_paths(storage_root, model_name)

    training_json_path   = paths_dict["training_json"]
    testing_json_path    = paths_dict["testing_json"]
    evaluation_json_path = paths_dict["evaluation_json"]
    base_dir = BASE_DIR
    if not ensure_json_from_template(training_json_path, "Training.json", base_dir):
        return False

    if not ensure_json_from_template(testing_json_path, "Testing.json", base_dir):
        return False

    if not ensure_json_from_template(evaluation_json_path, "Evaluation.json", base_dir):
        return False

    n_classes = len(classes)

    if not update_training_json(
        training_json_path,
        model_name,
        storage_root,
        class_lst,
        class_bin,
        class_prbwt,
        train_imgs,
        val_imgs,
        n_classes,
        rect_dims=rect_dims,
        roi_type=roi_type,
    ):
        return False


    if not update_testing_json(
        testing_json_path,
        model_name,
        class_lst,
        class_bin,
        class_prbwt,
        test_imgs,
        n_classes,
        rect_dims=rect_dims,
        roi_type=roi_type,
    ):
        return False

    eval_imgs = train_imgs + val_imgs
    if not update_evaluation_json(
        evaluation_json_path,
        model_name,
        class_lst,
        class_bin,
        class_prbwt,
        eval_imgs,
        n_classes,
        rect_dims=rect_dims,
        roi_type=roi_type,
    ):
        return False

    logger.info("All JSON configurations have been updated successfully.")
    return True
def inf_csv_scan_dataset_and_update_configs(
    dataset_csv_path:str,
    storage_root: Optional[str] = None,
    label=None
) -> bool:

    logger.info("Scanning dataset and updating JSON configurations...")
    df= pd.read_csv(dataset_csv_path)
    # train_dir, val_dir, test_dir = resolve_dataset_paths(custom_dataset_path)
    # classes = df['class_label'].unique().tolist()
    # classes.sort()
    # classes = discover_classes(train_dir)
    imported_json = os.path.join(storage_root, "Imported_model", "Training.json")
    if os.path.exists(imported_json):
        with open(imported_json) as f:
            source_data = json.load(f)
        classes = [item['ClassName'] for item in source_data['classlst']]
    else:
        classes = sorted(df['class_label'].unique().tolist())
    if not classes:
        logger.warning("No classes found. Skipping JSON update.")
        return False

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    ok_class = df.loc[df["class_type"] == "OK", "class_label"].iloc[0]
    class_lst, class_bin = build_class_lists(classes, ok_class)
    class_prbwt = [{"ClassName": item['ClassName'],"ClassProb": 0.0} for item in class_lst]

    rect_dims = _read_imported_rect_dims(storage_root)
    roi_type = _read_imported_roi_type(storage_root)

    base_train_img_list, base_val_img_list, base_test_img_list, new_train_img_list, new_val_img_list, new_test_img_list = csv_scan_images(dataset_csv_path, class_to_idx,label)
    print('************************************ ',label)

    print(len(base_train_img_list), len(base_val_img_list), len(base_test_img_list), len(new_train_img_list),
          len(new_val_img_list), len(new_test_img_list) )

    if "Deployed" in label:
        if "Baseline" in label:
            train_imgs = base_train_img_list
            val_imgs = base_val_img_list
            test_imgs = base_test_img_list
        elif "New_Test" in label:
            train_imgs = new_train_img_list
            val_imgs = new_val_img_list
            test_imgs =new_train_img_list + new_val_img_list + new_test_img_list
    else:
        if "Baseline" in label:
            train_imgs = base_train_img_list
            val_imgs = base_val_img_list
            test_imgs = base_test_img_list + new_test_img_list
        elif "New_Test" in label:
            train_imgs = new_train_img_list
            val_imgs = new_val_img_list
            test_imgs =new_train_img_list + new_val_img_list + new_test_img_list

    logger.info(
        f"Found {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images."
    )
    logger.info(f"Classes identified: {classes}")

    model_name = get_latest_model_name(storage_root)
    paths_dict = get_model_paths(storage_root, model_name)

    training_json_path   = paths_dict["training_json"]
    testing_json_path    = paths_dict["testing_json"]
    evaluation_json_path = paths_dict["evaluation_json"]
    base_dir = BASE_DIR
    if not ensure_json_from_template(training_json_path, "Training.json", base_dir):
        return False

    if not ensure_json_from_template(testing_json_path, "Testing.json", base_dir):
        return False

    if not ensure_json_from_template(evaluation_json_path, "Evaluation.json", base_dir):
        return False

    n_classes = len(classes)

    if not update_training_json(
        training_json_path,
        model_name,
        storage_root,
        class_lst,
        class_bin,
        class_prbwt,
        train_imgs,
        val_imgs,
        n_classes,
        rect_dims=rect_dims,
        roi_type=roi_type,
    ):
        return False


    if not update_testing_json(
        testing_json_path,
        model_name,
        class_lst,
        class_bin,
        class_prbwt,
        test_imgs,
        n_classes,
        rect_dims=rect_dims,
        roi_type=roi_type,
    ):
        return False

    eval_imgs = train_imgs + val_imgs
    if not update_evaluation_json(
        evaluation_json_path,
        model_name,
        class_lst,
        class_bin,
        class_prbwt,
        eval_imgs,
        n_classes,
        rect_dims=rect_dims,
        roi_type=roi_type,
    ):
        return False

    logger.info("All JSON configurations have been updated successfully.")
    return True


def csv_scan_seg_images(csv_path: str) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    df = pd.read_csv(csv_path)
    base_train, base_val, base_test = [], [], []
    new_train, new_val, new_test = [], [], []

    for row in df.itertuples(index=True):
        mask_path = row.model_mask_url if hasattr(row, 'model_mask_url') and pd.notna(row.model_mask_url) else None
        entry = {
            "ImageName":        row.image_url,
            "Label":            1,
            "MaskPath":         mask_path,
            "Captions":         None,
            "DatasetId":        1,
            "CaptionsSentence": None,
            "TabularInfo":      None,
        }
        if row.Data_type == "Base":
            if row.set == "Train":
                base_train.append(entry)
            elif row.set == "Val":
                base_val.append(entry)
            else:
                base_test.append(entry)
        else:
            if row.set == "Train":
                new_train.append(entry)
            elif row.set == "Val":
                new_val.append(entry)
            else:
                new_test.append(entry)
            # New data has no Test set; New_Test inference uses new_train + new_val

    return base_train, base_val, base_test, new_train, new_val, new_test


def csv_scan_dataset_and_update_configs_seg(
    dataset_csv_path: str,
    storage_root: str,
    hyperparams: Optional[Dict] = None,
) -> Dict:
    logger.info("Scanning segmentation dataset from CSV and updating JSON configurations...")
    if hyperparams is None:
        hyperparams = {}

    _fail = {"success": False, "has_test": False}

    base_train, base_val, base_test, new_train, new_val, new_test = csv_scan_seg_images(dataset_csv_path)
    # process_logger.info(
    #     f"NODE:seg_scan, Base: {len(base_train)} train / {len(base_val)} val / {len(base_test)} test | "
    #     f"New: {len(new_train)} train / {len(new_val)} val / {len(new_test)} test"
    # )

    train_imgs = base_train + new_train
    val_imgs   = base_val  + new_val
    test_imgs  = base_test
    has_test   = len(test_imgs) > 0

    if not val_imgs and train_imgs:
        val_ratio = hyperparams.get("valRatio", 20) / 100.0
        split_idx = max(1, int(len(train_imgs) * (1 - val_ratio)))
        val_imgs   = train_imgs[split_idx:]
        train_imgs = train_imgs[:split_idx]
        logger.info(f"No Val data in CSV; auto-split → {len(train_imgs)} train, {len(val_imgs)} val.")

    model_name = get_latest_model_name(storage_root)
    paths_dict = get_model_paths(storage_root, model_name)
    train_json = paths_dict["training_json"]
    test_json  = paths_dict["testing_json"]
    eval_json  = paths_dict["evaluation_json"]

    # Training.json: keep as-is if it already has imported model params.
    if not ensure_json_from_template(train_json, "Training.json", SEGMENTATION_TEMPLATE_DIR):
        return _fail

    # Evaluation.json and Testing.json: always overwrite from the segmentation template
    # so they are never derived from Training.json content.
    for target, name in [(eval_json, "Evaluation.json"), (test_json, "Testing.json")]:
        candidates = [
            os.path.join(SEGMENTATION_TEMPLATE_DIR, name.lower()),
            os.path.join(SEGMENTATION_TEMPLATE_DIR, name),
        ]
        src = next((c for c in candidates if os.path.exists(c)), None)
        if src is None:
            logger.error(f"Segmentation template {name} not found in {SEGMENTATION_TEMPLATE_DIR}")
            return _fail
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(src, target)

    logger.info(f"Seg CSV: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images.")

    rect_dims = _read_imported_rect_dims(storage_root)
    roi_type = _read_imported_roi_type(storage_root)

    if not update_seg_training_json(train_json, storage_root, train_imgs, val_imgs, hyperparams, rect_dims=rect_dims, roi_type=roi_type):
        return _fail

    if not update_seg_evaluation_json(eval_json, storage_root, val_imgs, hyperparams, rect_dims=rect_dims, roi_type=roi_type):
        return _fail

    if has_test:
        if not update_seg_testing_json(test_json, storage_root, test_imgs, hyperparams, rect_dims=rect_dims, roi_type=roi_type):
            return _fail

    setup_seg_dirs(storage_root, model_name)
    logger.info("Segmentation CSV JSON configurations updated successfully.")
    return {"success": True, "has_test": has_test}


def inf_csv_scan_dataset_and_update_configs_seg(
    dataset_csv_path: str,
    storage_root: str,
    label=None,
) -> Dict:
    logger.info("Scanning segmentation dataset from CSV for inference...")

    _fail = {"success": False, "has_test": False}

    base_train, base_val, base_test, new_train, new_val, new_test = csv_scan_seg_images(dataset_csv_path)

    print('**** Seg inf label: ', label)
    print(len(base_train), len(base_val), len(base_test), len(new_train), len(new_val), len(new_test))
    if "Deployed" in label:
        if "Baseline" in label:
            train_imgs = base_train
            val_imgs   = base_val
            test_imgs  = base_test
        elif "New_Test" in label:
            train_imgs = new_train
            val_imgs   = new_val
            test_imgs  = new_train + new_val + new_test
    else:
        if "Baseline" in label:
            train_imgs = base_train
            val_imgs   = base_val
            test_imgs  = base_test + new_test
        elif "New_Test" in label:
            train_imgs = new_train
            val_imgs   = new_val
            test_imgs  = new_train + new_val + new_test
            
    has_test = len(test_imgs) > 0

    model_name = get_latest_model_name(storage_root)
    paths_dict = get_model_paths(storage_root, model_name)
    train_json = paths_dict["training_json"]
    test_json  = paths_dict["testing_json"]
    eval_json  = paths_dict["evaluation_json"]

    # Training.json: keep the imported model's config (has correct model params).
    if not ensure_json_from_template(train_json, "Training.json", SEGMENTATION_TEMPLATE_DIR):
        return _fail

    # Evaluation.json and Testing.json: always overwrite from the segmentation template.
    # The imported model export only contains Training.json, so frontier_changes_ may have
    # written these files from a wrong source (e.g. Training.json content). Force the
    # correct segmentation templates here so update_seg_*_json works on the right base.
    for target, name in [(eval_json, "Evaluation.json"), (test_json, "Testing.json")]:
        candidates = [
            os.path.join(SEGMENTATION_TEMPLATE_DIR, name.lower()),
            os.path.join(SEGMENTATION_TEMPLATE_DIR, name),
        ]
        src = next((c for c in candidates if os.path.exists(c)), None)
        if src is None:
            logger.error(f"Segmentation template {name} not found in {SEGMENTATION_TEMPLATE_DIR}")
            return _fail
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(src, target)

    hyperparams: Dict = {}

    rect_dims = _read_imported_rect_dims(storage_root)
    roi_type = _read_imported_roi_type(storage_root)

    if not update_seg_training_json(train_json, storage_root, train_imgs, val_imgs, hyperparams, rect_dims=rect_dims, roi_type=roi_type):
        return _fail

    # Eval uses val images (distinct from test); fall back to test if val is empty (e.g. New_Test with no new val rows)
    eval_imgs = val_imgs if val_imgs else test_imgs
    if not update_seg_evaluation_json(eval_json, storage_root, eval_imgs, hyperparams, rect_dims=rect_dims, roi_type=roi_type):
        return _fail

    if has_test:
        if not update_seg_testing_json(test_json, storage_root, test_imgs, hyperparams, rect_dims=rect_dims, roi_type=roi_type):
            return _fail

    setup_seg_dirs(storage_root, model_name)
    logger.info("Segmentation CSV inference JSON configurations updated successfully.")
    return {"success": True, "has_test": has_test}


def scan_dataset_and_update_configs(
    custom_dataset_path: Optional[str] = None,
    storage_root: Optional[str] = None,
) -> bool:
    """Classification: scan dataset, populate Training/Testing/Evaluation JSONs."""
    logger.info("Scanning classification dataset and updating JSON configurations...")

    train_dir, val_dir, test_dir = resolve_dataset_paths(custom_dataset_path)
    classes = discover_classes(train_dir)
    if not classes:
        logger.warning("No classes found in Train directory. Skipping JSON update.")
        return False

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    class_lst, class_bin = build_class_lists(classes)
    
    class_prbwt = [{"ClassName": item['ClassName'],"ClassProb": 0.0} for item in class_lst]
    train_imgs = scan_images(train_dir, classes, class_to_idx)
    val_imgs   = scan_images(val_dir,   classes, class_to_idx)
    test_imgs  = scan_images(test_dir,  classes, class_to_idx)

    logger.info(f"Found {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images.")
    logger.info(f"Classes identified: {classes}")

    model_name   = get_latest_model_name(storage_root)
    paths_dict   = get_model_paths(storage_root, model_name)
    train_json   = paths_dict["training_json"]
    test_json    = paths_dict["testing_json"]
    eval_json    = paths_dict["evaluation_json"]

    for target, name in [(train_json, "Training.json"), (test_json, "Testing.json"), (eval_json, "Evaluation.json")]:
        if not ensure_json_from_template(target, name, CLASSIFICATION_TEMPLATE_DIR):
            return False

    n_classes = len(classes)
    if not update_training_json(train_json, model_name, storage_root or "", class_lst, class_bin, class_prbwt, train_imgs, val_imgs, n_classes):
        return False
    if not update_testing_json(test_json, model_name, class_lst, class_bin, class_prbwt, test_imgs, n_classes):
        return False
    eval_imgs = train_imgs + val_imgs
    if not update_evaluation_json(eval_json, model_name, class_lst, class_bin, class_prbwt, eval_imgs, n_classes):
        return False

    logger.info("Classification JSON configurations updated successfully.")
    return True


# ----------------------------------------------------------------------
# Segmentation: dataset scanning helpers
# ----------------------------------------------------------------------
_IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def scan_seg_images(split_dir: str, masks_dir: str) -> List[Dict]:
    """
    Scan split_dir/<class>/ for images and match each against masks_dir/ by base filename.
    Dataset layout:
        BaseDataset/Train (or Val/Test)/<class>/image.jpg
        BaseDataset/Masks/image.jpg   ← shared across all splits
    Class folders with no images are silently skipped.
    Returns list of dicts with MaskPath populated.
    """
    imgs = []
    if not os.path.isdir(split_dir):
        return imgs

    mask_lookup = {}
    if os.path.isdir(masks_dir):
        mask_lookup = {
            os.path.splitext(f)[0]: os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
        }

    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_file in sorted(os.listdir(class_dir)):
            if os.path.splitext(img_file)[1].lower() not in _IMAGE_EXTS:
                continue
            base = os.path.splitext(img_file)[0]
            mask_val = mask_lookup.get(base)
            imgs.append({
                "ImageName":        os.path.abspath(os.path.join(class_dir, img_file)),
                "Label":            1,
                "MaskPath":         os.path.abspath(mask_val) if mask_val else None,
                "Captions":         None,
                "DatasetId":        0,
                "CaptionsSentence": None,
                "TabularInfo":      None,
            })
    return imgs


def _read_imported_rect_dims(storage_root: str) -> Optional[Tuple[int, int]]:
    """Return (iRectWidth, iRectHeight) from Imported_model/Training.json, or None if not present."""
    imported_json = os.path.join(storage_root, "Imported_model", "Training.json")
    if not os.path.exists(imported_json):
        return None
    try:
        with open(imported_json, "r", encoding="utf-8") as f:
            src = json.load(f)
        m = src.get("Model", {})
        w = m.get("iRectWidth")
        h = m.get("iRectHeight")
        if w and h:
            return (int(w), int(h))
    except Exception as exc:
        logger.warning(f"Could not read dims from imported model JSON: {exc}")
    return None


def _read_imported_roi_type(storage_root: str) -> Optional[str]:
    """Return strROIType (resize/center-crop/etc.) from Imported_model/Training.json, or None if not present."""
    imported_json = os.path.join(storage_root, "Imported_model", "Training.json")
    if not os.path.exists(imported_json):
        return None
    try:
        with open(imported_json, "r", encoding="utf-8") as f:
            src = json.load(f)
        roi_type = src.get("Model", {}).get("strROIType")
        if roi_type:
            return roi_type
    except Exception as exc:
        logger.warning(f"Could not read strROIType from imported model JSON: {exc}")
    return None


def _read_image_dimensions(img_path: str) -> Tuple[int, int]:
    """Return (width, height) of the image, or (0, 0) on failure."""
    try:
        with PILImage.open(img_path) as img:
            return img.size  # (width, height)
    except Exception as exc:
        logger.warning(f"Could not read image dimensions from {img_path}: {exc}")
        return 0, 0


def setup_seg_dirs(storage_root: str, model_name: str) -> None:
    """Create directory structure needed by the segmentation EXE."""
    Path(os.path.join(storage_root, "Train", model_name)).mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.join(storage_root, "Test", model_name, "MaskImages"), exist_ok=True)

    model_config_src = os.path.join(SEGMENTATION_TEMPLATE_DIR, "ModelConfig")
    if os.path.isdir(model_config_src):
        for mc_dst in [
            os.path.join(storage_root, "Train", model_name, "ModelConfig"),
            os.path.join(storage_root, "Test",  model_name, "ModelConfig"),
        ]:
            shutil.copytree(model_config_src, mc_dst, dirs_exist_ok=True)

    # Always ensure net.json is flat in ModelConfig/ — DLProcessWrapper requires
    # Train/{model}/ModelConfig/net.json but templates place it in a ResUNet14/ subdir.
    for mc_dst in [
        os.path.join(storage_root, "Train", model_name, "ModelConfig"),
        os.path.join(storage_root, "Test",  model_name, "ModelConfig"),
    ]:
        if not os.path.isdir(mc_dst):
            continue
        flat_net = os.path.join(mc_dst, "net.json")
        if not os.path.exists(flat_net):
            for sub in os.listdir(mc_dst):
                candidate = os.path.join(mc_dst, sub, "net.json")
                if os.path.isfile(candidate):
                    shutil.copy2(candidate, flat_net)
                    break


# ----------------------------------------------------------------------
# Segmentation: JSON update helpers
# ----------------------------------------------------------------------
def _apply_seg_model_fields(data: Dict, storage_root: str, train_count: int,
                             val_count: int, test_count: int, hyperparams: Dict,
                             train_imgs: List[Dict], model_dir: str = "Train",
                             rect_dims: Optional[Tuple[int, int]] = None,
                             roi_type: Optional[str] = None) -> None:
    """Mutate data['Model'] with segmentation-specific fields."""
    if "Model" not in data:
        return
    m = data["Model"]
    m["iTrainImgCount"]      = train_count
    m["iValidationImgCount"] = val_count
    m["iTestImgCount"]       = test_count
    m["SolutionDir"]         = str(storage_root) + "//"
    m["tfrec_lmdb_path"]     = str(storage_root) + "//"
    m["ModelDir"]            = model_dir
    m["name"]                = get_latest_model_name(storage_root)
    m["epochs"]              = hyperparams.get("epochs",   _DEFAULT_HPARAMS.get("epochs",   100))
    m["valRatio"]            = hyperparams.get("valRatio",   _DEFAULT_HPARAMS.get("valRatio",  20))
    m["minEpoch"]            = hyperparams.get("minEpoch",   _DEFAULT_HPARAMS.get("minEpoch",   0))
    m["patience"]            = hyperparams.get("patience",   _DEFAULT_HPARAMS.get("patience",  50))

    # Image dimensions: prefer imported model dims, then auto-detect, then leave template default
    if rect_dims:
        m["iRectWidth"]  = rect_dims[0]
        m["iRectHeight"] = rect_dims[1]
        m["iWidth"]      = rect_dims[0]
        m["iHeight"]     = rect_dims[1]
    elif train_imgs:
        width, height = _read_image_dimensions(train_imgs[0]["ImageName"])
        if width and height:
            m["iRectWidth"]  = width
            m["iRectHeight"] = height
            m["iWidth"]      = width
            m["iHeight"]     = height

    if roi_type:
        m["strROIType"] = roi_type


def update_seg_training_json(path: str, storage_root: str,
                              train_imgs: List[Dict], val_imgs: List[Dict],
                              hyperparams: Dict,
                              rect_dims: Optional[Tuple[int, int]] = None,
                              roi_type: Optional[str] = None) -> bool:
    data = load_json(path)
    if data is None:
        return False
    _apply_seg_model_fields(data, storage_root, len(train_imgs), len(val_imgs), 0, hyperparams, train_imgs, rect_dims=rect_dims, roi_type=roi_type)
    data["trainImglst"] = train_imgs
    data["ValImgList"]  = val_imgs
    data["testImglst"]  = None
    return dump_json(data, path)


def update_seg_testing_json(path: str, storage_root: str,
                             test_imgs: List[Dict], hyperparams: Dict,
                             rect_dims: Optional[Tuple[int, int]] = None,
                             roi_type: Optional[str] = None) -> bool:
    data = load_json(path)
    if data is None:
        return False
    _apply_seg_model_fields(data, storage_root, 0, 0, len(test_imgs), hyperparams, test_imgs, model_dir="Test", rect_dims=rect_dims, roi_type=roi_type)
    data["trainImglst"] = None
    data["ValImgList"]  = None
    data["testImglst"]  = test_imgs
    return dump_json(data, path)


def update_seg_evaluation_json(path: str, storage_root: str,
                                eval_imgs: List[Dict], hyperparams: Dict,
                                rect_dims: Optional[Tuple[int, int]] = None,
                                roi_type: Optional[str] = None) -> bool:
    data = load_json(path)
    if data is None:
        return False
    _apply_seg_model_fields(data, storage_root, len(eval_imgs), 0, 0, hyperparams, eval_imgs, rect_dims=rect_dims, roi_type=roi_type)
    data["trainImglst"] = None
    data["ValImgList"]  = None
    data["testImglst"]  = eval_imgs
    return dump_json(data, path)


def scan_dataset_and_update_configs_seg(
    custom_dataset_path: str,
    storage_root: str,
    hyperparams: Optional[Dict] = None,
) -> Dict:
    """
    Segmentation: scan dataset, populate Training/Testing/Evaluation JSONs.
    Returns {"success": bool, "has_test": bool}.
    When no Test folder or test images exist, testing JSON is skipped and
    evaluation falls back to val images.
    """
    logger.info("Scanning segmentation dataset and updating JSON configurations...")
    if hyperparams is None:
        hyperparams = {}

    _fail = {"success": False, "has_test": False}

    train_dir  = os.path.join(custom_dataset_path, "Train")
    val_dir    = next(
        (os.path.join(custom_dataset_path, name)
         for name in ("Val", "Valid", "val", "valid")
         if os.path.isdir(os.path.join(custom_dataset_path, name))),
        os.path.join(custom_dataset_path, "Val"),
    )
    test_dir   = os.path.join(custom_dataset_path, "Test")
    masks_dir  = os.path.join(custom_dataset_path, "Masks")

    model_name = get_latest_model_name(storage_root)
    paths_dict = get_model_paths(storage_root, model_name)
    train_json = paths_dict["training_json"]
    test_json  = paths_dict["testing_json"]
    eval_json  = paths_dict["evaluation_json"]

    if not ensure_json_from_template(train_json, "Training.json", SEGMENTATION_TEMPLATE_DIR):
        return _fail

    for target, name in [(eval_json, "Evaluation.json"), (test_json, "Testing.json")]:
        candidates = [
            os.path.join(SEGMENTATION_TEMPLATE_DIR, name.lower()),
            os.path.join(SEGMENTATION_TEMPLATE_DIR, name),
        ]
        src = next((c for c in candidates if os.path.exists(c)), None)
        if src is None:
            logger.error(f"Segmentation template {name} not found in {SEGMENTATION_TEMPLATE_DIR}")
            return _fail
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(src, target)

    train_imgs = scan_seg_images(train_dir, masks_dir)
    val_imgs   = scan_seg_images(val_dir,   masks_dir)
    test_imgs  = scan_seg_images(test_dir,  masks_dir)
    has_test   = len(test_imgs) > 0

    if not val_imgs and train_imgs:
        val_ratio = hyperparams.get("valRatio", 20) / 100.0
        split_idx = max(1, int(len(train_imgs) * (1 - val_ratio)))
        val_imgs   = train_imgs[split_idx:]
        train_imgs = train_imgs[:split_idx]
        logger.info(f"No Val folder found; auto-split train → {len(train_imgs)} train, {len(val_imgs)} val.")

    logger.info(f"Seg: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images.")
    if not has_test:
        logger.info("No Test folder or test images found; evaluation will use val images.")

    if not update_seg_training_json(train_json, storage_root, train_imgs, val_imgs, hyperparams):
        return _fail

    if not update_seg_evaluation_json(eval_json, storage_root, val_imgs, hyperparams):
        return _fail

    if has_test:
        if not update_seg_testing_json(test_json, storage_root, test_imgs, hyperparams):
            return _fail

    setup_seg_dirs(storage_root, model_name)
    logger.info("Segmentation JSON configurations updated successfully.")
    return {"success": True, "has_test": has_test}


# ----------------------------------------------------------------------
# Object Detection (YOLO OBB): dataset scanning + JSON update helpers
# ----------------------------------------------------------------------
# Unlike classification/segmentation, Training.json/Evaluation.json/Testing.json for
# object detection carry no embedded image list (trainImglst/ValImgList/testImglst stay
# null). The annotated image list instead lives in separate dataset-descriptor JSONs:
#   - d.json            : combined Train+Val annotations, next to Training.json/Evaluation.json
#   - Input_Train.json  : Train-only annotations, same folder
#   - Input_Val.json    : Val-only annotations, same folder
#   - d.json            : Test-only annotations, next to Testing.json (a *different* file
#                          of the same name, since it lives in the Test model folder)
# Evaluation.json/Testing.json point at their local d.json via Model.multi_dataset_names
# and the top-level Datasets list (both reference dataset name "d" == "d.json" resolved in
# the same directory as the JSON file itself). Training.json has no such pointer — it picks
# up Input_Train.json/Input_Val.json from its own folder by fixed filename.
#
# Points format (per image entry): a list of per-object dicts, one per line in the
# annotation .txt file —
#     {"CId": "<class index as string>", "X": [x1, x2, x3, x4], "Y": [y1, y2, y3, y4]}
# with X/Y holding the 4 OBB corner coordinates in absolute pixel space (not normalized).
# The source .txt referenced by model_bbox_url is a standard YOLO-OBB label file —
#     class_index x1 y1 x2 y2 x3 y3 x4 y4
# with all 8 coordinates normalized to [0, 1] relative to image width/height — so
# de-normalizing means x_abs = x_norm * ImgW, y_abs = y_norm * ImgH.
def _parse_yolo_obb_txt(txt_path: Optional[str], img_w, img_h) -> List[Dict]:
    """Parse a YOLO-OBB label .txt file into absolute-pixel Points entries."""
    points: List[Dict] = []
    if not txt_path or not os.path.exists(txt_path):
        return points
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 9:
                    continue
                cls_id = parts[0]
                xs_norm = [float(parts[i]) for i in (1, 3, 5, 7)]
                ys_norm = [float(parts[i]) for i in (2, 4, 6, 8)]
                points.append({
                    "CId": str(cls_id),
                    "X":   [round(x * img_w, 2) for x in xs_norm],
                    "Y":   [round(y * img_h, 2) for y in ys_norm],
                })
    except Exception as exc:
        logger.warning(f"Failed to parse OBB label file {txt_path}: {exc}")
    return points


def csv_scan_objdet_annotations(
    csv_path: str,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Read train_dataset.csv and build ANNOTATIONS-format entries (ImagePath/MaskPath/
    Rect/Points/ImgW/ImgH) split into base/new x train/val/test buckets, mirroring
    csv_scan_seg_images. `model_bbox_url` points at the YOLO-OBB label .txt for the image;
    its content is parsed and embedded as absolute-pixel Points (MaskPath itself is not
    carried into the output — only used here to locate the label file).
    """
    df = pd.read_csv(csv_path)
    base_train, base_val, base_test = [], [], []
    new_train, new_val, new_test = [], [], []

    for row in df.itertuples(index=True):
        bbox_path = row.model_bbox_url if hasattr(row, 'model_bbox_url') and pd.notna(row.model_bbox_url) else None
        width, height = _read_image_dimensions(row.image_url)
        entry = {
            "ImagePath": row.image_url,
            "MaskPath":  None,
            "Rect":      [],
            "Points":    _parse_yolo_obb_txt(bbox_path, width, height),
            "ImgW":      width,
            "ImgH":      height,
        }
        if row.Data_type == "Base":
            if row.set == "Train":
                base_train.append(entry)
            elif row.set == "Val":
                base_val.append(entry)
            else:
                base_test.append(entry)
        else:
            if row.set == "Train":
                new_train.append(entry)
            elif row.set == "Val":
                new_val.append(entry)
            else:
                new_test.append(entry)
            # New data has no Test set; New_Test inference uses new_train + new_val

    return base_train, base_val, base_test, new_train, new_val, new_test


def write_objdet_annotations_json(path: str, entries: List[Dict]) -> bool:
    """Write a dataset-descriptor JSON (d.json / Input_Train.json / Input_Val.json)."""
    return dump_json({"ANNOTATIONS": entries}, path)


def _apply_objdet_dataset_pointer(data: Dict, dataset_name: str) -> None:
    """Point Model.multi_dataset_names / top-level Datasets at <dataset_name>.json,
    resolved relative to the folder the JSON file itself lives in."""
    if "Model" in data:
        data["Model"]["multi_dataset_names"] = [dataset_name]
    data["Datasets"] = [{"Id": 1, "Name": dataset_name, "BatchRate": 1.0}]


def _apply_objdet_model_fields(data: Dict, storage_root: str, train_count: int,
                                val_count: int, test_count: int, hyperparams: Dict,
                                ref_imgs: List[Dict], model_dir: str = "Train",
                                rect_dims: Optional[Tuple[int, int]] = None,
                                roi_type: Optional[str] = None) -> None:
    """Mutate data['Model'] with object-detection-specific fields (counts, dims, hyperparams)."""
    if "Model" not in data:
        return
    m = data["Model"]
    m["iTrainImgCount"]      = train_count
    m["iValidationImgCount"] = val_count
    m["iTestImgCount"]       = test_count
    m["SolutionDir"]         = str(storage_root) + "//"
    m["tfrec_lmdb_path"]     = str(storage_root) + "//"
    m["ModelDir"]            = model_dir
    m["name"]                = get_latest_model_name(storage_root)
    m["epochs"]              = hyperparams.get("epochs",     _DEFAULT_HPARAMS.get("epochs",   100))
    m["valRatio"]            = hyperparams.get("valRatio",   _DEFAULT_HPARAMS.get("valRatio",  20))
    m["minEpoch"]            = hyperparams.get("minEpoch",   _DEFAULT_HPARAMS.get("minEpoch",   0))
    m["patience"]            = hyperparams.get("patience",   _DEFAULT_HPARAMS.get("patience",  50))

    # Image dimensions: prefer imported model dims, then the first annotated image, then template default
    if rect_dims:
        m["iRectWidth"]  = rect_dims[0]
        m["iRectHeight"] = rect_dims[1]
        m["iWidth"]      = rect_dims[0]
        m["iHeight"]     = rect_dims[1]
    elif ref_imgs:
        width, height = ref_imgs[0].get("ImgW"), ref_imgs[0].get("ImgH")
        if width and height:
            m["iRectWidth"]  = width
            m["iRectHeight"] = height
            m["iWidth"]      = width
            m["iHeight"]     = height

    if roi_type:
        m["strROIType"] = roi_type


def update_objdet_training_json(path: str, storage_root: str,
                                 train_imgs: List[Dict], val_imgs: List[Dict],
                                 hyperparams: Dict,
                                 rect_dims: Optional[Tuple[int, int]] = None,
                                 roi_type: Optional[str] = None) -> bool:
    data = load_json(path)
    if data is None:
        return False
    _apply_objdet_model_fields(data, storage_root, len(train_imgs), len(val_imgs), 0,
                                hyperparams, train_imgs or val_imgs, rect_dims=rect_dims, roi_type=roi_type)
    data["trainImglst"] = None
    data["ValImgList"]  = None
    data["testImglst"]  = None
    return dump_json(data, path)


def update_objdet_evaluation_json(path: str, storage_root: str,
                                   eval_imgs: List[Dict], hyperparams: Dict,
                                   rect_dims: Optional[Tuple[int, int]] = None,
                                   roi_type: Optional[str] = None) -> bool:
    data = load_json(path)
    if data is None:
        return False
    _apply_objdet_model_fields(data, storage_root, len(eval_imgs), 0, 0,
                                hyperparams, eval_imgs, rect_dims=rect_dims, roi_type=roi_type)
    _apply_objdet_dataset_pointer(data, "d")
    data["trainImglst"] = None
    data["ValImgList"]  = None
    data["testImglst"]  = None
    return dump_json(data, path)


def update_objdet_testing_json(path: str, storage_root: str,
                                test_imgs: List[Dict], hyperparams: Dict,
                                rect_dims: Optional[Tuple[int, int]] = None,
                                roi_type: Optional[str] = None) -> bool:
    data = load_json(path)
    if data is None:
        return False
    _apply_objdet_model_fields(data, storage_root, 0, 0, len(test_imgs),
                                hyperparams, test_imgs, model_dir="Test", rect_dims=rect_dims, roi_type=roi_type)
    _apply_objdet_dataset_pointer(data, "d")
    data["trainImglst"] = None
    data["ValImgList"]  = None
    data["testImglst"]  = None
    return dump_json(data, path)


def _ensure_objdet_templates(train_json: str, eval_json: str, test_json: str) -> bool:
    """Copy Training/Evaluation/Testing.json from the objectdetection templates.
    Training.json is kept as-is if already present (e.g. carries imported-model params);
    Evaluation.json/Testing.json are always overwritten so they're never derived from
    stale Training.json content."""
    if not ensure_json_from_template(train_json, "Training.json", OBJECTDETECTION_TRAIN_TEMPLATE_DIR):
        return False

    for target, name, tpl_dir in [
        (eval_json, "Evaluation.json", OBJECTDETECTION_TRAIN_TEMPLATE_DIR),
        (test_json, "Testing.json",    OBJECTDETECTION_TEST_TEMPLATE_DIR),
    ]:
        candidates = [os.path.join(tpl_dir, name.lower()), os.path.join(tpl_dir, name)]
        src = next((c for c in candidates if os.path.exists(c)), None)
        if src is None:
            logger.error(f"Object detection template {name} not found in {tpl_dir}")
            return False
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(src, target)

    return True


def _write_objdet_dataset_files(paths_dict: Dict, train_imgs: List[Dict], val_imgs: List[Dict],
                                 eval_imgs: List[Dict], test_imgs: List[Dict], has_test: bool) -> bool:
    """Write d.json/Input_Train.json/Input_Val.json next to Training.json — d.json mirrors
    whatever Evaluation.json's dataset pointer resolves to (eval_imgs) — and the test-side
    d.json next to Testing.json."""
    train_dir = paths_dict["train_dir"]
    test_dir  = paths_dict["test_dir"]
    os.makedirs(train_dir, exist_ok=True)

    if not write_objdet_annotations_json(os.path.join(train_dir, "d.json"), eval_imgs):
        return False
    if not write_objdet_annotations_json(os.path.join(train_dir, "Input_Train.json"), train_imgs):
        return False
    if not write_objdet_annotations_json(os.path.join(train_dir, "Input_Val.json"), val_imgs):
        return False

    if has_test:
        os.makedirs(test_dir, exist_ok=True)
        if not write_objdet_annotations_json(os.path.join(test_dir, "d.json"), test_imgs):
            return False

    return True


def csv_scan_dataset_and_update_configs_objdet(
    dataset_csv_path: str,
    storage_root: str,
    hyperparams: Optional[Dict] = None,
) -> Dict:
    """
    Object detection (YOLO OBB): scan train_dataset.csv and update Training/Evaluation/Testing
    JSONs plus the dataset descriptor files (d.json / Input_Train.json / Input_Val.json).
    Returns {"success": bool, "has_test": bool}.
    """
    logger.info("Scanning object detection dataset from CSV and updating JSON configurations...")
    if hyperparams is None:
        hyperparams = {}

    _fail = {"success": False, "has_test": False}

    base_train, base_val, base_test, new_train, new_val, new_test = csv_scan_objdet_annotations(dataset_csv_path)

    train_imgs = base_train + new_train
    val_imgs   = base_val + new_val
    test_imgs  = base_test
    has_test   = len(test_imgs) > 0

    if not val_imgs and train_imgs:
        val_ratio = hyperparams.get("valRatio", 20) / 100.0
        split_idx = max(1, int(len(train_imgs) * (1 - val_ratio)))
        val_imgs   = train_imgs[split_idx:]
        train_imgs = train_imgs[:split_idx]
        logger.info(f"No Val data in CSV; auto-split → {len(train_imgs)} train, {len(val_imgs)} val.")

    model_name = get_latest_model_name(storage_root)
    paths_dict = get_model_paths(storage_root, model_name)
    train_json = paths_dict["training_json"]
    test_json  = paths_dict["testing_json"]
    eval_json  = paths_dict["evaluation_json"]

    if not _ensure_objdet_templates(train_json, eval_json, test_json):
        return _fail

    logger.info(f"ObjDet CSV: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images.")

    rect_dims = _read_imported_rect_dims(storage_root)
    roi_type  = _read_imported_roi_type(storage_root)

    if not update_objdet_training_json(train_json, storage_root, train_imgs, val_imgs, hyperparams,
                                        rect_dims=rect_dims, roi_type=roi_type):
        return _fail

    eval_imgs = train_imgs + val_imgs
    if not update_objdet_evaluation_json(eval_json, storage_root, eval_imgs, hyperparams,
                                          rect_dims=rect_dims, roi_type=roi_type):
        return _fail

    if has_test:
        if not update_objdet_testing_json(test_json, storage_root, test_imgs, hyperparams,
                                           rect_dims=rect_dims, roi_type=roi_type):
            return _fail

    if not _write_objdet_dataset_files(paths_dict, train_imgs, val_imgs, eval_imgs, test_imgs, has_test):
        return _fail

    logger.info("Object detection CSV JSON configurations updated successfully.")
    return {"success": True, "has_test": has_test}


def inf_csv_scan_dataset_and_update_configs_objdet(
    dataset_csv_path: str,
    storage_root: str,
    label=None,
) -> Dict:
    """Object detection inference-time variant of csv_scan_dataset_and_update_configs_objdet —
    buckets Base/New_Test images the same way inf_csv_scan_dataset_and_update_configs_seg does."""
    logger.info("Scanning object detection dataset from CSV for inference...")

    _fail = {"success": False, "has_test": False}

    base_train, base_val, base_test, new_train, new_val, new_test = csv_scan_objdet_annotations(dataset_csv_path)

    if "Deployed" in label:
        if "Baseline" in label:
            train_imgs = base_train
            val_imgs   = base_val
            test_imgs  = base_test
        elif "New_Test" in label:
            train_imgs = new_train
            val_imgs   = new_val
            test_imgs  = new_train + new_val + new_test
    else:
        if "Baseline" in label:
            train_imgs = base_train
            val_imgs   = base_val
            test_imgs  = base_test + new_test
        elif "New_Test" in label:
            train_imgs = new_train
            val_imgs   = new_val
            test_imgs  = new_train + new_val + new_test

    has_test = len(test_imgs) > 0

    model_name = get_latest_model_name(storage_root)
    paths_dict = get_model_paths(storage_root, model_name)
    train_json = paths_dict["training_json"]
    test_json  = paths_dict["testing_json"]
    eval_json  = paths_dict["evaluation_json"]

    if not _ensure_objdet_templates(train_json, eval_json, test_json):
        return _fail

    hyperparams: Dict = {}

    rect_dims = _read_imported_rect_dims(storage_root)
    roi_type  = _read_imported_roi_type(storage_root)

    if not update_objdet_training_json(train_json, storage_root, train_imgs, val_imgs, hyperparams,
                                        rect_dims=rect_dims, roi_type=roi_type):
        return _fail

    # Eval uses train+val combined (matches d.json); fall back to test if both are empty
    # (e.g. New_Test with no new train/val rows).
    eval_imgs = (train_imgs + val_imgs) if (train_imgs or val_imgs) else test_imgs
    if not update_objdet_evaluation_json(eval_json, storage_root, eval_imgs, hyperparams,
                                          rect_dims=rect_dims, roi_type=roi_type):
        return _fail

    if has_test:
        if not update_objdet_testing_json(test_json, storage_root, test_imgs, hyperparams,
                                           rect_dims=rect_dims, roi_type=roi_type):
            return _fail

    if not _write_objdet_dataset_files(paths_dict, train_imgs, val_imgs, eval_imgs, test_imgs, has_test):
        return _fail

    logger.info("Object detection CSV inference JSON configurations updated successfully.")
    return {"success": True, "has_test": has_test}


# ----------------------------------------------------------------------
# DL Process Wrapper
# ----------------------------------------------------------------------
def generate_config_json(config_path, mode="Train", params=None, training_type="classification"):
    """Update hyperparameter fields in an existing JSON config for DLProcessWrapper."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        return False
    try:
        with open(config_path, 'r',encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return False

    if params and "Model" in config_data:
        if training_type == "segmentation":
            if "epochs"     in params: config_data["Model"]["epochs"]    = int(params["epochs"])
            if "valRatio"   in params: config_data["Model"]["valRatio"]  = int(params["valRatio"])
            if "minEpoch"   in params: config_data["Model"]["minEpoch"]  = int(params["minEpoch"])
            if "patience"   in params: config_data["Model"]["patience"]  = int(params["patience"])
            if "batch_size" in params: config_data["Model"]["iBatchSize"] = int(params["batch_size"])
            if "lr"         in params: config_data["Model"]["fBaseLR"]    = float(params["lr"])
        else:
            if "epochs"     in params: config_data["Model"]["epochs"]     = int(params["epochs"])
            if "batch_size" in params: config_data["Model"]["iBatchSize"] = int(params["batch_size"])
            if "lr"         in params: config_data["Model"]["fBaseLR"]    = float(params["lr"])

    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to write config file: {e}")
        return False


def parse_status_log(log_path):
    """Parse Status.txt and return the last epoch info dict, or None.

    Supports two formats:
      Classification:
        epoch: N, val_loss: X, train_loss: X, best_epoch: N, best_loss: X, val_acc: X%
      Segmentation:
        epoch: N, loss: X, acc: X, val_loss: X, val_acc: X, val_match: X, best_epoch: N, best_acc: X
    """
    if not os.path.exists(log_path):
        return None
    last_info = None

    # Classification regex
    _cls_re = re.compile(
        r"epoch:\s*(\d+),\s*val_loss:\s*([\d.]+),\s*train_loss:\s*([\d.]+),"
        r"\s*best_epoch:\s*(\d+),\s*best_loss:\s*([\d.]+),\s*val_acc:\s*([\d.]+?)%"
    )
    # Segmentation regex
    _seg_re = re.compile(
        r"epoch:\s*(\d+),\s*loss:\s*([\d.]+),\s*acc:\s*([\d.]+),\s*val_loss:\s*([\d.]+),"
        r"\s*val_acc:\s*([\d.]+).*?best_epoch:\s*(\d+),\s*best_acc:\s*([\d.]+)"
    )

    try:
        with open(log_path, 'r') as f:
            for line in f:
                m = _cls_re.search(line)
                if m:
                    last_info = {
                        "epoch":      int(m.group(1)),
                        "val_loss":   float(m.group(2)),
                        "train_loss": float(m.group(3)),
                        "best_epoch": int(m.group(4)),
                        "best_loss":  float(m.group(5)),
                        "val_acc":    float(m.group(6)),
                    }
                    continue
                m = _seg_re.search(line)
                if m:
                    last_info = {
                        "epoch":      int(m.group(1)),
                        "train_loss": float(m.group(2)),
                        "val_loss":   float(m.group(4)),
                        "val_acc":    float(m.group(5)),
                        "best_epoch": int(m.group(6)),
                        "best_acc":   float(m.group(7)),
                        "best_loss":  float(m.group(4)),  # use val_loss as proxy
                    }
    except Exception as e:
        logger.error(f"Error parsing status log: {e}")
    return last_info


def monitor_log(path, status):
    log_path = os.path.join(os.path.dirname(path), "Status.txt")
    dl_path  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))), "SEMDLTF.log")
    while not os.path.exists(log_path):
        time.sleep(0.5)

    with open(log_path, "r") as f:
        while True:
            with open(dl_path, "r") as file:
                ln = file.readlines()
                if ln and ("ERROR" in ln[-1].strip()):
                    print("breaking monitor thread....")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        import tensorflow as tf
                        tf.keras.backend.clear_session()
                    except Exception:
                        pass
                    process_logger.error(
                        f"NODE:88, Something went wrong please check the frontier log - {dl_path}"
                    )
                    os._exit(1)
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            line = line.strip()
            process_logger.info(f"NODE:{status}, {line.split(']')[-1]}")
            if "End Learning" in line:
                break


def run_dl_process_wrapper(config_path, status, mode="Train"):
    """
    Execute DLProcessWrapper.exe with the given config and mode string.
    For classification    : mode in {"Train", "Evaluate", "Test"}
    For segmentation      : mode in {"segTrainBinary", "segEvaluateBinary", "segTestBinary"}
    For object detection  : mode in {"ODMTrain", "ODMEvaluate", "ODMTest"}
    """
    exe_path = DL_PROCESS_WRAPPER_PATH
    if not os.path.exists(exe_path):
        logger.error(f"DLProcessWrapper.exe not found at {exe_path}.")
        return False

    cmd = [exe_path, config_path, mode]
    logger.info(f"Executing: {' '.join(cmd)}")

    try:
        is_training_mode = mode in ("Train", "segTrainBinary", "ODMTrain")
        if is_training_mode and status != "trial":
            import threading
            t = threading.Thread(target=monitor_log, args=(config_path, status))
            t.start()
            time.sleep(2)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 1:
            logger.error(f"DLProcessWrapper failed with code {process.returncode}")
            logger.error(f"Stderr: {stderr.decode('utf-8')}")
            return False

        logger.info("DLProcessWrapper finished successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to run DLProcessWrapper: {e}")
        return False


def poll_for_status(file_path, timeout=600):
    """
    Poll Status.txt until SUCCESS/TENSORRT/End Learning is found, or timeout.
    Returns parsed status dict, {"status": "error", ...} on error, or None on timeout.
    """
    start_time = time.time()
    logger.info(f"Polling {file_path} for completion status...")

    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                if "SUCCESS" in content or "TENSORRT" in content or "End Learning" in content:
                    logger.info("Training process reported SUCCESS/TENSORRT/End Learning.")
                    return parse_status_log(file_path)
                content_upper = content.upper()
                if "ERROR" in content_upper or "EXCEPTION" in content_upper or "FAIL" in content_upper:
                    logger.error(f"Error detected in Status.txt: {content.strip()}")
                    return {"status": "error", "message": content.strip()}
            except Exception:
                pass
        time.sleep(2)

    logger.warning(f"Timeout waiting for training completion in {file_path}")
    return None


def poll_for_file(file_path, timeout=300):
    """Poll until file exists and is non-empty. Returns True if found, False on timeout."""
    start_time = time.time()
    logger.info(f"Polling for file: {file_path}")
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            try:
                if os.path.getsize(file_path) > 0:
                    logger.info(f"File found and ready: {file_path}")
                    return True
            except Exception:
                pass
        time.sleep(2)
    logger.warning(f"Timeout waiting for file: {file_path}")
    return False


# ----------------------------------------------------------------------
# Results parsing (classification)
# ----------------------------------------------------------------------
def calculate_missoverkill(file_path, ok_classes_norm):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    miss_count, total_defects, overkill_count, total_ok = 0, 0, 0, 0
    matrix_start_idx = -1
    for i, line in enumerate(lines):
        if "Confusion Matrix :" in line:
            matrix_start_idx = i
            break

    if matrix_start_idx != -1 and matrix_start_idx + 1 < len(lines):
        header_line = lines[matrix_start_idx + 1]
        headers     = [h.strip() for h in header_line.split() if h.strip()]
        header_map  = {h.lower(): idx for idx, h in enumerate(headers)}

        ok_col_indices  = [idx for h, idx in header_map.items() if h in ok_classes_norm]
        unknown_col_idx = header_map.get("unknown", -1)

        current_idx = matrix_start_idx + 2
        while current_idx < len(lines):
            line = lines[current_idx].strip()
            if not line or "," in line:
                break
            parts = [p.strip() for p in line.split() if p.strip()]
            if not parts:
                current_idx += 1
                continue
            row_label = parts[0]
            try:
                counts = [int(x) for x in parts[1:]]
            except ValueError:
                logger.debug(f"Skipping line due to non-integer counts: {line}")
                current_idx += 1
                continue

            if row_label.lower() == "unknown":
                current_idx += 1
                continue

            row_total         = sum(counts)
            unknown_pred_count = counts[unknown_col_idx] if unknown_col_idx != -1 and unknown_col_idx < len(counts) else 0
            valid_row_total   = row_total - unknown_pred_count
            pred_ok_total     = sum(counts[idx] for idx in ok_col_indices if idx < len(counts))
            is_ok_row         = row_label.lower() in ok_classes_norm

            if is_ok_row:
                overkill_count += valid_row_total - pred_ok_total
                total_ok       += valid_row_total
            else:
                miss_count    += pred_ok_total
                total_defects += valid_row_total

            current_idx += 1

    miss_rate    = (miss_count    / total_defects * 100.0) if total_defects > 0 else 0.0
    overkill_rate = (overkill_count / total_ok     * 100.0) if total_ok      > 0 else 0.0
    return miss_rate, overkill_rate, lines


def parse_csv_like(lines, miss_rate, overkill_rate, results):
    import math

    def safe_float(v):
        try:
            f = float(v)
            return 0.0 if math.isnan(f) or math.isinf(f) else f
        except Exception:
            return 0.0

    total_metrics = {}
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            try:
                row_label = parts[0].strip()
                if row_label.lower() == "unknown":
                    continue
                metrics = {
                    "total":      int(parts[1]),
                    "correct":    int(parts[2]),
                    "incorrect":  int(parts[3]),
                    "accuracy":   safe_float(parts[4]),
                    "error_rate": safe_float(parts[5]),
                }
                if row_label.lower() == "sum":
                    total_metrics = metrics
                    total_metrics["miss_rate"]    = miss_rate
                    total_metrics["overkill_rate"] = overkill_rate
                else:
                    results[row_label] = metrics
            except ValueError:
                continue
    return results, total_metrics


def parse_confusion_matrix_file(file_path, ok_classes=None):
    """
    Parse ConfMatrixTest.txt or ConfMatrixEval.txt.
    Returns {"class_metrics": ..., "total_metrics": ...} or None on failure.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Confusion matrix file not found: {file_path}")
        return None
    if ok_classes is None:
        ok_classes = ["OK"]
    ok_classes_norm = [c.lower() for c in ok_classes]
    results = {}
    try:
        miss_rate, overkill_rate, lines = calculate_missoverkill(file_path, ok_classes_norm)
        results, total_metrics = parse_csv_like(lines, miss_rate, overkill_rate, results)
    except Exception as e:
        logger.error(f"Error parsing confusion matrix file: {e}")
        return None
    return {"class_metrics": results, "total_metrics": total_metrics}


def parse_iou_matrix_file(file_path):
    """
    Parse IouMatrixEvaluation.txt or IouMatrixTest.txt.

    Expected format (two lines):
        MIoU, Precision, Recall, F1Score
        0.986915, 0.991919, 0.994914, 0.993404

    Returns a dict with those four keys, or None on failure.
    """
    if not os.path.exists(file_path):
        logger.warning(f"IoU matrix file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) < 2:
            logger.warning(f"IoU matrix file has fewer than 2 lines: {file_path}")
            return None
        headers = [h.strip() for h in lines[0].split(',')]
        values  = [v.strip() for v in lines[1].split(',')]
        if len(headers) != len(values):
            logger.warning(f"IoU matrix header/value count mismatch in {file_path}")
            return None
        result = {}
        for h, v in zip(headers, values):
            try:
                result[h] = float(v)
            except ValueError:
                result[h] = v
        return result
    except Exception as e:
        logger.error(f"Error parsing IoU matrix file {file_path}: {e}")
        return None


def generate_miss_overkill_csv(decision_list_path, output_csv_path, class_map, ok_classes_norm, output_csv_path_all=None):
    if not os.path.exists(decision_list_path):
        logger.warning(f"Decision list file not found: {decision_list_path}")
        return False

    records     = []
    records_all = []
    try:
        with open(decision_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                doc = {}
                for part in parts:
                    if '|' in part:
                        k, v = part.split('|', 1)
                        doc[k.strip()] = v.strip()

                if "Image Name" not in doc or "Actual Label" not in doc or "Predicted Label" not in doc:
                    continue

                img_path       = doc["Image Name"]
                actual_lbl_idx = doc["Actual Label"]
                pred_lbl_idx   = doc["Predicted Label"]
                prob_score     = doc.get("Probability Score", "")

                actual_class = class_map.get(str(actual_lbl_idx), f"Unknown_{actual_lbl_idx}")
                pred_class   = class_map.get(str(pred_lbl_idx),   f"Unknown_{pred_lbl_idx}")

                if actual_class.lower() == "unknown" or pred_class.lower() == "unknown":
                    continue

                is_actual_ok = actual_class.lower() in ok_classes_norm
                is_pred_ok   = pred_class.lower()   in ok_classes_norm
                
                if actual_lbl_idx == pred_lbl_idx:
                    prediction_type = "Hit"
                elif is_actual_ok and not is_pred_ok:
                    prediction_type = "Overkill"
                elif not is_actual_ok and is_pred_ok:
                    prediction_type = "Miss"
                else:
                    prediction_type = "Incorrect_NG"

                row = {
                    "File_Path":         img_path,
                    "Prediction Type":   prediction_type,
                    "Original":          actual_class,
                    "Predicted":         pred_class,
                    "Probability Score": prob_score,
                }
                records_all.append(row)
                if prediction_type in ("Miss", "Overkill"):
                    records.append(row)

        fieldnames = ["File_Path", "Prediction Type", "Original", "Predicted", "Probability Score"]
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow(row)

        if records:
            logger.info(f"Generated Miss/Overkill CSV with {len(records)} records at {output_csv_path}")
        else:
            logger.info(f"No misses or overkills found. Empty CSV created at {output_csv_path}")

        if output_csv_path_all:
            with open(output_csv_path_all, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in records_all:
                    writer.writerow(row)
            logger.info(f"Generated full Miss/Overkill CSV with {len(records_all)} records at {output_csv_path_all}")

        return True

    except Exception as e:
        logger.error(f"Error generating miss/overkill CSV from {decision_list_path}: {e}")
        return False


def _parse_seg_decision_line(line: str) -> Dict:
    """
    Parse one line of a segmentation decision list.

    Test format (with mask and IoU fields):
        Image Name|D:\\path\\OK\\img.bmp,Mask Name|mask.png, Predicted Label|1,
        Width|48, Height|108, Min Probability Score|0.5088, Max Probability Score|1.0000,
        Length|107, Area|4930, X|189, Y|19, IoU|0.989371, Precision|0.992230,
        Recall|0.997097, F1Score|0.994657

    Evaluation format (no detection / below threshold):
        Image Name|D:\\path\\OK\\img.bmp, Predicted Label|-2, Max Probability Score|0.4672
    """
    doc = {}

    # --- Image Name: anchor on Mask Name| if present, else on Predicted Label| ---
    m = re.search(r'Image Name\|(.+?),\s*Mask Name\|', line)
    if m:
        doc["Image Name"] = m.group(1).strip()
    else:
        m = re.search(r'Image Name\|(.+?),\s*Predicted Label\|', line)
        if m:
            doc["Image Name"] = m.group(1).strip()

    # --- Mask Name (value ends at next `, Word|`) ---
    m = re.search(r'Mask Name\|([^,]+?)(?=,\s*\w)', line)
    if m:
        doc["Mask Name"] = m.group(1).strip()

    # --- Numeric fields (handle negative values such as Predicted Label|-2) ---
    for key in (
        "Predicted Label", "Width", "Height", "Length", "Area", "X", "Y",
        "IoU", "Precision", "Recall", "F1Score",
        "Min Probability Score", "Max Probability Score",
    ):
        m = re.search(rf'{re.escape(key)}\|(-?[\d.]+)', line)
        if m:
            doc[key] = m.group(1).strip()

    return doc


def generate_seg_iou_csv(decision_list_path, output_csv_path):
    """
    Parse a segmentation decision list and write a per-image IoU CSV.
    All detected regions are recorded with their IoU metrics.
    """
    if not os.path.exists(decision_list_path):
        logger.warning(f"Seg decision list not found: {decision_list_path}")
        return False

    fieldnames = [
        "File_Path", "Mask_Name", "Predicted_Label",
        "IoU", "Precision", "Recall", "F1Score",
        "Min_Probability_Score", "Max_Probability_Score",
        "Width", "Height", "Area",
    ]
    records = []

    try:
        with open(decision_list_path, 'r', encoding='utf-8') as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue

                doc = _parse_seg_decision_line(raw)
                img_path = doc.get("Image Name", "")
                if not img_path:
                    continue

                records.append({
                    "File_Path":             img_path,
                    "Mask_Name":             doc.get("Mask Name", ""),
                    "Predicted_Label":       doc.get("Predicted Label", ""),
                    "IoU":                   doc.get("IoU", ""),
                    "Precision":             doc.get("Precision", ""),
                    "Recall":                doc.get("Recall", ""),
                    "F1Score":               doc.get("F1Score", ""),
                    "Min_Probability_Score": doc.get("Min Probability Score", ""),
                    "Max_Probability_Score": doc.get("Max Probability Score", ""),
                    "Width":                 doc.get("Width", ""),
                    "Height":                doc.get("Height", ""),
                    "Area":                  doc.get("Area", ""),
                })

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow(row)

        logger.info(f"Seg IoU CSV: {len(records)} records -> {output_csv_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating seg IoU CSV from {decision_list_path}: {e}")
        return False


# ----------------------------------------------------------------------
# Main service functions
# ----------------------------------------------------------------------
def reset_jsons(custom_params=None, storage_root=None, status1=None):
    """Reset Training/Evaluation/Testing JSONs from base templates when custom params are given."""
    if not (custom_params and len(custom_params) > 0) or not storage_root:
        return
    model_name = get_latest_model_name(storage_root)
    paths_dict = get_model_paths(storage_root, model_name)
    for tpl_name, target_key in [
        ("Training.json",   "training_json"),
        ("Evaluation.json", "evaluation_json"),
        ("Testing.json",    "testing_json"),
    ]:
        target_path   = paths_dict[target_key]
        template_path = os.path.join(storage_root, tpl_name)
        if not os.path.exists(template_path):
            template_path = os.path.join(storage_root, tpl_name.lower())
        if os.path.exists(template_path):
            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy(template_path, target_path)
            except Exception as e:
                process_logger.warning(f"NODE:{status1}, Failed to reset {target_path}: {e}")


def _get_ok_classes_from_json(training_json: str) -> List[str]:
    """Extract OK class names from Training.json classlst/classBin."""
    ok_classes = ["OK"]
    try:
        with open(training_json, 'r',encoding="utf-8") as f:
            t_data = json.load(f)
        class_lst = t_data.get("classlst", [])
        class_bin = t_data.get("classBin", [])
        if class_lst and class_bin and len(class_lst) == len(class_bin):
            found_ok = [
                c.get("ClassName") for i, c in enumerate(class_lst)
                if i < len(class_bin) and str(class_bin[i].get("classBinName", "")).lower() != "ng"
                and c.get("ClassName")
            ]
            if found_ok:
                ok_classes = found_ok
            else:
                ok_classes = class_lst[0]['ClassName']
        
    except Exception as e:
        logger.warning(f"Failed to parse OK classes from {training_json}: {e}")
        process_logger.error("NODE:88, Failed to parse OK classes")
    return ok_classes


def _build_class_map_from_json(training_json: str) -> Dict[str, str]:
    class_map = {}
    try:
        with open(training_json, 'r',encoding="utf-8") as f:
            t_data = json.load(f)
        for i, c in enumerate(t_data.get("classlst", [])):
            class_map[str(i)] = c.get("ClassName", f"Class_{i}")
    except Exception:
        pass
    return class_map


def run_automated_training(
    custom_params=None,
    dataset_csv_path=None,
    storage_root=None,
    label: Optional[str] = None,
    status1=12,
    training_type: str = "classification",
):
    """
    Automated train → evaluate → test pipeline using DLProcessWrapper.

    Args:
        training_type: "classification" (default) or "segmentation"
    """
    process_logger.info(f"NODE:{status1}, Starting Automated {training_type.title()} Training via Frontier")
    modes = DL_MODES[training_type]

    # --- Reset JSONs if custom params provided ---
    reset_jsons(custom_params=custom_params, storage_root=storage_root, status1=status1)

    # --- Step 0: Scan dataset and update JSON configs ---
    # if not custom_dataset_path or not storage_root:
    #     return {"status": "error", "message": "custom_dataset_path and storage_root are required."}

        
    has_test_data = True
    if training_type == "segmentation":
        seg_scan = csv_scan_dataset_and_update_configs_seg(
            dataset_csv_path=dataset_csv_path,
            storage_root=storage_root,
            hyperparams=custom_params or {},
        )
        if not seg_scan["success"]:
            return {"status": "error", "message": "Failed to scan segmentation dataset and update configurations."}
        has_test_data = seg_scan["has_test"]
    elif training_type == "objectdetection":
        objdet_scan = csv_scan_dataset_and_update_configs_objdet(
            dataset_csv_path=dataset_csv_path,
            storage_root=storage_root,
            hyperparams=custom_params or {},
        )
        if not objdet_scan["success"]:
            return {"status": "error", "message": "Failed to scan object detection dataset and update configurations."}
        has_test_data = objdet_scan["has_test"]
    else:
        # if not scan_dataset_and_update_configs(
        #     custom_dataset_path=custom_dataset_path,
        #     storage_root=storage_root,
        # ):
        #     return {"status": "error", "message": "Failed to scan dataset and update configurations."}
        if not csv_scan_dataset_and_update_configs( dataset_csv_path= dataset_csv_path,storage_root=storage_root, label=label):
            return {"status": "error", "message": "Failed to scan dataset and update configurations."}

    model_name   = get_latest_model_name(storage_root)
    paths_dict   = get_model_paths(storage_root, model_name)
    training_json = paths_dict["training_json"]
    eval_json     = paths_dict["evaluation_json"]
    status_file   = paths_dict["status_file"]
    train_dir_json = paths_dict["train_dir"]

    # Clean up stale Status.txt
    try:
        if os.path.exists(status_file):
            with open(status_file, "w") as f:
                pass
            # os.remove(status_file)
    except Exception as e:
        process_logger.warning(f"NODE:{status1}, Failed to clean up stale Status.txt: {e}")

    # --- Hyperparameter resolution ---
    default_epochs = 100
    full_epochs    = (custom_params or {}).get("epochs", default_epochs)
    if label != "OPTUNA":
        process_logger.info(f"NODE:{status1}, Using provided hyperparameters: {custom_params}")
        best_params = custom_params or {}
        if "epochs" not in best_params:
            best_params["epochs"] = full_epochs
    else:
        process_logger.info(f"NODE:{status1}, No hyperparameters provided. Starting Optuna tuning...")
        try:
            def objective(trial):
                seg_batch_lst = [8, 16, 32]
                cls_batch_lst = [8, 16, 32, 64]
                trial_lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                trial_batch_size = trial.suggest_categorical("batch_size", cls_batch_lst) if training_type.lower() == 'classification' else trial.suggest_categorical("batch_size", seg_batch_lst)
                trial_params     = {"lr": trial_lr, "batch_size": trial_batch_size,
                                    "epochs": (custom_params or {}).get("trial_epoch", 10)}
                process_logger.info(f"NODE:{status1}, Optuna Trial {trial.number}: {trial_params}")

                if not generate_config_json(training_json, mode="Train", params=trial_params, training_type=training_type):
                    raise optuna.exceptions.TrialPruned("Failed to generate training configuration.")

                if os.path.exists(status_file):
                    try: os.remove(status_file)
                    except Exception: pass

                for stop_fname in ("StopbufferTrain.txt", "StopbufferTest.txt"):
                    try:
                        with open(os.path.join(train_dir_json, stop_fname), 'w') as fp:
                            fp.write("1")
                    except Exception as e:
                        process_logger.warning(f"NODE:{status1}, Failed to create {stop_fname}: {e}")

                process_logger.info(f"NODE:{status1}, Launching Training for Trial {trial.number}...")
                if not run_dl_process_wrapper(training_json, "trial", modes["Train"]):
                    raise optuna.exceptions.TrialPruned("DLProcessWrapper execution failed.")

                status = poll_for_status(status_file, timeout=60000)
                if status and isinstance(status, dict) and status.get("status") == "error":
                    raise optuna.exceptions.TrialPruned(f"Training failed: {status.get('message')}")
                if not status:
                    status = parse_status_log(status_file)
                if status and "val_loss" in status:
                    return status["val_loss"]
                if status and "best_loss" in status:
                    return status["best_loss"]
                raise optuna.exceptions.TrialPruned("Could not read loss from Status.txt")

            optuna.logging.set_verbosity(optuna.logging.INFO)
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=(custom_params or {}).get("n_trials", 6))
            best_params = study.best_params
            best_params["epochs"] = full_epochs
            process_logger.info(f"NODE:{status1}, Optuna completed. Best params: {best_params}")
        except ImportError:
            process_logger.warning(f"NODE:{status1}, Optuna not installed. Using default parameters.")
            best_params = {"epochs": full_epochs}
        except Exception as e:
            process_logger.warning(f"NODE:{status1}, Optuna error: {e}. Falling back to defaults.")
            best_params = {"epochs": full_epochs}

    # Apply best params to all JSON configs
    process_logger.info(f"NODE:{status1}, Generating final configs with params: {best_params}")
    for target_key in ("training_json", "evaluation_json", "testing_json"):
        target_path = paths_dict.get(target_key)
        if target_path and os.path.exists(target_path):
            if not generate_config_json(target_path, mode="Train", params=best_params,
                                        training_type=training_type):
                process_logger.warning(f"NODE:{status1}, Failed to apply params to {target_path}")

    # --- 3. Run final Training ---
    ok_classes = _get_ok_classes_from_json(training_json) if training_type == "classification" else ["OK"]

    if os.path.exists(status_file):
        try: os.remove(status_file)
        except Exception: pass

    for stop_fname in ("StopbufferTrain.txt", "StopbufferTest.txt"):
        try:
            with open(os.path.join(train_dir_json, stop_fname), 'w') as fp:
                fp.write("1")
        except Exception as e:
            process_logger.warning(f"NODE:{status1}, Failed to create {stop_fname}: {e}")

    if not run_dl_process_wrapper(training_json, status1, modes["Train"]):
        return {"status": "error", "message": "Training process failed execution"}

    # --- 4. Poll for training completion ---
    last_status = poll_for_status(status_file, timeout=60000)
    if last_status and isinstance(last_status, dict) and last_status.get("status") == "error":
        return last_status
    if not last_status:
        process_logger.warning(f"NODE:{status1}, Training completion marker not found. Proceeding with caution.")
        last_status = parse_status_log(status_file)

    # --- 5. Run Evaluation ---
    eval_conf_matrix_path = os.path.join(train_dir_json, "ConfMatrixEvaluation.txt")
    eval_iou_matrix_path  = os.path.join(train_dir_json, "IouMatrixEvaluation.txt")
    # For segmentation the primary completion signal is the IoU matrix
    eval_poll_path = eval_iou_matrix_path if training_type == "segmentation" else eval_conf_matrix_path
    stop_buffer_path = os.path.join(train_dir_json, "StopbufferTest.txt")

    for stale in (eval_conf_matrix_path, eval_iou_matrix_path):
        if os.path.exists(stale):
            try: os.remove(stale)
            except Exception: pass

    try:
        with open(stop_buffer_path, 'w') as fp:
            fp.write("1")
    except Exception as e:
        process_logger.warning(f"NODE:{status1}, Failed to create StopbufferTest.txt: {e}")

    if not run_dl_process_wrapper(eval_json, status1, modes["Evaluate"]):
        # return {"status": "error", "message": "Evaluation process failed execution",
        #         "last_training_status": last_status}
        return {"status": "error", "message": "Error in run_dl_process_wrapper function.",
                "last_training_status": last_status}

    if not poll_for_file(eval_poll_path, timeout=600):
        process_logger.warning(f"NODE:{status1}, Timeout waiting for Evaluation results.")

    # --- 6. Parse Eval results ---
    eval_results    = None
    class_map       = {}
    ok_classes_norm = [c.lower() for c in ok_classes]

    if training_type == "classification":
        eval_results = parse_confusion_matrix_file(eval_conf_matrix_path, ok_classes=ok_classes)
        class_map    = _build_class_map_from_json(training_json)
        eval_decision_list = os.path.join(train_dir_json, "Evaluation_decision_list.txt")
        eval_csv_path      = os.path.join(train_dir_json, "Evaluation_Miss_Overkill.csv")
        eval_csv_path_all  = os.path.join(train_dir_json, "Evaluation_Miss_Overkill_All.csv")
        if os.path.exists(eval_decision_list):
            generate_miss_overkill_csv(eval_decision_list, eval_csv_path, class_map, ok_classes_norm, eval_csv_path_all)
    elif training_type == "segmentation":
        eval_results = {
            "iou_metrics":  parse_iou_matrix_file(eval_iou_matrix_path),
            "conf_metrics": parse_confusion_matrix_file(eval_conf_matrix_path) if os.path.exists(eval_conf_matrix_path) else None,
        }
        eval_decision_list = os.path.join(train_dir_json, "Evaluation_decision_list.txt")
        eval_csv_path      = os.path.join(train_dir_json, "Evaluation_IoU.csv")
        if os.path.exists(eval_decision_list):
            generate_seg_iou_csv(eval_decision_list, eval_csv_path)

    # --- 7. Run Testing ---
    test_results = None
    if training_type == "segmentation" and not has_test_data:
        process_logger.info(f"NODE:{status1}, No test data found; skipping testing step.")
    else:
        test_dir_path       = paths_dict["test_dir"]
        real_test_json_path = paths_dict["testing_json"]
        os.makedirs(test_dir_path, exist_ok=True)

        test_conf_matrix_path = os.path.join(test_dir_path, "ConfMatrixTest.txt")
        test_iou_matrix_path  = os.path.join(test_dir_path, "IouMatrixTest.txt")
        test_poll_path   = test_iou_matrix_path if training_type == "segmentation" else test_conf_matrix_path
        stop_buffer_test = os.path.join(test_dir_path, "StopbufferTest.txt")

        for stale in (test_conf_matrix_path, test_iou_matrix_path):
            if os.path.exists(stale):
                try: os.remove(stale)
                except Exception: pass

        try:
            with open(stop_buffer_test, 'w') as fp:
                fp.write("1")
        except Exception as e:
            process_logger.warning(f"NODE:{status1}, Failed to create StopbufferTest.txt in Test dir: {e}")

        if not run_dl_process_wrapper(real_test_json_path, status1, modes["Test"]):
            return {"status": "error", "message": "Testing process failed execution",
                    "last_training_status": last_status, "eval_results": eval_results}

        if not poll_for_file(test_poll_path, timeout=600):
            process_logger.warning(f"NODE:{status1}, Timeout waiting for Testing results.")

        # --- 8. Parse Test results ---
        if training_type == "classification":
            test_results = parse_confusion_matrix_file(test_conf_matrix_path, ok_classes=ok_classes)
            test_decision_list = os.path.join(test_dir_path, "test_decision_list.txt")
            test_csv_path      = os.path.join(test_dir_path, "Test_Miss_Overkill.csv")
            test_csv_path_all  = os.path.join(test_dir_path, "Test_Miss_Overkill_All.csv")
            if os.path.exists(test_decision_list):
                generate_miss_overkill_csv(test_decision_list, test_csv_path, class_map, ok_classes_norm, test_csv_path_all)
        elif training_type == "segmentation":
            test_results = {
                "iou_metrics":  parse_iou_matrix_file(test_iou_matrix_path),
                "conf_metrics": parse_confusion_matrix_file(test_conf_matrix_path) if os.path.exists(test_conf_matrix_path) else None,
            }
            test_decision_list = os.path.join(test_dir_path, "test_decision_list.txt")
            test_csv_path      = os.path.join(test_dir_path, "Test_IoU.csv")
            if os.path.exists(test_decision_list):
                generate_seg_iou_csv(test_decision_list, test_csv_path)

    return {
        "status":               "success",
        "last_training_status": last_status,
        "eval_results":         eval_results,
        "test_results":         test_results,
        "model_name":           model_name,
        "train_dir":            train_dir_json,
        "best_params":          best_params,
    }

def run_export_model(storage_root: str, training_type: str = "classification", status=14, label=None) -> bool:
    """Export the trained model using DLProcessWrapper (Export / segExport mode)."""
    process_logger.info(f"NODE:{status}, Starting model export ({training_type})")
    # export_json_path = os.path.join(storage_root,"Expo")
    model_name = get_latest_model_name(storage_root)
    paths_dict = get_model_paths(storage_root, model_name)
    training_json  = paths_dict["training_json"]
    export_json_path = paths_dict["export_json"]


    template_rel = json_data['paths'][training_type.lower()]['export_json']
    template_path = os.path.join(os.getcwd(), template_rel)
    if not os.path.exists(template_path):
        logger.error(f"Export_TF.json template not found at: {template_path}")
        return False

    export_data = load_json(template_path)
    if export_data is None:
        return False

    t_data = load_json(training_json)
    if t_data and "Model" in t_data:
        m = t_data["Model"]
        export_data["iHeight"]    = m.get("iRectHeight", m.get("iHeight", export_data.get("iHeight", 144)))
        export_data["iWidth"]     = m.get("iRectWidth",  m.get("iWidth",    export_data.get("iWidth", 256)))
        export_data["iChannels"]  = m.get("iChannels",                      export_data.get("iChannels", 3))
        export_data["SolutionDir"] = m.get("SolutionDir",                   export_data.get("SolutionDir", ""))
        if m.get("strROIType"):
            export_data["strROIType"] = m["strROIType"]

    template_model_name = export_data.get("name", model_name)
    template_save_path = export_data.get("ExportSavePath", "")
    base_path = template_save_path.rsplit(template_model_name, 1)[0]
    export_save_path = os.path.join(storage_root, f"Exported_model_{label}" if isinstance(label, str) and label else "Exported_model")
    export_data["name"] = model_name
    export_data["ExportSavePath"] = export_save_path

    os.makedirs(export_save_path, exist_ok=True)

    if not dump_json(export_data, export_json_path):
        return False

    export_mode = "segExport" if training_type == "segmentation" else "Export"
    if not run_dl_process_wrapper(export_json_path, status, export_mode):
        process_logger.error("NODE:88, Model export failed")
        return False
    if training_type != "segmentation":
        shutil.copy(os.path.join(storage_root,"Imported_model\Class_Info.txt"), export_save_path)
    source = os.path.join(storage_root,f"Test\{model_name}")
   
    process_logger.info(f"NODE:{status}, Model exported to {export_save_path}")
    return export_save_path

def export_misclassified(txt_file, csv_file, class_map, csv_file_all=None, ok_classes_norm=None):
    rows     = []
    rows_all = []
    with open(txt_file) as f:
        for line in f:
            actual    = line.split("Actual Label|")[1].split(",")[0].strip()
            pred      = line.split("Predicted Label|")[1].split(",")[0].strip()
            file_path = line.split("Image Name|")[1].split("Actual Label|")[0].strip().rstrip(',')
            prob      = line.split("Probability Score|")[1].split(",")[0].strip()
            actual_class = class_map[actual]
            pred_class   = class_map[pred]

            if csv_file_all is not None:
                if actual == pred:
                    prediction_type = "Hit"
                elif ok_classes_norm is not None:
                    is_actual_ok = actual_class.lower() in ok_classes_norm
                    is_pred_ok   = pred_class.lower()   in ok_classes_norm
                    if is_actual_ok and not is_pred_ok:
                        prediction_type = "Overkill"
                    elif not is_actual_ok and is_pred_ok:
                        prediction_type = "Miss"
                    else:
                        prediction_type = "Incorrect_NG"
                else:
                    prediction_type = "Mismatch"
                rows_all.append([file_path, prediction_type, actual_class, pred_class, prob])

            if actual != pred:
                rows.append([file_path, actual_class, pred_class, prob])

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File_Path", "Original", "Predicted", "Probability Score"])
        writer.writerows(rows)

    if csv_file_all is not None:
        with open(csv_file_all, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["File_Path", "Prediction Type", "Original", "Predicted", "Probability Score"])
            writer.writerows(rows_all)

def run_inference(
    dataset_csv_path=None,
    storage_root=None,
    status: Optional[str] = None,
    label=None,
    training_type: str = "classification",
):
    """
    Run Evaluation + Testing (inference) on an existing trained model.

    Args:
        training_type: "classification" (default) or "segmentation"
    """
    logger.info(f"Starting {training_type.title()} Inference process...")
    modes = DL_MODES[training_type]

    if not storage_root:
        return {"status": "error", "message": "storage_root is required."}

    has_test_data = True
    if training_type == "segmentation":
        seg_scan = inf_csv_scan_dataset_and_update_configs_seg(
            dataset_csv_path=dataset_csv_path,
            storage_root=storage_root,
            label=label,
        )
        if not seg_scan["success"]:
            return {"status": "error", "message": "Failed to scan segmentation dataset and update configurations."}
        has_test_data = seg_scan["has_test"]
    elif training_type == "objectdetection":
        objdet_scan = inf_csv_scan_dataset_and_update_configs_objdet(
            dataset_csv_path=dataset_csv_path,
            storage_root=storage_root,
            label=label,
        )
        if not objdet_scan["success"]:
            return {"status": "error", "message": "Failed to scan object detection dataset and update configurations."}
        has_test_data = objdet_scan["has_test"]
    else:
        # if not scan_dataset_and_update_configs(
        #     custom_dataset_path=file_path,
        #     storage_root=storage_root,
        # ):
        #     return {"status": "error", "message": "Failed to scan dataset and update configurations."}
        if not inf_csv_scan_dataset_and_update_configs(dataset_csv_path=dataset_csv_path ,storage_root=storage_root, label=label):
            return {"status": "error", "message": "Failed to scan dataset and update configurations."}  
    model_name    = get_latest_model_name(storage_root)
    paths_dict    = get_model_paths(storage_root, model_name)
    training_json = paths_dict["training_json"]
    eval_json     = paths_dict["evaluation_json"]
    test_json     = paths_dict["testing_json"]
    train_dir_json = paths_dict["train_dir"]
    test_dir_path  = paths_dict["test_dir"]

    # Variables used in the classification branches below; initialized here so they are always bound.
    classes:   List[str]  = []
    test_imgs: List[Dict] = []

    # --- 1. Run Evaluation ---
    eval_conf_matrix_path = os.path.join(train_dir_json, "ConfMatrixEvaluation.txt")
    eval_iou_matrix_path  = os.path.join(train_dir_json, "IouMatrixEvaluation.txt")
    eval_poll_path   = eval_iou_matrix_path if training_type == "segmentation" else eval_conf_matrix_path
    stop_buffer_path = os.path.join(train_dir_json, "StopbufferTest.txt")

    for stale in (eval_conf_matrix_path, eval_iou_matrix_path):
        if os.path.exists(stale):
            try: os.remove(stale)
            except Exception: pass

    try:
        with open(stop_buffer_path, 'w') as fp:
            fp.write("1")
    except Exception as e:
        process_logger.warning(f"NODE:{status}, Failed to create StopbufferTest.txt: {e}")

    if not run_dl_process_wrapper(eval_json, status, modes["Evaluate"]):
        # return {"status": "error", "message": "Evaluation process failed execution"}
        return {"status": "error", "message": "Error in run_dl_process_wrapper function"}

    if not poll_for_file(eval_poll_path, timeout=600):
        process_logger.warning(f"NODE:{status}, Timeout waiting for Evaluation results.")

    # Parse eval results
    eval_results    = None
    class_map       = {}
    ok_classes      = ["OK"]
    ok_classes_norm = ["ok"]

    if training_type == "classification":
        ok_classes      = _get_ok_classes_from_json(training_json)
        ok_classes_norm = [c.lower() for c in ok_classes]
        class_map       = _build_class_map_from_json(training_json)
        eval_results    = parse_confusion_matrix_file(eval_conf_matrix_path, ok_classes=ok_classes)

        eval_decision_list = os.path.join(train_dir_json, "Evaluation_decision_list.txt")
        eval_csv_path      = os.path.join(train_dir_json, "Evaluation_Miss_Overkill.csv")
        eval_csv_path_all  = os.path.join(train_dir_json, "Evaluation_Miss_Overkill_All.csv")
        if os.path.exists(eval_decision_list):
            generate_miss_overkill_csv(eval_decision_list, eval_csv_path, class_map, ok_classes_norm, eval_csv_path_all)
    elif training_type == "segmentation":
        eval_results = {
            "iou_metrics":  parse_iou_matrix_file(eval_iou_matrix_path),
            "conf_metrics": parse_confusion_matrix_file(eval_conf_matrix_path) if os.path.exists(eval_conf_matrix_path) else None,
        }
        eval_decision_list = os.path.join(train_dir_json, "Evaluation_decision_list.txt")
        eval_csv_path      = os.path.join(train_dir_json, "Evaluation_IoU.csv")
        if os.path.exists(eval_decision_list):
            generate_seg_iou_csv(eval_decision_list, eval_csv_path)

    # --- 2. Run Testing ---
    test_results = None
    if training_type == "segmentation" and not has_test_data:
        process_logger.info(f"NODE:{status}, No test data found; skipping testing step.")
    else:
        os.makedirs(test_dir_path, exist_ok=True)

        if training_type == "classification":
            """if not os.path.exists(test_json):
                return {"status": "error", "message": "Testing configuration missing."}
            try:
                with open(test_json, 'r') as f:
                    test_data = json.load(f)
                if "Model" in test_data:
                    test_data["Model"]["iTestImgCount"]       = len(test_imgs)
                    test_data["Model"]["iTotalClasses"]       = len(classes)
                    test_data["Model"]["iTrainImgCount"]      = 0
                    test_data["Model"]["iValidationImgCount"] = 0
                test_data["testImglst"]  = test_imgs
                test_data["trainImglst"] = []
                test_data["ValImgList"]  = []
                with open(test_json, 'w') as f:
                    json.dump(test_data, f, indent=2)
            except Exception as e:
                return {"status": "error", "message": "Failed to update Testing configuration."}"""

        test_conf_matrix_path = os.path.join(test_dir_path, "ConfMatrixTest.txt")
        test_iou_matrix_path  = os.path.join(test_dir_path, "IouMatrixTest.txt")
        test_poll_path   = test_iou_matrix_path if training_type == "segmentation" else test_conf_matrix_path
        stop_buffer_test = os.path.join(test_dir_path, "StopbufferTest.txt")

        for stale in (test_conf_matrix_path, test_iou_matrix_path):
            if os.path.exists(stale):
                try: os.remove(stale)
                except Exception: pass

        try:
            with open(stop_buffer_test, 'w') as fp:
                fp.write("1")
        except Exception as e:
            process_logger.warning(f"NODE:{status}, Failed to create StopbufferTest.txt in Test dir: {e}")

        if not run_dl_process_wrapper(test_json, status, modes["Test"]):
            return {"status": "error", "message": "Testing process failed execution",
                    "eval_results": eval_results}

        if not poll_for_file(test_poll_path, timeout=600):
            process_logger.warning(f"NODE:{status}, Timeout waiting for Testing results.")

        if training_type == "classification":
            test_results = parse_confusion_matrix_file(test_conf_matrix_path, ok_classes=ok_classes)
            label_item = label.split(" ")[0]
            workflow_tag = label.replace(" ","_")
            test_decision_list = os.path.join(test_dir_path, "test_decision_list.txt")
            # test_csv_path      = os.path.join(test_dir_path, f"{workflow_tag}.csv")
            test_csv_path      = os.path.join(test_dir_path, f"Test_Miss_Overkill_{label_item}.csv")
            test_csv_path_all  = os.path.join(test_dir_path, f"Test_Miss_Overkill_{label_item}_All.csv")
            if os.path.exists(test_decision_list):
                # generate_miss_overkill_csv(test_decision_list, test_csv_path, class_map, ok_classes_norm)
                export_misclassified(test_decision_list, test_csv_path, class_map, test_csv_path_all, ok_classes_norm)
        elif training_type == "segmentation":
            test_results = {
                "iou_metrics":  parse_iou_matrix_file(test_iou_matrix_path),
                "conf_metrics": parse_confusion_matrix_file(test_conf_matrix_path) if os.path.exists(test_conf_matrix_path) else None,
            }
            test_decision_list = os.path.join(test_dir_path, "test_decision_list.txt")
            test_csv_path      = os.path.join(test_dir_path, "Test_IoU.csv")
            if os.path.exists(test_decision_list):
                generate_seg_iou_csv(test_decision_list, test_csv_path)

    return {
        "status":       "success",
        "eval_results": eval_results,
        "test_results": test_results,
        "model_name":   model_name,
        "csv_path":     os.path.join(test_dir_path,f"Test_Miss_Overkill_{label_item}.csv") if training_type == "classification" else None,
    }