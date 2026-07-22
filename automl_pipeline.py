import logging
import json
import os
import uuid
import shutil
import random
import zipfile
import csv
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel
import sys
sys.path.append(os.getcwd())
from backend.plato_main import PlatoService
from backend.all_consts import IMAGE_EXTENSIONS, IMAGE_EXTENSIONS_Dot
from backend.config import MODELS_DIR, get_latest_model_name
# from backend.services.training_service import run_automated_training, run_inference, run_export_model
from backend.services.training_service import run_automated_training, run_inference, run_export_model
from backend.services.data_service import apply_fix
from backend.api_processes import execute_filter_data
from backend.services.agent_service import analyze_situation_and_decide
from backend.fid_calculator import calculateFID, remove_path, process_one_subfolder
from backend.auto_cluster import create_clusters
from backend.utils import concatenate_csv_files, ceil_or_highest, is_r2_better_than_r1
from backend.process_logger import process_logger
from backend.torch_models.train import Train
from datetime import datetime
from collections import Counter
from pathlib import Path
import shutil

random.seed(999)

ANALYSIS_CODES = [36, 46, 56]
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(_CONFIG_PATH, "r") as _f:
    _APP_CONFIG = json.load(_f)

_PATHS = _APP_CONFIG["paths"]

def _task_paths(task_type: str) -> dict:
    return _PATHS[task_type.lower()]

NON_NG_CLASS_NAME = _APP_CONFIG['non_ng_class']
OPTUNA_PARAMS = _APP_CONFIG['Optuna_parameters']
DEFAULT_HYPERPARAMS: Dict[str, Any] = _APP_CONFIG["default_hyperparameters"]
CLASSES = _APP_CONFIG["classes"]
NON_NG_CLASS_NAME = _APP_CONFIG['non_ng_class']
ACCURACY_GAN_THRESHOLD = 0.9
MAX_ANALYSE_CYCLES = 4
FID_COEFFICIENT= _APP_CONFIG['FID_coeff']
CLEANLAB_CONF = _APP_CONFIG['cleanlab_conf']
DEBUG_MODE = _APP_CONFIG['DEBUG_MODE']
DEBUG_EPOCHS = _APP_CONFIG['DEBUG_EPOCHS']
DEBUG_IMPROVED = True
if DEBUG_MODE:
    DEBUG_IMPROVED = False

def _json_path(job: dict) -> Path:
    return (Path(job['config']["storage_path"]) / job['job'] / job['config']["triggerId"] / "Doc" / "data.json")


def store_json(job: dict, key: str, value) -> None:
    p = _json_path(job)
    data = json.loads(p.read_text()) if p.exists() else {}
    data.setdefault(job['config']['triggerId'], {})[key] = value
    p.write_text(json.dumps(data, indent=4, default=str))


def load_json(job: dict) -> dict:
    return json.loads(_json_path(job).read_text()).get(job['config']['triggerId'], {})


def count_image_types(folder_path):
    path = Path(folder_path)
    ext_list = [
        f.suffix.lower() for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS_Dot
    ]
    return Counter(ext_list)


def get_fid_calculation(cluster_folder_path, gan_input_path, gan_output_path, storage_folder=None):
    print(f"Number of classes: {len(os.listdir(cluster_folder_path))}")
    print(f"gan_output_path: {gan_output_path}")
    # gan_output_path = find_longest_number_folder(os.path.join(gan_output_path, "Image/result"))
    current_path = os.path.dirname(os.getcwd())
    base_fid = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if storage_folder is None:
        dataset_folder = os.path.join(current_path, 'storage', 'cluster_results', timestamp)
    else:
        dataset_folder = os.path.join(storage_folder, 'cluster_results', timestamp)

    print("NODE:27, Calculating Base FID Score....")
    for folder in os.listdir(cluster_folder_path):
        first_50 = os.path.join(dataset_folder, f"{folder}_first_50")
        second_50 = os.path.join(dataset_folder, f"{folder}_second_50")
        remove_path(first_50)
        remove_path(second_50)
        for sub in os.listdir(os.path.join(cluster_folder_path, folder)):
            sub_folder_path = Path(os.path.join(cluster_folder_path, folder, sub))
            if sub_folder_path.is_dir():
                process_one_subfolder(sub_folder_path, first_50, second_50, rng=42)
        result = calculateFID(ground_path=first_50, predicted_path=second_50)
        base_fid += result

    output_fid = 0.0
    fid_len = 0.0
    print("NODE:27, Calculating Gan Input & Output FID Score....")
    for folder in os.listdir(gan_input_path):
        folder_input_path = os.path.join(gan_input_path, folder)
        counts = count_image_types(folder_input_path)
        count = sum(counts.values())
        if count <= 1:
            continue
        
        folder_output_path = os.path.join(gan_output_path, 'train/images', folder, "channel1")
        result = calculateFID(ground_path=folder_input_path, predicted_path=folder_output_path)
        output_fid += result
        fid_len += 1

    gan_fid_value = output_fid / fid_len
    base_fid_value = base_fid / fid_len
    return base_fid_value, gan_fid_value

def api_export_model(storage_root=None, training_type="classification", status=14, label=None) -> None:
    process_logger.info(f"NODE:{status}, Exporting improved model")
    export_save_path = run_export_model(storage_root=storage_root, training_type=training_type, status=status, label=label)
    if not export_save_path:
        process_logger.error("NODE:88, Model export failed")
        raise HTTPException(500, detail="Model export failed")
    process_logger.info(f"NODE:{status}, Model export completed")
    return export_save_path

def extract_accuracy(result: Dict[str, Any]) -> Optional[float]:
    # Segmentation: use MIoU as the accuracy metric
    try:
        val = result["eval_results"]["iou_metrics"]["MIoU"]
        return float(val)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        val = result["test_results"]["iou_metrics"]["MIoU"]
        return float(val)
    except (KeyError, TypeError, ValueError):
        pass
    # Classification
    try:
        val = result["test_results"]["total_metrics"]["accuracy"]
        return float(val) / 100.0 if float(val) > 1.0 else float(val)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        val = result["eval_results"]["total_metrics"]["accuracy"]
        return float(val) / 100.0 if float(val) > 1.0 else float(val)
    except (KeyError, TypeError, ValueError):
        pass
    for key in ("accuracy", "val_accuracy", "f1", "metric", "score", "top1"):
        if key in result:
            try:
                return float(result[key])
            except (TypeError, ValueError):
                pass
    for wrapper in ("result", "metrics", "report"):
        if wrapper in result and isinstance(result[wrapper], dict):
            for key in ("accuracy", "val_accuracy", "f1", "metric", "score", "top1"):
                if key in result[wrapper]:
                    try:
                        return float(result[wrapper][key])
                    except (TypeError, ValueError):
                        pass
    return None


def extract_miss_overkill(result):
    try:
        total_metrics = result['test_results']['total_metrics']
        return total_metrics['miss_rate'], total_metrics['overkill_rate']
    except:
        return None, None


def has_improved(new_acc, best_acc,
                 new_miss_rate=None, new_overkill_rate=None,
                 new_test_miss_rate=None, new_test_overkill_rate=None,
                 best_miss_rate=None, best_overkill_rate=None,
                 best_new_test_miss_rate=None, best_new_test_overkill_rate=None):
    all_rates = [new_miss_rate, new_overkill_rate, best_miss_rate, best_overkill_rate,
                 new_test_miss_rate, new_test_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate]
    # Segmentation: no miss/overkill rates — fall back to accuracy-only comparison
    if all(v is None for v in all_rates):
        if new_acc is not None and best_acc is not None:
            improved = new_acc > best_acc
            return improved, None, None, None, None
        return False, None, None, None, None

    if new_miss_rate is None or new_overkill_rate is None or best_miss_rate is None or best_overkill_rate is None or \
            new_test_miss_rate is None or new_test_overkill_rate is None or best_new_test_miss_rate is None or best_new_test_overkill_rate is None:
        return False, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate

    if (new_miss_rate <= best_miss_rate and new_overkill_rate <= best_overkill_rate) and \
            (new_test_miss_rate <= best_new_test_miss_rate and new_test_overkill_rate <= best_new_test_overkill_rate):
        return True, new_miss_rate, new_overkill_rate, new_test_miss_rate, new_test_overkill_rate
    else:
        return False, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate


def check_performance_gain(inference_result: Dict[str, Any],
                            new_test_inference_result: Dict[str, Any],
                            best_acc: Optional[float],
                            label: str=None,
                            status: int=None,
                            best_miss_rate=None,
                            best_overkill_rate=None,
                            best_new_test_miss_rate=None,
                            best_new_test_overkill_rate=None,
                            task_type: str="classification"):
    new_acc = extract_accuracy(inference_result)
    new_miss_rate, new_overkill_rate = extract_miss_overkill(inference_result)
    new_test_miss_rate, new_test_overkill_rate = extract_miss_overkill(new_test_inference_result)

    improved, updated_miss, updated_overkill, updated_new_test_miss_rate, updated_new_test_overkill_rate = has_improved(
        new_acc, best_acc,
        new_miss_rate, new_overkill_rate,
        new_test_miss_rate, new_test_overkill_rate,
        best_miss_rate, best_overkill_rate,
        best_new_test_miss_rate, best_new_test_overkill_rate)

    is_seg = task_type == "segmentation"
    acc_label = "MIoU" if is_seg else "Accuracy"

    prev_res = f"{best_acc:.4f}" if best_acc is not None else "N/A"
    process_logger.info(f"NODE:{status}, Baseline {acc_label}: {prev_res}")
    if not is_seg:
        prev_miss_res    = f"{best_miss_rate:.4f}" if best_miss_rate is not None else "N/A"
        prev_ok_res      = f"{best_overkill_rate:.4f}" if best_overkill_rate is not None else "N/A"
        prev_nt_miss_res = f"{best_new_test_miss_rate:.4f}" if best_new_test_miss_rate is not None else "N/A"
        prev_nt_ok_res   = f"{best_new_test_overkill_rate:.4f}" if best_new_test_overkill_rate is not None else "N/A"
        process_logger.info(f"NODE:{status}, Baseline Miss rate: {prev_miss_res}, Overkill rate: {prev_ok_res}")
        process_logger.info(f"NODE:{status}, Baseline New Test Miss rate: {prev_nt_miss_res}, Overkill rate: {prev_nt_ok_res}")

    cur_res = f"{new_acc:.4f}" if new_acc is not None else "N/A"
    process_logger.info(f"NODE:{status}, Current {acc_label}: {cur_res}")
    if not is_seg:
        new_miss_res    = f"{new_miss_rate:.4f}" if new_miss_rate is not None else "N/A"
        new_ok_res      = f"{new_overkill_rate:.4f}" if new_overkill_rate is not None else "N/A"
        nt_miss_res     = f"{new_test_miss_rate:.4f}" if new_test_miss_rate is not None else "N/A"
        nt_ok_res       = f"{new_test_overkill_rate:.4f}" if new_test_overkill_rate is not None else "N/A"
        process_logger.info(f"NODE:{status}, Current Miss rate: {new_miss_res}, Overkill rate: {new_ok_res}")
        process_logger.info(f"NODE:{status}, Current New Test Miss rate: {nt_miss_res}, Overkill rate: {nt_ok_res}")

    process_logger.info(f"NODE:{status}, Improved: {improved}")
    updated = max(filter(None, [new_acc, best_acc])) if any([new_acc, best_acc]) else None
    return improved, updated, updated_miss, updated_overkill, updated_new_test_miss_rate, updated_new_test_overkill_rate




def api_train(dataset_csv_path=None, hyperparameters=None, storage_root=None, label="", status=12, training_type="classification") -> Dict[str, Any]:
    res = run_automated_training(custom_params=hyperparameters, dataset_csv_path=dataset_csv_path, storage_root=storage_root, label=label, status1=status, training_type=training_type)
    result = {"status": "completed", "result": res}
    if result["status"] == "error":
        process_logger.error(f"NODE:88, {result['message']}")
        raise HTTPException(404, detail=result["message"])
    return result


def api_inference(dataset_csv_path=None, storage_root=None, label="", status=13, training_type="classification") -> Dict[str, Any]:
    kwargs = {"dataset_csv_path": dataset_csv_path, "storage_root": storage_root, "status": status, "label": label, "training_type": training_type}
    process_logger.info(f"NODE:{status}, Model inference started on {label} test set")
    result = run_inference(**kwargs)
    if result["status"] == "error":
        process_logger.error(f"NODE:88, {result['message']}")
        raise HTTPException(404, detail=result["message"])
    process_logger.info(f"NODE:{status}, Inference Completed")
    return result


def api_similar_data(train_dir=None, csv_path=None, output_folder=None, train_csv=None):
    svc = PlatoService()
    dummy_payload = {
        "payload": {
            "result": {
                "train_dir": train_dir,
                "csv_path": csv_path,
                "train_csv": train_csv
            }
        }
    }
    out = svc.get_similar_data(dummy_payload, output_folder=output_folder)
    return out['result']


def _create_csv_gan(payload: dict, trigger_path: str) -> None:
    src = payload["config"]["source"]
    image_pth, channel, cls_name, img_name = [], [], [], []
    channels = payload['config']["pathSetting"]["noOfChannel"]
    for j in os.listdir(src):
        imgs = os.listdir(os.path.join(src, j))
        ch_check = len(imgs) % channels
        imgs = imgs[:-ch_check] if ch_check else imgs
        for idx, i in enumerate(imgs, start=1):
            if i.split(".")[-1] in IMAGE_EXTENSIONS:
                img_name.append(i)
                image_pth.append(os.path.join(src, j, i))
                channel.append(f"channel{idx % channels or channels}")
                cls_name.append(j)
    csv_out = os.path.join(trigger_path, "dataset.csv")
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "channel_name", "class_name", "image_name"])
        w.writerows(zip(image_pth, channel, cls_name, img_name))


def api_GAN(gan_config_path: str, gan_source: str, v_path: str, status=26) -> str:
    print(gan_config_path)
    with open(gan_config_path,encoding="utf-8") as f:
        payload = json.load(f)

    payload["config"].update({
        'source': gan_source,
        'datasetId': 1,
        'pathSetting': {**payload["config"]['pathSetting'],
                        'versionPath': v_path, 'datasetPath': gan_source},
    })
    payload["config"]['dataSettings']['height'] = payload["config"]['dataSettings']['width']
    payload["config"]['dataSettings']['classes'] = os.listdir(gan_source)
    payload["progress"] = 0.0
    payload["status"] = status

    from PIL import Image
    class_list = os.listdir(gan_source)
    img_list = os.listdir(os.path.join(gan_source, class_list[0]))
    for i in img_list:
        if i.split(".")[-1].lower() in IMAGE_EXTENSIONS:
            img = Image.open(os.path.join(gan_source, class_list[0], i))
            break
    width, height = img.size
    payload['config']['postProcess']['generateHeight'], payload['config']['postProcess']['generateWidth'] = height, width
    result_size = ceil_or_highest(width)
   
    payload["config"]['dataSettings']['height'] = result_size
    payload["config"]['dataSettings']['width'] = result_size
    
    _create_csv_gan(payload, trigger_path=v_path)
    best_result_path = None
    try:
        best_result_path = Train(payload).process()
    except Exception as err:
        process_logger.error(f"NODE:88, Error in api_GAN function: {err}")
        raise
    return v_path, best_result_path


def api_validator(input_folder=None, output_folder=None, storage_folder=None):
    result = execute_filter_data(
        {'input_folder': input_folder, 'gaudi_output_folder': output_folder,
         'job_id': uuid.uuid4(), 'storage_folder': storage_folder})
    return result

def api_analyse(file_path: str, csv_out: str) -> Dict[str, Any]:
    try:
        result = analyze_situation_and_decide(file_path, csv_out, CLEANLAB_CONF)

    except Exception as e:
        process_logger.error(f"NODE:88, data analysis error: {e}")
        raise
    return result


def find_longest_number_folder(folder_path) -> Path:
    folder_path = Path(folder_path)
    numeric_folders = []
    for item in folder_path.iterdir():
        if item.is_dir():
            try:
                numeric_folders.append((int(item.name), item))
            except ValueError:
                continue
    if not numeric_folders:
        raise FileNotFoundError(f"No numeric sub-folders found in {folder_path}")
    numeric_folders.sort(key=lambda x: x[0], reverse=True)
    return numeric_folders[0][1]


class job_skeleton(BaseModel):
    server_ip: Union[str, None] = None
    server_port: Union[int, None] = None
    backend_ip: Union[str, None] = None
    backend_port: Union[int, None] = None
    smart_id: Union[str, None] = None
    config: dict
    gpu_id: Union[int, None] = None
    gpu_no: Union[int, None] = None
    job: Union[str, None] = None
    job_type: Union[str, None] = None
    logs: str = "Assigning Job to Server"
    progress: float = 0.0
    error: Union[str, None] = None
    allowNextJob: bool
    system_id: Union[int, None] = None


def write_json(json_path, frontier_dir, task_type,label: Optional[str] = None, class_list=None, class_bin=None, classPrbwt=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # if class_list is not None:
    #     data['classlst'] = class_list
    #     data['Model']['iTotalClasses'] = len(class_list)
    # if class_bin is not None:
    #     data['classBin'] = class_bin
    # if classPrbwt is not None:
    #     data['classPrbwt'] = classPrbwt
    data['Model']['SolutionDir'] = frontier_dir + os.sep
    data['Model']['tfrec_lmdb_path'] = frontier_dir + os.sep
    if label:
        sol_dir = os.path.dirname(os.path.dirname(frontier_dir))
        data['Model']['SolutionDir'] = sol_dir + os.sep
        data['Model']['tfrec_lmdb_path'] = frontier_dir + os.sep
        if os.path.basename(json_path) in ["Testing.json", "Evaluation.json"]:
            with open(os.path.join(sol_dir, "Imported_model\Training.json"),encoding='utf-8') as f:
                source_data= json.load(f)
            data['classlst'] = source_data['classlst']
            data['classBin']=source_data['classBin']

            data['classPrbwt'] = [{"ClassName": item['ClassName'],
                                   "ClassProb": 0.0} for item in source_data['classlst']]
        # else:
        #     if task_type=="classification":
        #         with open(os.path.join(sol_dir, "Imported_model\Training.json")) as f:
        #             source_data= json.load(f)
        #         ok_class = next(cls['ClassName'] for cls, bin_info in zip(source_data['classlst'], source_data['classBin']) if bin_info['classBinName'] == "OK")
        #         json_path = os.path.join(os.getcwd(), 'backend/config.json')
        #         with open(json_path, 'r',encoding="utf-8") as json_file:
        #             json_data = json.load(json_file)
        #         json_data['non_ng_class']=ok_class
        #         with open(json_path, 'w') as f:
        #             json.dump(json_data, f, indent=4)
             
    output_path = os.path.join(frontier_dir, os.path.basename(json_path))
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def get_latest_model_name(storage_root):
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

def _move_loose_items_into_model_dir(container_dir, model_dir):
    """If a template ships files directly under Train/ or Test/ (no Model_N/
    subfolder — e.g. objectdetection's Test/), relocate them into .../Model_1/ so the
    layout matches classification/segmentation templates, which already nest everything
    under Model_1/. Must run before anything does shutil.copy2(..., model_dir): if
    model_dir doesn't exist yet as a directory, copy2 treats it as a destination
    *filename* instead and silently creates a file where a folder was expected."""
    if not os.path.exists(container_dir):
        return
    os.makedirs(model_dir, exist_ok=True)
    for item in os.listdir(container_dir):
        if item.startswith("Model_"):
            continue
        src_path = os.path.join(container_dir, item)
        dst_path = os.path.join(model_dir, item)
        if not os.path.exists(dst_path):
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
        if os.path.isfile(src_path):
            os.remove(src_path)
        elif os.path.isdir(src_path):
            shutil.rmtree(src_path)


def setup_model_directories(storage_root, import_files, task_type="classification"):
    try:
        if import_files and os.path.isfile(import_files):
            import_files = os.path.dirname(import_files)
        model_root = os.path.join(storage_root, "model")
        import_model_dir = os.path.join(model_root, "Imported_model")
        train_model_dir = os.path.join(model_root, "Train", "Model_1")
        test_model_dir = os.path.join(model_root, "Test", "Model_1")
        req_model_dir = os.path.join("backend", "req_files", task_type.lower(), "model")
        shutil.copytree(req_model_dir, model_root, dirs_exist_ok=True)
        # Move any files/dirs that landed at Train/ or Test/ level into Model_1/ where
        # they belong (no-op for classification/segmentation, whose templates already
        # nest everything under Model_1/).
        _move_loose_items_into_model_dir(os.path.join(model_root, "Train"), train_model_dir)
        _move_loose_items_into_model_dir(os.path.join(model_root, "Test"), test_model_dir)
        if not import_files or not os.path.exists(import_files):
            process_logger.error("NODE:88, Please import a valid frontier model...")
            return 
        shutil.copytree(import_files, import_model_dir, dirs_exist_ok=True)
        shutil.copytree(import_model_dir, train_model_dir, dirs_exist_ok=True)
        # The net config lives in Train/Model_N/ModelConfig/ of the source model.
        # When import_files is Exported_model/, go up to the parent model dir to find it.
        parent_model_dir = os.path.dirname(import_files)
        imported_model_name = get_latest_model_name(parent_model_dir)
        # Object detection models are PyTorch: best_weights.pth + net.yaml.
        # Classification/segmentation models are Keras: model_best_epoch.h5 + net.json.
        is_objdet = task_type.lower() == "objectdetection"
        weights_name = "best_weights.pth" if is_objdet else "model_best_epoch.h5"
        net_name = "net.yaml" if is_objdet else "net.json"
        # Prefer the original trained weights over the exported copy so baseline
        # inference uses the same weights as base training inference did.
        src_weights = os.path.join(parent_model_dir, "Train", imported_model_name, weights_name)
        if not os.path.exists(src_weights):
            src_weights = os.path.join(import_model_dir, weights_name)
        shutil.copy2(src_weights, test_model_dir)
        shutil.copy2(src_weights, train_model_dir)
        net_file = os.path.join(import_files, net_name)
        if not os.path.exists(net_file):
            net_file = os.path.join(parent_model_dir, "Train", imported_model_name, "ModelConfig", net_name)
        if os.path.exists(net_file):
            for mc_dst in [os.path.join(train_model_dir, "ModelConfig"),
                           os.path.join(test_model_dir, "ModelConfig")]:
                os.makedirs(mc_dst, exist_ok=True)
                shutil.copy2(net_file, mc_dst)
        else:
            process_logger.error(f"NODE:88, {net_name} not found at {net_file} - ModelConfig will be incomplete")
        return
    except Exception as e:
        process_logger.error(f"NODE:88, Model setup failed: {e}")
        raise e
    

def _resolve_task_template(json_path):
    """These config.json-listed templates are static (opened by exact filename, no
    Model_N versioning) and normally sit flat — but tolerate a Model_1/ subfolder too,
    in case someone applies the runtime model scaffold's Model_N/ nesting convention
    here by mistake. Falls back to the original flat path if neither exists, so a
    genuinely missing template still errors with the expected canonical path."""
    if os.path.exists(json_path):
        return json_path
    nested = os.path.join(os.path.dirname(json_path), "Model_1", os.path.basename(json_path))
    return nested if os.path.exists(nested) else json_path


def frontier_changes_(new_classes, storage_root, label: Optional[str] = None, task_type: str = "classification", analysis=False, gan=False):
    task_cfg = _task_paths(task_type)
    train_json = _resolve_task_template(task_cfg["train_json"])
    test_json = _resolve_task_template(task_cfg["test_json"])
    eval_json = _resolve_task_template(task_cfg["eval_json"])
    model_config_src = task_cfg["model_config_src"]

    # frontier_dir = os.path.join(storage_root, "modelConfig", "model")
    frontier_dir = os.path.join(storage_root, "model")
    os.makedirs(frontier_dir, exist_ok=True)

    global CLASSES
    if new_classes is not None:
        CLASSES = new_classes

    global NON_NG_CLASS_NAME
    if new_classes is not None and NON_NG_CLASS_NAME not in new_classes:
        NON_NG_CLASS_NAME = new_classes[0]

    class_list = []
    class_bin = []
    classPrbwt = []
    if new_classes is not None and task_type != "segmentation":
        for new_class in new_classes:
            class_list.append({"ClassName": new_class, "iClassWeights": 1})
            classPrbwt.append({"ClassName": new_class, "ClassProb": 0.0})
            if NON_NG_CLASS_NAME == new_class:
                class_bin.append({"classBinName": NON_NG_CLASS_NAME})
            else:
                class_bin.append({"classBinName": "NG"})

    is_seg = task_type == "segmentation"
    write_json(train_json, frontier_dir,task_type,
               class_list=None if is_seg else class_list,
               class_bin=None if is_seg else class_bin,
               classPrbwt=None if is_seg else classPrbwt)
    write_json(test_json, frontier_dir,task_type,
               class_list=None if is_seg else class_list,
               class_bin=None if is_seg else class_bin,
               classPrbwt=None if is_seg else classPrbwt)
    write_json(eval_json, frontier_dir,task_type,
               class_list=None if is_seg else class_list,
               class_bin=None if is_seg else class_bin,
               classPrbwt=None if is_seg else classPrbwt)

    dst = os.path.join(frontier_dir, "ModelConfig")
    shutil.copytree(model_config_src, dst, dirs_exist_ok=True)

    if label:
        front_train = os.path.join(frontier_dir, "Train")
        front_test = os.path.join(frontier_dir, "Test")

        if os.path.exists(front_train):
            model_name = get_latest_model_name(front_train)
            front_train = os.path.join(front_train, model_name)
            os.makedirs(front_train, exist_ok=True)
            for json_name in ("Training.json", "Evaluation.json"):
                target = os.path.join(front_train, json_name)
                if not os.path.exists(target):
                    src = os.path.join(frontier_dir, json_name)
                    if os.path.exists(src):
                        shutil.copy(src, target)
            write_json(os.path.join(front_train, "Training.json"), front_train, task_type,label, classPrbwt=None if is_seg else classPrbwt)
            write_json(os.path.join(front_train, "Evaluation.json"), front_train, task_type,label, classPrbwt=None if is_seg else classPrbwt)
        else:
            process_logger.error("NODE:88, No trained model found in frontier Train dir")
            raise

        if os.path.exists(front_test):
            model_name = get_latest_model_name(front_test)
            front_test = os.path.join(front_test, model_name)
            os.makedirs(front_test, exist_ok=True)
            target = os.path.join(front_test, "Testing.json")
            if not os.path.exists(target):
                src = os.path.join(frontier_dir, "Testing.json")
                if os.path.exists(src):
                    shutil.copy(src, target)
            write_json(os.path.join(front_test, "Testing.json"), front_test, task_type,label, classPrbwt=None if is_seg else classPrbwt)

    return frontier_dir


def remove_empty_subfolders(parent_folder):
    for root, dirs, files in os.walk(parent_folder, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)


def _analyse_and_pause_(job=None, 
                        doc_dir=None,
                        dataset_csv_path:str=None, 
                        cycle_num: int=None, 
                        status: int=None, 
                        new_test=None,
                        ) -> None:
    process_logger.info(f"NODE:{status}, Analyse cycle {cycle_num}/{MAX_ANALYSE_CYCLES} running")
    csv_out = os.path.join(doc_dir, f"analyse_cycle_{cycle_num}.csv")
    analyse_result = api_analyse(dataset_csv_path, csv_out)
    if len(analyse_result['new_issues'])==0:
        process_logger.info("NODE:83, Stopped as no cleanlab suggestions")
        process_logger.progress(100.0)
        return 
    saved_csv = analyse_result.get("issues_csv_path")
    if new_test is None:
        store_json(job, "analyse_cycle_count", cycle_num)
        store_json(job, f"last_analyse_csv-NODE:{status}", saved_csv)
    process_logger.info(f"NODE:{status}, STATUS:WAITING, Cycle-{cycle_num}/{MAX_ANALYSE_CYCLES}, User-review CSV-{Path(saved_csv).as_posix()}")
    process_logger.info("NODE:82, Stopped after analyse cycle")
    process_logger.progress(100.0)

    return saved_csv

def initialize(job=None, hyperparameters=None):
    
    cfg = job['config']
    storage_root = os.path.join(cfg["storage_path"], job['job'], cfg["triggerId"])
    doc_dir = os.path.join(storage_root, "Doc")
    os.makedirs(doc_dir, exist_ok=True) 
    
    try:
        dataset_csv_path = cfg['dataset_csv_path']
    except:
        dataset_csv_path = os.path.join(storage_root, "train_dataset.csv")


    process_logger.http_setup(ip=job['server_ip'], port=job['server_port'])
    process_logger.configure_logger(path=doc_dir, trigger_id=cfg['triggerId'],
                                    filename=f"{cfg['triggerId']}.log")
    process_logger.progress(1.0)
    print(f"{'*'*10} Job starting with DEBUG MODE: {DEBUG_MODE} and DEBUG IMPROVED: {DEBUG_IMPROVED} {'*'*10}")
    process_logger.info(f"NODE:11, DEBUG MODE: {DEBUG_MODE} and DEBUG_IMPROVED: {DEBUG_IMPROVED}")

    task_type = job["job"].lower()
    hparams = hyperparameters or DEFAULT_HYPERPARAMS
    best_acc: Optional[float] = None
    active_hparams = hparams

    gan_config = _task_paths(task_type)["default_gan_config"]

    GAN_STATUS_CODE = 27

    return cfg, GAN_STATUS_CODE, dataset_csv_path, storage_root, doc_dir, task_type, gan_config, hparams, best_acc

def run_training_with_gan(
        job=None,
        dataset_csv_path=None,
        storage_root=None,
        task_type=None,
        doc_dir=None

):

    process_logger.info("NODE:31, GAN-validated images received (status[-1]==27)")

    data_json = job['config']["storage_path"] + "/" + job['job'] + '/' + job['config']["triggerId"] + "/Doc/data.json"
    with open(data_json, "r",encoding="utf-8") as f:
        data = json.load(f)

    tid = job['config']["triggerId"]
    base_accuracy_result = data[tid].get("base_accuracy")
    new_test_base_result = data[tid].get("New_Test_Base")
    best_acc = extract_accuracy(base_accuracy_result)
    best_miss_rate, best_overkill_rate = extract_miss_overkill(base_accuracy_result)
    best_new_test_miss_rate, best_new_test_overkill_rate = extract_miss_overkill(new_test_base_result)
    active_hparams = data[tid].get("active_hparams")

    df = pd.read_csv(dataset_csv_path)
    classes = df['class_label'].unique().tolist()
    frontier_dir = frontier_changes_(classes, storage_root, label="Improvement", task_type=task_type, gan=True)
    api_train(dataset_csv_path=dataset_csv_path, hyperparameters=active_hparams, storage_root=frontier_dir, label="GAN-validated Training", status=32, training_type=task_type)

    process_logger.progress(60.0)
    inf = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="Baseline GAN-validated Training", status=33, training_type=task_type)
    export_conf_files(storage_root=storage_root, label="GAN_trained", which_data="Baseline", task_type=task_type)
    new_test_inf = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="New_Test GAN-validated Training", status=33)
    export_conf_files(storage_root=storage_root, label="GAN_trained", which_data="New_Test", task_type=task_type)

    improved, best_acc, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate = check_performance_gain(
        inf, new_test_inf, best_acc, "After GAN", status=33,
        best_miss_rate=best_miss_rate, best_overkill_rate=best_overkill_rate,
        best_new_test_miss_rate=best_new_test_miss_rate, best_new_test_overkill_rate=best_new_test_overkill_rate,
        task_type=task_type)

    store_json(job, "Best_acc", inf)
    store_json(job, "Best_acc-NODE:34", inf)
    store_json(job, "New_Test-NODE:34", new_test_inf)
    store_json(job, "New_Test", new_test_inf)
    process_logger.progress(80.0)
    
    export_save_path = api_export_model(storage_root=frontier_dir, training_type=task_type, status=34, label="GAN_trained")

    if improved and DEBUG_IMPROVED:
        res         = f"{best_acc:.4f}" if best_acc is not None else "N/A"
        miss_res    = f"{best_miss_rate:.4f}" if best_miss_rate is not None else "N/A"
        ok_res      = f"{best_overkill_rate:.4f}" if best_overkill_rate is not None else "N/A"
        nt_miss_res = f"{best_new_test_miss_rate:.4f}" if best_new_test_miss_rate is not None else "N/A"
        nt_ok_res   = f"{best_new_test_overkill_rate:.4f}" if best_new_test_overkill_rate is not None else "N/A"
        process_logger.info(f"NODE:14, Best Accuracy- {res},  Miss rate- {miss_res}, Overkill rate- {ok_res}")
        process_logger.info(f"NODE:14,  New Test Miss rate- {nt_miss_res}, Overkill rate- {nt_ok_res}")
        improved_path = os.path.join(os.path.dirname(export_save_path), "improved")
        shutil.copytree(export_save_path,improved_path, dirs_exist_ok=True)
        process_logger.info(f"NODE:80, Improved model saved at - {improved_path}")
        # _save_improved_zip(frontier_dir,storage_root,
        #     label="base_training", best_accuracy=best_acc, miss_rate=best_miss_rate, overkill_rate=best_overkill_rate)
        
        # process_logger.info("NODE:34, Improved with GAN generated dataset  — STOPPING")
        # api_export_model(storage_root=frontier_dir, training_type=task_type, status=34)
        process_logger.progress(100.0)
        return
    else:
        process_logger.info("NODE:34,  No improvement with GAN generated dataset")
        process_logger.info(f"NODE:35, Accuracy < Target - Analysis workflow, cycle 1/{MAX_ANALYSE_CYCLES}")
        _analyse_and_pause_(job, doc_dir, dataset_csv_path, cycle_num=1, status=35)
        return

def run_training_with_analysis(job=None,
                               dataset_csv_path=None,
                               storage_root=None,
                               task_type=None,
                               doc_dir=None):
    
    data_json = os.path.join(job['config']["storage_path"], job['job'], job['config']["triggerId"], "Doc/data.json")
    with open(data_json, "r",encoding="utf-8") as f:
        data = json.load(f)

    tid = job['config']["triggerId"]
    base_accuracy_result = data[tid].get("base_accuracy")
    new_test_base_result = data[tid].get("New_Test_Base")
    best_acc = extract_accuracy(base_accuracy_result)
    best_miss_rate, best_overkill_rate = extract_miss_overkill(base_accuracy_result)
    best_new_test_miss_rate, best_new_test_overkill_rate = extract_miss_overkill(new_test_base_result)
    cycle_num = data[tid].get("analyse_cycle_count")
    active_hparams = data[job['config']["triggerId"]].get("active_hparams")
    next_cycle = cycle_num + 1
    num_ = next_cycle * 10 + 20

    if next_cycle > MAX_ANALYSE_CYCLES:
        res = f"{best_acc:.4f}" if best_acc is not None else "N/A"
        process_logger.warning(f"NODE:{num_ + 1}, All {MAX_ANALYSE_CYCLES} analyse cycles exhausted. Best accuracy- {res}")
        process_logger.info(f"NODE:83, All {MAX_ANALYSE_CYCLES} analyse cycles exhausted — ending pipeline")
        process_logger.progress(100.0)
        return

    process_logger.info(f"NODE:{num_ + 1}, processing input data..")
    process_logger.info(f"NODE:{num_ + 2}, Training after fix (cycle %d)", cycle_num)
    df = pd.read_csv(dataset_csv_path)
    classes = df['class_label'].unique().tolist()
    frontier_dir = frontier_changes_(classes, storage_root, label="Improvement", task_type=task_type, analysis=True)
    api_train(dataset_csv_path=dataset_csv_path, hyperparameters=active_hparams, storage_root=frontier_dir, label=f"Post-fix cycle {cycle_num}", status=num_ + 2, training_type=task_type)
    process_logger.info(f"NODE:{num_ + 2}, Model Training Completed & model Saved")
    process_logger.progress(60.0)

    inf = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label=f"Baseline Post-fix cycle {cycle_num}", status=num_ + 3, training_type=task_type)
    export_conf_files(storage_root=storage_root, label=f"{cycle_num}_Cleanlab_trained", which_data="Baseline", task_type=task_type)
    new_test_inf = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label=f"New_Test Post-fix cycle {cycle_num}", status=num_ + 3, training_type=task_type)
    export_conf_files(storage_root=storage_root, label=f"{cycle_num}_Cleanlab_trained", which_data="New_Test", task_type=task_type)
    
    improved, best_acc, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate = check_performance_gain(
        inf, new_test_inf, best_acc, "After Analysis", status=num_ + 4,
        best_miss_rate=best_miss_rate, best_overkill_rate=best_overkill_rate,
        best_new_test_miss_rate=best_new_test_miss_rate, best_new_test_overkill_rate=best_new_test_overkill_rate,
        task_type=task_type)

    store_json(job, "Best_acc", inf)
    store_json(job, "New_Test", new_test_inf)
    store_json(job, f"Best_acc-NODE:{num_ + 4}", inf)
    store_json(job, f"New_Test-NODE:{num_ + 4}", new_test_inf)
    export_save_path =api_export_model(storage_root=frontier_dir, training_type=task_type, status=num_ + 4, label=f"{cycle_num}_Cleanlab_trained")

    if improved and DEBUG_IMPROVED:
        res         = f"{best_acc:.4f}" if best_acc is not None else "N/A"
        miss_res    = f"{best_miss_rate:.4f}" if best_miss_rate is not None else "N/A"
        ok_res      = f"{best_overkill_rate:.4f}" if best_overkill_rate is not None else "N/A"
        nt_miss_res = f"{best_new_test_miss_rate:.4f}" if best_new_test_miss_rate is not None else "N/A"
        nt_ok_res   = f"{best_new_test_overkill_rate:.4f}" if best_new_test_overkill_rate is not None else "N/A"
        process_logger.info(f"NODE:{num_ + 4}, Best Accuracy- {res},  Miss rate- {miss_res}, Overkill rate- {ok_res}")
        process_logger.info(f"NODE:{num_ + 4},  New Test Miss rate- {nt_miss_res}, Overkill rate- {nt_ok_res}")
        improved_path = os.path.join(os.path.dirname(export_save_path), "improved")
        shutil.copytree(export_save_path,improved_path, dirs_exist_ok=True)
        process_logger.info(f"NODE:80, Improved model saved at - {improved_path}")
        # _save_improved_zip(frontier_dir,storage_root,
        #     label="base_training", best_accuracy=best_acc, miss_rate=best_miss_rate, overkill_rate=best_overkill_rate)
        # process_logger.info(f"NODE:{num_ + 4}, Improved after analysis {next_cycle} — STOPPING")
        # api_export_model(storage_root=frontier_dir, training_type=task_type, status=num_ + 4)
        process_logger.progress(100.0)
        return
    if cycle_num == 3:
        store_json(job, "analyse_cycle_count", next_cycle)
        res = f"{best_acc:.4f}" if best_acc is not None else "N/A"
        process_logger.warning(f"NODE:{num_ + 1}, All 3 analyse cycles exhausted. Best accuracy- {res}")
        process_logger.info("NODE:83, All 3 analyse cycles exhausted — ending pipeline")
        process_logger.progress(100.0)
        return
    _analyse_and_pause_(job, doc_dir, dataset_csv_path, next_cycle, status=num_ + 5)
    return

def run_segmentation_retry(job=None,
                            dataset_csv_path=None,
                            storage_root=None,
                            task_type=None,
                            doc_dir=None):
    process_logger.info("NODE:42, Resuming with updated dataset (status[-1]==82)")

    data_json = os.path.join(doc_dir, "data.json")
    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tid = job['config']["triggerId"]

    if "Best_acc-NODE:44" in data[tid]:
        process_logger.info("NODE:83, Retry already used for this trigger - ending pipeline")
        process_logger.progress(100.0)
        return

    best_acc_result = data[tid].get("base_accuracy")
    best_acc = extract_accuracy(best_acc_result)
    active_hparams = data[tid].get("active_hparams")

    df = pd.read_csv(dataset_csv_path)
    classes = df['class_label'].unique().tolist()
    frontier_dir = frontier_changes_(classes, storage_root, label="Improvement", task_type=task_type, analysis=True)

    api_train(dataset_csv_path=dataset_csv_path, hyperparameters=active_hparams, storage_root=frontier_dir, label="Retry", status=42, training_type=task_type)
    process_logger.info("NODE:42, Model Training Completed & model Saved")
    process_logger.progress(60.0)

    inf = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="Baseline Retry", status=43, training_type=task_type)
    export_conf_files(storage_root=storage_root, label="manual_mask_correction", which_data="Baseline", task_type=task_type)
    new_test_inf = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="New_Test Retry", status=43, training_type=task_type)
    export_conf_files(storage_root=storage_root, label="manual_mask_correction", which_data="New_Test", task_type=task_type)

    improved, best_acc, _, _, _, _ = check_performance_gain(
        inf, new_test_inf, best_acc, "After Retry", status=44,
        task_type=task_type)

    store_json(job, "Best_acc", inf)
    store_json(job, "Best_acc-NODE:44", inf)
    store_json(job, "New_Test", new_test_inf)
    store_json(job, "New_Test-NODE:44", new_test_inf)

    export_save_path = api_export_model(storage_root=frontier_dir, training_type=task_type, status=44, label="manual_mask_correction")

    if improved and DEBUG_IMPROVED:
        res = f"{best_acc:.4f}" if best_acc is not None else "N/A"
        process_logger.info(f"NODE:44, Best MIoU: {res}")
        improved_path = os.path.join(os.path.dirname(export_save_path), "improved")
        shutil.copytree(export_save_path, improved_path, dirs_exist_ok=True)
        process_logger.info("NODE:44, Improved after retry - STOPPING")
        process_logger.info(f"NODE:80, Improved model saved at - {improved_path}")
    else:
        process_logger.info("NODE:83, Retry with updated dataset did not improve MIoU - ending pipeline")

    process_logger.progress(100.0)
    return

def run_base_training(job=None,
                        dataset_csv_path=None,
                        storage_root=None,
                        task_type=None,
                        best_acc=None,
                        hparams=None):
    process_logger.info("NODE:11, Dataset upload..")
    process_logger.info("NODE:12, Base model training started")
    df= pd.read_csv(dataset_csv_path)
    classes = df['class_label'].unique().tolist()
    frontier_dir = frontier_changes_(classes, storage_root=storage_root, task_type=task_type)

    # api_train(dataset_path=dataset_path,
    #         hyperparameters=hparams,storage_root=frontier_dir,label="BASE", status = 12, training_type=task_type)
    api_train(dataset_csv_path=dataset_csv_path, hyperparameters=hparams, storage_root=frontier_dir,label="BASE", status=12, training_type=task_type)
    process_logger.info("NODE:12, Model Training Completed & model Saved")
    process_logger.progress(60.0)
    # inf = api_inference(file_path=dataset_path, storage_root=frontier_dir, label="Base Param Training", status = 13, training_type=task_type)
    inf = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="Baseline Base Training", status=13, training_type=task_type)
    process_logger.progress(80.0)
    acc = extract_accuracy(inf)
    miss_rate, overkill_rate = extract_miss_overkill(result=inf)
    
    store_json(job, "Best_acc", inf)
    store_json(job, "Best_acc-NODE:13", inf)
    store_json(job, "frontier_dir", frontier_dir)
    if best_acc is not None:
        res = f"{best_acc:.4f}"
    else:
        res = "N/A"
    if miss_rate is not None:
        miss_res = f"{miss_rate:.4f}"
    else:
        miss_res = "N/A"
    if overkill_rate is not None:
        overkill_res = f"{overkill_rate:.4f}"
    else:
        overkill_res = "N/A"
    process_logger.info(f"NODE:13, Best Accuracy- {res}, Miss rate- {miss_res}, Overkill rate- {overkill_res}")
    # _save_improved_zip(frontier_dir,storage_root,
    #                     label="base_training", best_accuracy=acc, miss_rate=miss_rate, overkill_rate=overkill_rate)
    # process_logger.info("NODE:80, Base training — STOPPING")
    export_save_path = api_export_model(storage_root=frontier_dir, training_type=task_type)
    
    process_logger.progress(100.0)
    return

def validate_dataset(file_path, data_type):
    """
    Validates CSV schema and file paths based on the dataset type.
    
    Args:
        file_path (str): Path to the CSV file.
        data_type (str): Either 'miss' or 'train'.
    """
    
    # Define schemas
    # For 'miss', all columns are mandatory
    MISS_SCHEMA = ['File_Path', 'Original', 'Predicted', 'Probability Score']
    
    # For 'train', all columns except 's.no.' are mandatory
    TRAIN_MANDATORY = ['image_id', 'image_url', 'class_type', 'class_label', 
                       'Data_type', 'set', 'model_mask_url']

    try:
        df = pd.read_csv(file_path)
        actual_columns = set(df.columns)
        print(f"\n--- Validating [{data_type.upper()}] Dataset: {file_path} ---")

        # 1. Schema Validation Logic
        if data_type == 'miss':
            mandatory_set = set(MISS_SCHEMA)
            path_column = 'File_Path'
        elif data_type == 'train':
            mandatory_set = set(TRAIN_MANDATORY)
            path_column = 'image_url'
        else:
            print(f"[ERROR] Invalid data_type provided: '{data_type}'. Use 'miss' or 'train'.")
            return False

        # Check for missing mandatory columns
        missing_cols = mandatory_set - actual_columns
        if missing_cols:
            print(f"[ERROR] Missing mandatory columns for {data_type}: {list(missing_cols)}")
            return False
        else:
            print(f"[SUCCESS] Schema validation passed for {data_type}.")

        # 2. Path/URL Existence Validation
        print(f"--- Verifying existence of files in '{path_column}' ---")
        
        # Check if the column exists before attempting path check
        if path_column in df.columns:
            # Check existence (works for local paths)
            path_exists_series = df[path_column].apply(lambda x: os.path.exists(str(x)))
            
            missing_files = df[~path_exists_series][path_column].tolist()
            total_rows = len(df)
            missing_count = len(missing_files)

            if missing_count == 0:
                print(f"[SUCCESS] All {total_rows} files/paths are valid.")
            else:
                print(f"[WARNING] Path validation failed!")
                print(f"Found: {total_rows - missing_count}/{total_rows} files.")
                print(f"Missing: {missing_count} files.")
                print("Sample of missing paths:")
                for path in missing_files[:5]:
                    print(f" - {path}")
                return False
        else:
            print(f"[ERROR] Reference column '{path_column}' not found for path validation.")
            return False

        print(f"[RESULT] All checks passed for {data_type} dataset.")
        return True

    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return False

def run_gan_worflow(job=None,
                    gan_config=None,
                    miss_csv=None,
                    storage_root=None,
                    dataset_csv_path=None,
                    doc_dir=None):
    process_logger.info("NODE:25, Accuracy >= Target so Starting - GAN workflow")
    process_logger.info("NODE:25, Finding similar data for GAN training..")
    try:    
        miss_flag = validate_dataset(file_path=miss_csv, data_type='miss')
        train_dataset_flag = validate_dataset(file_path=dataset_csv_path, data_type='train')
        if not miss_flag or not train_dataset_flag:
            print(f"Please check dataset {dataset_csv_path} and {miss_csv}")
            process_logger.error(f"NODE:88, Please check dataset {dataset_csv_path} and {miss_csv}")
            raise
        
        similar = api_similar_data(train_dir=None, csv_path=miss_csv, output_folder=storage_root, train_csv=dataset_csv_path)
        # print("similar-----------------------", similar)
        first_path = None
        if len(similar) > 0:
            first_path = next(iter(similar.values()))[0]
            process_logger.progress(80.0)
            store_json(job, "similar_data", similar)
            process_logger.info(f"NODE:26, GAN training started for - {first_path}")
            remove_empty_subfolders(parent_folder=first_path)
            if not os.listdir(first_path):
                _analyse_and_pause_(job, doc_dir, dataset_csv_path, cycle_num=1, status=25)
                return 
    except Exception as e:
        print(f"Error in find similar/first path: {e}")
        process_logger.info(f"NODE:26, Error in find similar/first path: {e}")
        process_logger.error(f"NODE:88, Error in find similar/first path: {e}")
        raise
    
    try:
        if first_path is None:
            _analyse_and_pause_(job, doc_dir, dataset_csv_path, cycle_num=1, status=25)
            return 
        else:
            gan_path, best_result_path = api_GAN(gan_config, first_path, v_path=storage_root, status=26)
            process_logger.info(f"NODE:26, api gan results - gan_path: {gan_path}, best_result_path: {best_result_path}")
    except Exception as e:
        print(f"Error in api gan: {e}")
        process_logger.info(f"NODE:26, GAN training failed")
        process_logger.error(f"NODE:88, GAN training failed")
        raise
    
    try:
        result_path = create_clusters(image_folder=first_path, output=storage_root)
        process_logger.info(f"NODE:26, Cluster result path - {result_path}")
    except Exception as e:
        print(f"Error in cluster creation: {e}")
        process_logger.info(f"NODE:26, Clustering failed: {e}")
        process_logger.error(f"NODE:88, Clustering failed.")
        raise
    process_logger.info(f"NODE:27, Classes Cluster path: {result_path}")
    process_logger.info("NODE:27, Calculating FID Score....")

    try:
        base_fid, gan_fid = get_fid_calculation(cluster_folder_path=result_path,
                                                    gan_input_path=first_path,
                                                    gan_output_path=best_result_path,
                                                    storage_folder=storage_root)
    except Exception as e:
        print(f"Error in fid calculation: {e}")
        process_logger.info(f"NODE:27, FID calculation failed: {e}")
        process_logger.error(f"NODE:88, FID calculation failed.")
        raise
    
    store_json(job, "BASE_FID_Score", base_fid)
    store_json(job, "GAN_FID_Score", gan_fid)
    process_logger.info(f"NODE:27, BASE FID score - {base_fid} and GAN FID score - {gan_fid}")
   
    if gan_fid < base_fid * FID_COEFFICIENT:
    # if False:
        process_logger.progress(90.0)
        process_logger.info(f"NODE:27, Validating generated dataset-{gan_path}")
        try:
            validator_result = api_validator(input_folder=gan_path, output_folder=first_path, storage_folder=storage_root)
            process_logger.info(f"NODE:27, Validated dataset-{validator_result}")
            process_logger.info(f"NODE:81, GAN review")

        except Exception as e:
            print(f"Error in api validator: {e}")
            process_logger.info(f"NODE:27, API validator failed: {e}")
            process_logger.error(f"NODE:88, API validator failed.")
            raise
        store_json(job, "validator_data", validator_result)
        store_json(job, "gan_path", gan_path)
        
        process_logger.progress(100.0)
        return
    else:
        process_logger.info(f"NODE:27, GAN FID score - {gan_fid} > BASE FID * 1.5 ({1.5 * base_fid})")
        process_logger.info(f"NODE:35, Accuracy < Target - Analysis workflow, cycle 1/{MAX_ANALYSE_CYCLES}")
        _analyse_and_pause_(job, doc_dir, dataset_csv_path, cycle_num=1, status=25)
        return
def export_conf_files(storage_root=None, label=None, which_data=None, task_type="classification"):
    model_name = get_latest_model_name(storage_root=storage_root)
    source_folder = os.path.join(storage_root, f"model\Test\{model_name}")
    dest_folder = os.path.join(storage_root, f"model\Exported_model_{label}")
    os.makedirs(dest_folder, exist_ok=True)
    if task_type == "segmentation":
        files = [
            ("IouMatrixTest.txt", f"IouMatrixTest_{which_data}.txt"),
            ("Test_IoU.csv",      f"Test_IoU_{which_data}.csv"),
        ]
    else:
        files = [
            ("ConfMatrixTest.txt",                            f"ConfMatrixTest_{which_data}.txt"),
            (f"Test_Miss_Overkill_{which_data}.csv",          f"Test_Miss_Overkill_{which_data}.csv"),
            (f"Test_Miss_Overkill_{which_data}_All.csv",      f"Test_Miss_Overkill_{which_data}_All.csv"),
        ]
    for src_name, dst_name in files:
        src = os.path.join(source_folder, src_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest_folder, dst_name))
        else:
            process_logger.error(f"NODE:88, export_conf_files: {src_name} not found, skipping")


def run_model_improvement(job=None,
                          dataset_csv_path=None,
                          storage_root=None,
                          task_type=None,
                          hparams=None):
    print(">>> FLOW C: model improvement flow")

    
    
    setup_model_directories(storage_root=storage_root, import_files=job['config']["modelConfig"], task_type=task_type)
    process_logger.info("NODE:11, model_config present")
    process_logger.info("NODE:11, Starting model improvement process")
    process_logger.info("NODE:11, Computing baseline accuracy")

    df = pd.read_csv(dataset_csv_path)
    classes = df['class_label'].unique().tolist()
    frontier_dir = frontier_changes_(classes, storage_root, label="Improvement", task_type=task_type)

    base_inf = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="Baseline Deployed", status=11, training_type=task_type)
    export_conf_files(storage_root=storage_root, label="Deployed_param", which_data="Baseline", task_type=task_type)
    
    new_test_inf = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="New_Test Deployed", status=11, training_type=task_type)
    export_conf_files(storage_root=storage_root, label="Deployed_param", which_data="New_Test", task_type=task_type)
    
    export_save_path = api_export_model(storage_root=frontier_dir, training_type=task_type, status=11, label="Deployed_param")

    base_accuracy = extract_accuracy(base_inf)
    miss_rate, overkill_rate = extract_miss_overkill(result=base_inf)
    new_test_miss_rate, new_test_overkill_rate = extract_miss_overkill(result=new_test_inf)

    acc_label = "MIoU" if task_type == "segmentation" else "Accuracy"
    new_test_acc = extract_accuracy(new_test_inf)
    res         = f"{base_accuracy:.4f}" if base_accuracy is not None else "N/A"
    new_test_res = f"{new_test_acc:.4f}" if new_test_acc is not None else "N/A"
    process_logger.info(f"NODE:11, Baseline {acc_label}: {res}")
    process_logger.info(f"NODE:11, New Test {acc_label}: {new_test_res}")
    if task_type != "segmentation":
        miss_res     = f"{miss_rate:.4f}" if miss_rate is not None else "N/A"
        overkill_res = f"{overkill_rate:.4f}" if overkill_rate is not None else "N/A"
        new_test_miss_res     = f"{new_test_miss_rate:.4f}" if new_test_miss_rate is not None else "N/A"
        new_test_overkill_res = f"{new_test_overkill_rate:.4f}" if new_test_overkill_rate is not None else "N/A"
        process_logger.info(f"NODE:11, Miss rate: {miss_res}, Overkill rate: {overkill_res}")
        process_logger.info(f"NODE:11, New Test Miss rate: {new_test_miss_res}, Overkill rate: {new_test_overkill_res}")
    
    store_json(job, "base_accuracy", base_inf)
    store_json(job, "New_Test_Base", new_test_inf)
    store_json(job, "frontier_dir", frontier_dir)
    best_acc = base_accuracy
    best_acc_result = base_inf
    best_miss_rate = miss_rate
    best_overkill_rate = overkill_rate
    best_new_test_miss_rate = new_test_miss_rate
    best_new_test_overkill_rate = new_test_overkill_rate
    process_logger.progress(10.0)

    process_logger.info("NODE:11, Merging Baseline dataset and new dataset")
    process_logger.progress(15.0)
    process_logger.info("NODE:12, Training model with base hyper params")

    imported_model_json = os.path.join(frontier_dir, "Imported_model", "Training.json")
    with open(imported_model_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_fields = data.get("Model", {})
    process_logger.info(
        f"NODE:12, Imported model Training.json Model fields - "
        f"fBaseLR: {model_fields.get('fBaseLR')}, epochs: {model_fields.get('epochs')}, "
        f"iBatchSize: {model_fields.get('iBatchSize')}, valRatio: {model_fields.get('valRatio')}, "
        f"minEpoch: {model_fields.get('minEpoch')}, patience: {model_fields.get('patience')}"
    )

    _cfg = hparams or {}
    base_hparams = {
        'lr':         model_fields.get('fBaseLR',   _cfg.get('lr',         0.0001)),
        'epochs':     model_fields.get('epochs',    _cfg.get('epochs',     100)) if not DEBUG_MODE else DEBUG_EPOCHS,
        'batch_size': model_fields.get('iBatchSize', _cfg.get('batch_size', 4)),
        'valRatio':   model_fields.get('valRatio',  _cfg.get('valRatio',   20)),
        'minEpoch':   model_fields.get('minEpoch',  _cfg.get('minEpoch',   0)),
        'patience':   model_fields.get('patience',  _cfg.get('patience',   50)),
        'fLRDecay':   model_fields.get('fLRDecay',  _cfg.get('fLRDecay',   0.5))
    }
    print(f"{'*'*20} Base_hparams values: {base_hparams}")
    api_train(dataset_csv_path=dataset_csv_path, hyperparameters=base_hparams, storage_root=frontier_dir, label="Round 1", status=12, training_type=task_type)
    process_logger.info("NODE:12, Model Training Completed & model Saved")
    process_logger.progress(60.0)

    inf1 = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="Baseline Base_param", status=13, training_type=task_type)
    
    export_conf_files(storage_root=storage_root, label="Base_param", which_data="Baseline", task_type=task_type)
    new_test_inf1 = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="New_Test Base_param", status=13, training_type=task_type)
    export_conf_files(storage_root=storage_root, label="Base_param", which_data="New_Test", task_type=task_type)
    
    store_json(job, "Best_acc-NODE:13", inf1)
    store_json(job, "New_Test-NODE:13", new_test_inf1)
    store_json(job, "New_Test", new_test_inf1)
    process_logger.info("NODE:14, Checking model performance..")

    improved, best_acc, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate = check_performance_gain(
        inf1, new_test_inf1, base_accuracy, "After Round 1", status=14,
        best_miss_rate=miss_rate, best_overkill_rate=overkill_rate,
        best_new_test_miss_rate=new_test_miss_rate, best_new_test_overkill_rate=new_test_overkill_rate,
        task_type=task_type)

    store_json(job, "Best_acc", inf1)
    process_logger.progress(65.0)
    export_save_path = api_export_model(storage_root=frontier_dir, training_type=task_type, status=14, label="Base_param")

    if improved and DEBUG_IMPROVED:
        acc_label = "MIoU" if task_type == "segmentation" else "Accuracy"
        res = f"{best_acc:.4f}" if best_acc is not None else "N/A"
        process_logger.info(f"NODE:14, Best {acc_label}: {res}")
        if task_type != "segmentation":
            miss_res     = f"{best_miss_rate:.4f}" if best_miss_rate is not None else "N/A"
            overkill_res = f"{best_overkill_rate:.4f}" if best_overkill_rate is not None else "N/A"
            nt_miss_res  = f"{best_new_test_miss_rate:.4f}" if best_new_test_miss_rate is not None else "N/A"
            nt_ok_res    = f"{best_new_test_overkill_rate:.4f}" if best_new_test_overkill_rate is not None else "N/A"
            process_logger.info(f"NODE:14, Miss rate: {miss_res}, Overkill rate: {overkill_res}")
            process_logger.info(f"NODE:14, New Test Miss rate: {nt_miss_res}, Overkill rate: {nt_ok_res}")
        improved_path = os.path.join(os.path.dirname(export_save_path), "improved")
        shutil.copytree(export_save_path,improved_path, dirs_exist_ok=True)
        process_logger.info("NODE:14, Improved after Round 1 - STOPPING")
        process_logger.info(f"NODE:80, Improved model saved at - {improved_path}")            
            

        # _save_improved_zip(frontier_dir,storage_root, label="base_training", best_accuracy=best_acc, miss_rate=best_miss_rate, overkill_rate=best_overkill_rate)

        # api_export_model(storage_root=frontier_dir, training_type=task_type, status=14)
        process_logger.progress(100.0)
        return improved, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate, frontier_dir, None, None

    process_logger.info("NODE:21, Round 2(hyperparams) tunning")
    process_logger.info("NODE:22, Round 2 Training with (hyperparams) tunning")
    OPTUNA_PARAMS['epochs'] = base_hparams['epochs']
    train_results = api_train(dataset_csv_path=dataset_csv_path, hyperparameters=OPTUNA_PARAMS, storage_root=frontier_dir, label="OPTUNA", status=22, training_type=task_type)
    process_logger.info("NODE:22, Model Training Completed & model Saved")
    train_dir = train_results["result"]["train_dir"]
    store_json(job, "train_data", train_dir)
    process_logger.progress(70.0)

    inf2 = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="Baseline Optuna", status=23, training_type=task_type)
    export_conf_files(storage_root=storage_root, label="Hyper_param", which_data="Baseline", task_type=task_type)
    new_test_inf2 = api_inference(dataset_csv_path=dataset_csv_path, storage_root=frontier_dir, label="New_Test Optuna", status=23, training_type=task_type)
    export_conf_files(storage_root=storage_root, label="Hyper_param", which_data="New_Test", task_type=task_type)

    process_logger.info("NODE:24, Checking model performance..")
    improved_vs_best, best_acc, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate = check_performance_gain(
        inf2, new_test_inf2, base_accuracy, "After Round 2", status=24,
        best_miss_rate=miss_rate, best_overkill_rate=overkill_rate,
        best_new_test_miss_rate=new_test_miss_rate, best_new_test_overkill_rate=new_test_overkill_rate,
        task_type=task_type)
    process_logger.progress(75.0)

    r1_miss_rate, r1_overkill_rate = extract_miss_overkill(inf1)
    r2_miss_rate, r2_overkill_rate = extract_miss_overkill(inf2)
    new_test_r1_miss_rate, new_test_r1_overkill_rate = extract_miss_overkill(new_test_inf1)
    new_test_r2_miss_rate, new_test_r2_overkill_rate = extract_miss_overkill(new_test_inf2)

    if task_type == "segmentation":
        r1_acc = extract_accuracy(inf1)
        r2_acc = extract_accuracy(inf2)
        round2_better = (r1_acc is not None and r2_acc is not None and r2_acc > r1_acc)
    else:
        data = {
            'baseline': {
                'r1_miss': r1_miss_rate, 'r1_overkill': r1_overkill_rate, 
                'r2_miss': r2_miss_rate, 'r2_overkill': r2_overkill_rate
            },
            'new_test': {
                'r1_miss': new_test_r1_miss_rate, 'r1_overkill': new_test_r1_overkill_rate, 
                'r2_miss': new_test_r2_miss_rate, 'r2_overkill': new_test_r2_overkill_rate
            }
        }

        # Miss = 80%, Overkill = 20%
        r2_result = is_r2_better_than_r1(data, miss_weight=0.8, overkill_weight=0.2)
        # round2_better = (
        #     (r2_miss_rate and r2_overkill_rate and r1_miss_rate and r1_overkill_rate) and
        #     (r2_miss_rate <= r1_miss_rate and r2_overkill_rate <= r1_overkill_rate) and
        #     (new_test_r2_miss_rate <= new_test_r1_miss_rate and new_test_r2_overkill_rate <= new_test_r1_overkill_rate)
        # )
        round2_better = r2_result

    if round2_better:
        active_hparams = train_results["result"]['best_params']
        process_logger.info("NODE:24, Round 2 > Round 1 --> tuned hyperparams will be used going forward.")
        miss_csv = inf2.get('csv_path')
        miss_csv = concatenate_csv_files(file1_path=inf2.get('csv_path'), file2_path=new_test_inf2.get('csv_path'), output_filename=f"{storage_root}/Doc/merged_miss_csv.csv")
    else:
        active_hparams = base_hparams
        miss_csv = inf1.get('csv_path')
        file1_path = os.path.join(storage_root, 'model/Exported_model_Base_param', 'Test_Miss_Overkill_Baseline.csv')
        file2_path = os.path.join(storage_root, 'model/Exported_model_Base_param', 'Test_Miss_Overkill_New_Test.csv')
        process_logger.info("NODE:24, Round 2 <= Round 1 --> base hyperparams retained.")
        # miss_csv = concatenate_csv_files(file1_path=inf1.get('csv_path'), file2_path=new_test_inf1.get('csv_path'), output_filename=f"{storage_root}/Doc/merged_miss_csv.csv")
        miss_csv = concatenate_csv_files(file1_path=file1_path, file2_path=file2_path, output_filename=f"{storage_root}/Doc/merged_miss_csv.csv")

    store_json(job, "active_hparams", active_hparams)
    # if improved_vs_best:
    #     best_acc_result = inf2
    store_json(job, "Best_acc", inf2)
    store_json(job, "Best_acc-NODE:24", inf2)
    store_json(job, "New_Test", new_test_inf2)
    store_json(job, "New_Test-NODE:24", new_test_inf2)
    export_save_path =  api_export_model(storage_root=frontier_dir, training_type=task_type, status=24, label="Hyper_param")

    if improved_vs_best and DEBUG_IMPROVED:
        acc_label = "MIoU" if task_type == "segmentation" else "Accuracy"
        res = f"{best_acc:.4f}" if best_acc is not None else "N/A"
        process_logger.info(f"NODE:24, Best {acc_label}: {res}")
        if task_type != "segmentation":
            miss_res     = f"{best_miss_rate:.4f}" if best_miss_rate is not None else "N/A"
            overkill_res = f"{best_overkill_rate:.4f}" if best_overkill_rate is not None else "N/A"
            nt_miss_res  = f"{best_new_test_miss_rate:.4f}" if best_new_test_miss_rate is not None else "N/A"
            nt_ok_res    = f"{best_new_test_overkill_rate:.4f}" if best_new_test_overkill_rate is not None else "N/A"
            process_logger.info(f"NODE:24, Miss rate: {miss_res}, Overkill rate: {overkill_res}")
            process_logger.info(f"NODE:24, New Test Miss rate: {nt_miss_res}, Overkill rate: {nt_ok_res}")
        improved_path = os.path.join(os.path.dirname(export_save_path), "improved")
        shutil.copytree(export_save_path,improved_path, dirs_exist_ok=True)
        process_logger.info("NODE:24, Improved after Round 2 - STOPPING")
        process_logger.info(f"NODE:80, Improved model saved at - {improved_path}")


        # _save_improved_zip(frontier_dir,storage_root, label="base_training", best_accuracy=best_acc, miss_rate=best_miss_rate, overkill_rate=best_overkill_rate)
        # api_export_model(storage_root=frontier_dir, training_type=task_type, status=24)

        process_logger.progress(100.0)
        return improved_vs_best, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate, frontier_dir, None, None
    elif improved_vs_best == False and task_type == "segmentation" :
        process_logger.info("NODE:82, Round 2 did not improve MIoU - waiting for updated dataset")

    return improved_vs_best, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate, frontier_dir, miss_csv, inf2

def run_pipeline(job: dict, hyperparameters: Optional[Dict[str, Any]] = None) -> None:
    try:
        
        cfg, GAN_STATUS_CODE, dataset_csv_path, storage_root, doc_dir, task_type, gan_config, hparams, best_acc = initialize(job)
        
        print(f"run_training_with_gan: dataset_csv_path={dataset_csv_path},storage_root={storage_root},task_type={task_type},hparams={hparams}")
        if (len(cfg['status']) > 1) and cfg.get("status", [])[-2] == GAN_STATUS_CODE:
            try:
                return run_training_with_gan(job=job,
                                            dataset_csv_path=dataset_csv_path,
                                            storage_root=storage_root,
                                            task_type=task_type,
                                            doc_dir=doc_dir)  
            except Exception as e:
                print(f"Error in run_training_with_gan: {e}")
                process_logger.error(f"NODE:88, Issue in run_training_with_gan: {e}")
                raise
        
        print(f"run_training_with_analysis: dataset_csv_path={dataset_csv_path},storage_root={storage_root},task_type={task_type},hparams={hparams}")
        if (len(cfg['status']) > 1) and cfg.get("status", [])[-1]!=1:
            if task_type == "segmentation":
                try:
                    return run_segmentation_retry(job=job,
                                    dataset_csv_path=dataset_csv_path,
                                    storage_root=storage_root,
                                    task_type=task_type,
                                    doc_dir=doc_dir)
                except Exception as e:
                    print(f"Error in run_segmentation_retry: {e}")
                    process_logger.error(f"NODE:88, Issue in run_segmentation_retry: {e}")
                    raise
            try:
                return run_training_with_analysis(job=job,
                                dataset_csv_path=dataset_csv_path,
                                storage_root=storage_root,
                                task_type=task_type,
                                doc_dir=doc_dir)
            except Exception as e:
                print(f"Error in run_training_with_analysis: {e}")
                process_logger.error(f"NODE:88, Issue in run_training_with_analysis: {e}")
                raise

        print(f"run_base_training: dataset_csv_path={dataset_csv_path},storage_root={storage_root},task_type={task_type},hparams={hparams}")
        if cfg.get("Base_training", False):
            try:
                return run_base_training(job=job,
                            dataset_csv_path=dataset_csv_path,
                            storage_root=storage_root,
                            task_type=task_type,
                            best_acc=best_acc,
                            hparams=hparams)
            except Exception as e:
                print(f"Error in run_base_training: {e}")
                process_logger.error(f"NODE:88, Issue in run_base_training: {e}")
                return

        print(f"run_model_improvement: dataset_csv_path={dataset_csv_path},storage_root={storage_root},task_type={task_type},hparams={hparams}")
        try:
            improved, best_miss_rate, best_overkill_rate, best_new_test_miss_rate, best_new_test_overkill_rate, frontier_dir, miss_csv, inf2 = run_model_improvement(job=job,
                            dataset_csv_path=dataset_csv_path,
                            storage_root=storage_root,
                            task_type=task_type,
                            hparams=hparams)
        except Exception as e:
            print(f"Error in run_model_improvement: {e}")
            process_logger.error(f"NODE:88, Issue in run_model_improvement: {e}")
            raise
        
        if improved and DEBUG_IMPROVED:
            return
        
        latest_miss_rate, latest_overkill_rate = extract_miss_overkill(inf2)

        if task_type == "classification" and latest_miss_rate is not None and latest_overkill_rate is not None:
            try:
                return run_gan_worflow(
                        job=job,
                        gan_config=gan_config,
                        miss_csv=miss_csv,
                        storage_root=storage_root,
                        dataset_csv_path=dataset_csv_path,
                        doc_dir=doc_dir
                )
            except Exception as e:
                print(f"Error in run_gan_worflow: {e}")
                process_logger.error(f"NODE:88, Issue in run_gan_worflow {e}")
                raise
        elif task_type in ("segmentation", "objectdetection"):
            process_logger.progress(100.0)
            return
        else:
            process_logger.info(f"NODE:25, Accuracy < Target — Analysis workflow, cycle 1/{MAX_ANALYSE_CYCLES}")
            _analyse_and_pause_(job, doc_dir, dataset_csv_path, cycle_num=1, status=25)
            return

    except Exception as e:
        process_logger.error(f"NODE:88, Failed to run job, please try again .... {e}")
        raise 

# if __name__ == "__main__":
#     miss_csv = r"\\107.108.32.206\plato\Plato_206_storage\Classification\Plato_2004\Doc\merged_miss_csv.csv"
#     dataset_csv_path = r"\\107.108.32.206\plato\Plato_206_storage\Classification\Plato_2004\train_dataset.csv"
#     storage_root = r"\\107.108.32.206\plato\Plato_206_storage\Classification\Plato_2004"
#     # run_automated_training(custom_params={"epochs":200}, dataset_csv_path=r"Z:\PLATO\Classification\Plato_618192005\train_dataset.csv", storage_root=storage_root, label="GAN-validated Training", status1=12, training_type="classification")
#     similar = api_similar_data(train_dir=None, csv_path=miss_csv, output_folder=storage_root, train_csv=dataset_csv_path)    
#     # breakpoint()
#     first_path = None
#     if len(similar) > 0:
#         first_path = next(iter(similar.values()))[0]
#         process_logger.progress(80.0)
#         # store_json(job, "similar_data", similar)
#         process_logger.info(f"NODE:26, GAN training started for - {first_path}")
#         remove_empty_subfolders(parent_folder=first_path)
#         if not os.listdir(first_path):
#             print('no similar data')
# if __name__ == "__main__":
    # miss_csv = r"Z:\PLATO\Classification\Plato_5500\Doc\merged_miss_csv.csv"
    # dataset_csv_path = r"Z:\PLATO\Classification\Plato_5500\train_dataset.csv"
    # result = run_gan_worflow(
    #     miss_csv=miss_csv,
    #     dataset_csv_path=dataset_csv_path
    # )
    # storage_root = "gan_dir"
    # passed = []
    # for folder in os.listdir("Z:/PLATO/Classification"):
    #     print(f"Processing .............. {folder}")
    #     try:
    #         train_csv_path = f"Z:/PLATO/Classification/{folder}/train_dataset.csv"
    #         miss_csv_path = f"Z:/PLATO/Classification/{folder}/Doc/merged_miss_csv.csv"
    #         miss_flag = validate_dataset(miss_csv_path, data_type='miss')
    #         print(f"miss_flag................", miss_flag)
    #         if not miss_flag:
    #             continue
    #         train_csv_flag = validate_dataset(train_csv_path, data_type='train')
    #         print(f"train_csv_flag.................{train_csv_flag}")
    #         if miss_flag and train_csv_flag: 
    #             similar = api_similar_data(train_dir=None, csv_path=miss_csv_path, output_folder=storage_root, train_csv=train_csv_path)
    #             first_path = next(iter(similar.values()))[0]
    #             passed.append(folder)

    #     except Exception as e:
    #         print(e)
    #         breakpoint()

    # print(passed)
    # print(len(passed))
    # breakpoint()


    # gan_config = r"E:\Rohit\plato_ai_python_2\plato_ai_python\backend\req_files\classification\config_gan.json" 
    # first_path = r"\\107.108.32.106\Pratik\pratik\SuwonOffice_dataset\Plato_dataset\PSMT dataset\classification_dataset\Frontier_Data_3225 BAND 512x400 CLASS_20251218_154003_good\Datasets\3225 BAND 512x400 CLASS\Valid"
    # storage_root = "gan_dir"
    # if not os.path.exists(storage_root):
    #     os.makedirs(storage_root)
    # gan_path, best_result_path = api_GAN(gan_config, first_path, v_path=storage_root, status=26)

    # job = {
    #     "server_ip": "107.99.131.65",
    #     "server_port": 9000,
    #     "backend_ip": "107.99.131.65",
    #     "backend_port": 8011,
    #     "trigger_id": "ALERT_M11_M12",
    #     "config": {
    #         "triggerId": "chechk_alert_1",
    #         "Base_training": False,
    #         "modelConfig": r"Z:\PLATO\Classification\Plato_9161\model\Imported_model",
    #         "dataset_csv_path": r"Z:\PLATO\Classification\Plato_9161\train_dataset.csv",
    #         "storage_path": r"Z:\PLATO",
    #         "status": [0],
    #     },
    #     "gpu_id": 859,
    #     "gpu_no": 0,
    #     "job": "Classification",
    #     "job_type": "gpu",
    #     "logs": "Job assigned.",
    #     "progress": 0.0,
    #     "error": None,
    #     "allowNextJob": False,
    #     "system_id": 910,
    # }
    # cfg, GAN_STATUS_CODE, dataset_csv_path, storage_root, doc_dir, task_type, gan_config, hparams, best_acc = initialize(job, hyperparameters=None)
    # print(cfg, GAN_STATUS_CODE, dataset_csv_path, storage_root, doc_dir, task_type, gan_config, hparams, best_acc)
    # run_base_training(job=job,
    #                         dataset_csv_path=dataset_csv_path,
    #                         storage_root=storage_root,
    #                         task_type=task_type,
    #                         best_acc=best_acc,
    #                         hparams=hparams)

    # run_pipeline(job=job)
    
# if __name__ == "__main__":
#     job = {
#         "server_ip": "107.99.131.65",
#         "server_port": 9000,
#         "backend_ip": "107.99.131.65",
#         "backend_port": 8011,
#         "trigger_id": "ALERT_M11_M12",
#         "config": {
#             "triggerId": "Plato_45004",
#             "Base_training": False,
#             "modelConfig": r"Z:\PLATO_Bench\Classification\Plato_45002\model\Imported_model",
#             "dataset_csv_path": r"Z:\PLATO_Bench\Classification\Plato_45002\train_dataset.csv",
#             "storage_path": r"Z:\PLATO",
#             "status": [0],
#         },
#         "gpu_id": 859,
#         "gpu_no": 0,
#         "job": "Classification",
#         "job_type": "gpu",
#         "logs": "Job assigned.",
#         "progress": 0.0,
#         "error": None,
#         "allowNextJob": False,
#         "system_id": 910,
#     }
#     cfg, GAN_STATUS_CODE, dataset_csv_path, storage_root, doc_dir, task_type, gan_config, hparams, best_acc = initialize(job, hyperparameters=None)
#     print(cfg, GAN_STATUS_CODE, dataset_csv_path, storage_root, doc_dir, task_type, gan_config, hparams, best_acc)
#     run_model_improvement(job=job,
#                             dataset_csv_path=dataset_csv_path,
#                             storage_root=storage_root,
#                             task_type=task_type,
#                             hparams=hparams)
#     run_pipeline(job=job)

 
# if __name__ == "__main__":
#     job = {
#         "server_ip": "107.99.131.65",
#         "server_port": 9000,
#         "backend_ip": "107.99.131.65",
#         "backend_port": 8011,
#         "trigger_id": "ALERT_M11_M12",
#         "config": {
#             "triggerId": "chechk_alert",
#             "Base_training": False,
#             "modelConfig": r"D:\Shared\current_task\dev\temp\export3\Frontier_Data_classification_20260601_170315\Models\Model_1_20260601_170319",
#             "dataset_csv_path": r"D:\Shared\current_task\dev\temp\dataset3\train_dataset.csv",
#             "storage_path": r"D:\Shared\current_task\dev\storage_path",
#             "status": [0],
#         },
#         "gpu_id": 859,
#         "gpu_no": 0,
#         "job": "Classification",
#         "job_type": "gpu",
#         "logs": "Job assigned.",
#         "progress": 0.0,
#         "error": None,
#         "allowNextJob": False,
#         "system_id": 910,
#     }


#     run_pipeline(job=job)

    # print(passed)
    # print(len(passed))
    # breakpoint()


    # gan_config = r"E:\Rohit\plato_ai_python_2\plato_ai_python\backend\req_files\classification\config_gan.json" 
    # first_path = r"Z:\PLATO\Classification\Plato_5556\copied_files_dir\20260603_170008_similar"
    # storage_root = "gan_dir"
    # if not os.path.exists(storage_root):
    #     os.makedirs(storage_root)
    # gan_path, best_result_path = api_GAN(gan_config, first_path, v_path=storage_root, status=26)
# 
if __name__ == "__main__":
    job = {
        "server_ip": "107.99.131.65",
        "server_port": 9000,
        "backend_ip": "107.99.131.65",
        "backend_port": 8011,
        "trigger_id": "ALERT_M11_M12",
        "config": {
            "triggerId": "Plato_9060_rohit9",
            "Base_training": False,
            "modelConfig": "Z:/PLATO/Objectdetection/Plato_1201112/model/Imported_model",
            "dataset_csv_path": "Z:/PLATO/Objectdetection/Plato_1201112/train_dataset.csv",
            "storage_path": "Z:/PLATO",
            "status": [0],
        },
        "gpu_id": 859,
        "gpu_no": 0,
        "job": "Objectdetection",
        "job_type": "gpu",
        "logs": "Job assigned.",
        "progress": 0.0,
        "error": None,
        "allowNextJob": False,
        "system_id": 910,
    }

    run_pipeline(job=job)