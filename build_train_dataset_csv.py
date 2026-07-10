"""
Test-only helper: reverse of the object-detection CSV -> JSON pipeline in
training_service.py. Reads Input_Train.json / Input_Val.json / a Test-side d.json
(ANNOTATIONS format: ImagePath/MaskPath/Rect/Points/ImgW/ImgH, where Points is a list of
per-object {"CId": "<class>", "X": [x1..x4], "Y": [y1..y4]} dicts in absolute pixel coords)
and reconstructs:

  - a YOLO-OBB label .txt per image (class_index x1 y1 x2 y2 x3 y3 x4 y4, coords
    re-normalized to [0,1] using ImgW/ImgH) written under --labels-dir
  - train_dataset.csv with columns:
        s.no., image_id, image_url, class_type, class_label, Data_type, set, model_bbox_url
    where model_bbox_url points at the regenerated .txt file above

This exists to round-trip csv_scan_objdet_annotations()/_parse_yolo_obb_txt(): run the
forward pipeline on some CSV, run this script on its JSON output, then run the forward
pipeline again and diff the two JSON outputs.

class_type/class_label are NOT present anywhere in the ANNOTATIONS JSONs (that info only
ever lived in the original CSV, not the dataset descriptor files), so they're filled in
with a fixed placeholder value here — good enough for round-tripping the dataset-generation
shape, not a faithful reconstruction of the original CSV's class data.

Usage:
    python build_train_dataset_csv.py \
        --input-train Input_Train.json \
        --input-val   Input_Val.json \
        --input-test  d.json \
        --output      train_dataset.csv \
        --labels-dir  labels
"""
import argparse
import csv
import json
import os


def _load_annotations(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("ANNOTATIONS", []) or []


def _image_id(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def _write_label_txt(entry, labels_dir):
    """Re-normalize entry['Points'] (absolute pixel corners) back into a YOLO-OBB .txt
    file. Returns the written path, or "" if there are no points / dims to normalize by."""
    points = entry.get("Points") or []
    img_w, img_h = entry.get("ImgW"), entry.get("ImgH")
    if not points or not img_w or not img_h:
        return ""

    lines = []
    for obj in points:
        xs, ys = obj.get("X") or [], obj.get("Y") or []
        if len(xs) != len(ys) or not xs:
            continue
        coords = []
        for x, y in zip(xs, ys):
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        lines.append(" ".join([str(obj.get("CId", "0"))] + coords))

    if not lines:
        return ""

    os.makedirs(labels_dir, exist_ok=True)
    txt_path = os.path.join(labels_dir, f"{_image_id(entry.get('ImagePath', ''))}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return txt_path


def build_rows(train_entries, val_entries, test_entries, class_label, labels_dir):
    rows = []
    for set_name, entries in (("Train", train_entries), ("Val", val_entries), ("Test", test_entries)):
        for entry in entries:
            rows.append({
                "image_id":       _image_id(entry.get("ImagePath", "")),
                "image_url":      entry.get("ImagePath", ""),
                "class_type":     class_label,
                "class_label":    class_label,
                "Data_type":      "Base",
                "set":            set_name,
                "model_bbox_url": _write_label_txt(entry, labels_dir),
            })
    return rows


def write_csv(rows, output_path):
    fieldnames = ["s.no.", "image_id", "image_url", "class_type", "class_label",
                  "Data_type", "set", "model_bbox_url"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({"s.no.": i, **row})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-train", required=True, help="Path to Input_Train.json")
    parser.add_argument("--input-val",   required=True, help="Path to Input_Val.json")
    parser.add_argument("--input-test",  default=None,  help="Path to the Test-side d.json (optional)")
    parser.add_argument("--output",      default="train_dataset.csv", help="Output CSV path")
    parser.add_argument("--labels-dir",  default="labels", help="Directory to write regenerated YOLO-OBB .txt label files into (default: labels)")
    parser.add_argument("--class-label", default="OK", help="Placeholder value for class_type/class_label (default: OK)")
    args = parser.parse_args()

    train_entries = _load_annotations(args.input_train)
    val_entries   = _load_annotations(args.input_val)
    test_entries  = _load_annotations(args.input_test)

    rows = build_rows(train_entries, val_entries, test_entries, args.class_label, args.labels_dir)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows ({len(train_entries)} train, {len(val_entries)} val, "
          f"{len(test_entries)} test) to {args.output}, labels under {args.labels_dir}/")


if __name__ == "__main__":
    main()
