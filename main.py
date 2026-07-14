import csv
import json
import shutil
from pathlib import Path

from PIL import Image


def collect_csv_files(csv_dir: Path):
    return sorted(csv_dir.glob("*.csv"))


def merge_csvs(csv_files, output_csv: Path):
    fieldnames = None
    rows = []
    seen_filenames = set()

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                filename = (row.get("filename") or "").strip()
                if not filename:
                    continue
                if filename in seen_filenames:
                    continue
                seen_filenames.add(filename)
                rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["filename"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def copy_images_to_flat_folder(csv_rows, source_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for child in output_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    copied = []
    missing = []

    for row in csv_rows:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue

        match = None
        for candidate in source_root.rglob(filename):
            if candidate.is_file():
                match = candidate
                break

        if match is None:
            missing.append(filename)
            continue

        target = output_dir / filename
        if not target.exists():
            shutil.copy2(match, target)
        copied.append(filename)

    return copied, missing


def merge_annotations_into_coco(csv_rows, coco_data, images_dir: Path, category_id: int = 1):
    if "images" not in coco_data:
        coco_data["images"] = []
    if "annotations" not in coco_data:
        coco_data["annotations"] = []

    existing_images = {img.get("file_name"): img for img in coco_data.get("images", []) if img.get("file_name")}
    existing_annotations = set()
    for annotation in coco_data.get("annotations", []):
        image_id = annotation.get("image_id")
        segmentation = annotation.get("segmentation")
        bbox = tuple(annotation.get("bbox", []))
        if image_id is not None and segmentation is not None:
            existing_annotations.add((image_id, tuple(tuple(point) for group in segmentation for point in group), bbox))

    next_image_id = max((img.get("id", 0) for img in coco_data.get("images", []) if isinstance(img.get("id"), int)), default=0) + 1
    next_annotation_id = max((ann.get("id", 0) for ann in coco_data.get("annotations", []) if isinstance(ann.get("id"), int)), default=0) + 1

    added_images = 0
    added_annotations = 0

    for row in csv_rows:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue

        if filename not in existing_images:
            image_path = images_dir / filename
            width = height = 0
            file_size = int(row.get("file_size") or 0)
            if image_path.exists():
                with Image.open(image_path) as image:
                    width, height = image.size
                file_size = file_size or image_path.stat().st_size

            image_entry = {
                "id": next_image_id,
                "file_name": filename,
                "width": width,
                "height": height,
                "file_size": file_size,
            }
            coco_data["images"].append(image_entry)
            existing_images[filename] = image_entry
            next_image_id += 1
            added_images += 1

        image_id = existing_images[filename]["id"]

        try:
            shape = json.loads(row.get("region_shape_attributes") or "{}")
        except json.JSONDecodeError:
            continue

        if shape.get("name") != "polygon":
            continue

        xs = shape.get("all_points_x", [])
        ys = shape.get("all_points_y", [])
        points = [[int(x), int(y)] for x, y in zip(xs, ys)]
        if not points:
            continue

        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)

        area = 0.0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        area = abs(area) / 2.0

        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        annotation_key = (image_id, tuple(tuple(point) for point in points), tuple(bbox))
        if annotation_key in existing_annotations:
            continue

        annotation_entry = {
            "id": next_annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [points],
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
        }
        coco_data["annotations"].append(annotation_entry)
        existing_annotations.add(annotation_key)
        next_annotation_id += 1
        added_annotations += 1

    return added_images, added_annotations


def main():
    workspace = Path(__file__).resolve().parent
    csv_dir = workspace / "csv"
    source_root = workspace / "downloaded_images"
    output_dir = workspace / "merged_annotations_output" / "images"
    output_csv = workspace / "merged_annotations_output" / "merged_annotations.csv"

    csv_files = collect_csv_files(csv_dir)
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return

    print(f"Found {len(csv_files)} CSV files in {csv_dir}")
    rows = merge_csvs(csv_files, output_csv)
    copied, missing = copy_images_to_flat_folder(rows, source_root, output_dir)

    coco_json_path = workspace / "training_annotations_with_negatives.json"
    with coco_json_path.open("r", encoding="utf-8") as handle:
        coco_data = json.load(handle)

    added_images, added_annotations = merge_annotations_into_coco(rows, coco_data, output_dir)

    with coco_json_path.open("w", encoding="utf-8") as handle:
        json.dump(coco_data, handle, indent=2)
        handle.write("\n")

    print(f"Merged CSV saved to: {output_csv}")
    print(f"Unique image entries: {len(rows)}")
    print(f"Images copied: {len(copied)}")
    print(f"Images missing from downloaded_images: {len(missing)}")
    print(f"COCO images added: {added_images}")
    print(f"COCO annotations added: {added_annotations}")
    print(f"Updated COCO JSON: {coco_json_path}")
    if missing:
        print("Sample missing files:")
        for item in missing[:10]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()