import argparse
import json
from pathlib import Path
from typing import Any


def normalize_for_dedup(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_for_dedup(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [normalize_for_dedup(item) for item in value]
    return value


def deduplicate_list(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    merged: list[Any] = []

    for item in items:
        signature = json.dumps(normalize_for_dedup(item), sort_keys=True, separators=(",", ":"))
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(item)

    return merged


def merge_coco_jsons(first_path: Path, second_path: Path, output_path: Path) -> dict[str, Any]:
    with first_path.open("r", encoding="utf-8") as handle:
        first_data = json.load(handle)

    with second_path.open("r", encoding="utf-8") as handle:
        second_data = json.load(handle)

    if not isinstance(first_data, dict) or not isinstance(second_data, dict):
        raise ValueError("Both input files must contain JSON objects")

    merged: dict[str, Any] = {}

    for key in sorted(set(first_data.keys()) | set(second_data.keys())):
        left_value = first_data.get(key)
        right_value = second_data.get(key)

        if isinstance(left_value, list) and isinstance(right_value, list):
            merged[key] = deduplicate_list(left_value + right_value)
        elif key in first_data and key in second_data:
            merged[key] = left_value
        elif key in first_data:
            merged[key] = left_value
        else:
            merged[key] = right_value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
        handle.write("\n")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two annotation JSON files without dropping unique items")
    parser.add_argument("--first", default="training_annotations_final.json", help="Path to the first JSON file")
    parser.add_argument("--second", default="training_annotations_with_negatives.json", help="Path to the second JSON file")
    parser.add_argument("--output", default="merged_annotations.json", help="Path to write the merged JSON file")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    first_path = Path(args.first)
    second_path = Path(args.second)
    output_path = Path(args.output)

    if not first_path.is_absolute():
        first_path = workspace / first_path
    if not second_path.is_absolute():
        second_path = workspace / second_path
    if not output_path.is_absolute():
        output_path = workspace / output_path

    merged = merge_coco_jsons(first_path, second_path, output_path)

    for key in ["images", "annotations", "categories"]:
        if key in merged:
            print(f"{key}: {len(merged[key])}")

    print(f"Merged JSON written to: {output_path}")


if __name__ == "__main__":
    main()
