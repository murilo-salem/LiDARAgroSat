from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd
import rasterio


RGB_EXTENSIONS = {".tif", ".tiff"}
LIDAR_EXTENSIONS = {".laz", ".las"}


@dataclass(slots=True)
class AnnotationRecord:
    scene_id: str
    filename: str
    member: str
    boxes: list[tuple[float, float, float, float]]


def _is_directory_member(member: str) -> bool:
    return member.endswith("/")


def _has_folder(member: str, folder_name: str) -> bool:
    return folder_name.lower() in {part.lower() for part in Path(member).parts}


def _index_source_members(source_zips: list[Path]) -> tuple[dict[str, tuple[Path, str]], dict[str, tuple[Path, str]], list[dict]]:
    rgb_index: dict[str, tuple[Path, str]] = {}
    lidar_index: dict[str, tuple[Path, str]] = {}
    duplicate_rows: list[dict] = []

    for zip_path in source_zips:
        with ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if _is_directory_member(member):
                    continue

                suffix = Path(member).suffix.lower()
                scene_id = Path(member).stem
                if suffix in RGB_EXTENSIONS and _has_folder(member, "RGB"):
                    if scene_id in rgb_index:
                        duplicate_rows.append(
                            {
                                "kind": "rgb",
                                "scene_id": scene_id,
                                "existing_zip": str(rgb_index[scene_id][0]),
                                "existing_member": rgb_index[scene_id][1],
                                "ignored_zip": str(zip_path),
                                "ignored_member": member,
                            }
                        )
                        continue
                    rgb_index[scene_id] = (zip_path, member)
                elif suffix in LIDAR_EXTENSIONS and _has_folder(member, "LiDAR"):
                    if scene_id in lidar_index:
                        duplicate_rows.append(
                            {
                                "kind": "lidar",
                                "scene_id": scene_id,
                                "existing_zip": str(lidar_index[scene_id][0]),
                                "existing_member": lidar_index[scene_id][1],
                                "ignored_zip": str(zip_path),
                                "ignored_member": member,
                            }
                        )
                        continue
                    lidar_index[scene_id] = (zip_path, member)

    return rgb_index, lidar_index, duplicate_rows


def _load_annotations(annotation_zip: Path) -> tuple[dict[str, AnnotationRecord], list[dict]]:
    annotations: dict[str, AnnotationRecord] = {}
    duplicate_rows: list[dict] = []

    with ZipFile(annotation_zip) as archive:
        for member in archive.namelist():
            if _is_directory_member(member) or not member.lower().endswith(".xml"):
                continue
            if "__macosx" in member.lower():
                continue

            root = ET.fromstring(archive.read(member))
            filename = (root.findtext("filename") or "").strip()
            if not filename:
                filename = f"{Path(member).stem}.tif"
            scene_id = Path(filename).stem

            boxes: list[tuple[float, float, float, float]] = []
            for object_node in root.findall("object"):
                bbox_node = object_node.find("bndbox")
                if bbox_node is None:
                    continue
                try:
                    xmin = float(bbox_node.findtext("xmin", "nan"))
                    ymin = float(bbox_node.findtext("ymin", "nan"))
                    xmax = float(bbox_node.findtext("xmax", "nan"))
                    ymax = float(bbox_node.findtext("ymax", "nan"))
                except ValueError:
                    continue
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append((xmin, ymin, xmax, ymax))

            record = AnnotationRecord(scene_id=scene_id, filename=filename, member=member, boxes=boxes)
            if scene_id in annotations:
                duplicate_rows.append(
                    {
                        "kind": "annotation",
                        "scene_id": scene_id,
                        "existing_member": annotations[scene_id].member,
                        "ignored_member": member,
                    }
                )
                continue
            annotations[scene_id] = record

    return annotations, duplicate_rows


def _extract_member(zip_path: Path, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        with archive.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def _bbox_center_to_world(transform, xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float]:
    center_col = 0.5 * (xmin + xmax)
    center_row = 0.5 * (ymin + ymax)
    x = transform.c + (center_col + 0.5) * transform.a + (center_row + 0.5) * transform.b
    y = transform.f + (center_col + 0.5) * transform.d + (center_row + 0.5) * transform.e
    return float(x), float(y)


def _is_projected_raster(dataset) -> bool:
    return dataset.crs is not None and not dataset.transform.is_identity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the official NeonTreeEvaluation ZIP files into LiDARAgroSat raw plot folders."
    )
    parser.add_argument(
        "--source-zips",
        nargs="+",
        required=True,
        help="One or more official source ZIPs such as training.zip and evaluation.zip.",
    )
    parser.add_argument("--annotations-zip", required=True, help="Path to annotations.zip.")
    parser.add_argument("--output-dir", default="data/neon/raw", help="Destination raw dataset directory.")
    parser.add_argument("--summary-dir", default="data/neon", help="Directory for import summary CSV/JSON files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already-imported scene directories.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    summary_dir = Path(args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_zips = [Path(path).resolve() for path in args.source_zips]
    annotations_zip = Path(args.annotations_zip).resolve()
    scene_metadata_dir = summary_dir / "scene_members"
    scene_metadata_dir.mkdir(parents=True, exist_ok=True)

    rgb_index, lidar_index, source_duplicates = _index_source_members(source_zips)
    annotations, annotation_duplicates = _load_annotations(annotations_zip)

    imported_rows: list[dict] = []
    skipped_rows: list[dict] = []
    for scene_id in sorted(annotations):
        annotation = annotations[scene_id]
        rgb_entry = rgb_index.get(scene_id)
        lidar_entry = lidar_index.get(scene_id)

        if rgb_entry is None:
            skipped_rows.append({"scene_id": scene_id, "reason": "missing_rgb", "annotation_member": annotation.member})
            continue
        if lidar_entry is None:
            skipped_rows.append({"scene_id": scene_id, "reason": "missing_lidar", "annotation_member": annotation.member})
            continue
        if not annotation.boxes:
            skipped_rows.append({"scene_id": scene_id, "reason": "empty_annotation", "annotation_member": annotation.member})
            continue

        scene_dir = output_dir / scene_id
        if scene_dir.exists():
            if not args.overwrite:
                skipped_rows.append({"scene_id": scene_id, "reason": "already_exists", "annotation_member": annotation.member})
                continue
            shutil.rmtree(scene_dir)
        scene_dir.mkdir(parents=True, exist_ok=True)

        rgb_zip_path, rgb_member = rgb_entry
        lidar_zip_path, lidar_member = lidar_entry
        rgb_path = scene_dir / f"rgb{Path(rgb_member).suffix.lower()}"
        lidar_path = scene_dir / f"lidar{Path(lidar_member).suffix.lower()}"
        _extract_member(rgb_zip_path, rgb_member, rgb_path)
        _extract_member(lidar_zip_path, lidar_member, lidar_path)

        try:
            with rasterio.open(rgb_path) as dataset:
                if not _is_projected_raster(dataset):
                    raise ValueError("unprojected_rgb")
                reference_rows = []
                for index, (xmin, ymin, xmax, ymax) in enumerate(annotation.boxes):
                    x, y = _bbox_center_to_world(dataset.transform, xmin, ymin, xmax, ymax)
                    reference_rows.append(
                        {
                            "tree_id": f"tree_{index:05d}",
                            "x": x,
                            "y": y,
                            "species": "tree",
                            "height": pd.NA,
                            "bbox_xmin": xmin,
                            "bbox_ymin": ymin,
                            "bbox_xmax": xmax,
                            "bbox_ymax": ymax,
                        }
                    )
                pd.DataFrame.from_records(reference_rows).to_csv(scene_dir / "trees_reference.csv", index=False)
        except Exception as exc:
            shutil.rmtree(scene_dir, ignore_errors=True)
            skipped_rows.append(
                {
                    "scene_id": scene_id,
                    "reason": str(exc),
                    "annotation_member": annotation.member,
                    "rgb_member": rgb_member,
                    "lidar_member": lidar_member,
                }
            )
            continue

        with (scene_metadata_dir / f"{scene_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "scene_id": scene_id,
                    "annotation_member": annotation.member,
                    "annotation_filename": annotation.filename,
                    "rgb_zip": str(rgb_zip_path),
                    "rgb_member": rgb_member,
                    "lidar_zip": str(lidar_zip_path),
                    "lidar_member": lidar_member,
                    "n_boxes": len(annotation.boxes),
                },
                handle,
                indent=2,
            )

        imported_rows.append(
            {
                "scene_id": scene_id,
                "annotation_member": annotation.member,
                "rgb_zip": str(rgb_zip_path),
                "rgb_member": rgb_member,
                "lidar_zip": str(lidar_zip_path),
                "lidar_member": lidar_member,
                "n_boxes": len(annotation.boxes),
            }
        )

    imported_table = pd.DataFrame.from_records(imported_rows).sort_values("scene_id") if imported_rows else pd.DataFrame()
    skipped_table = pd.DataFrame.from_records(skipped_rows).sort_values(["reason", "scene_id"]) if skipped_rows else pd.DataFrame()
    duplicate_table = pd.DataFrame.from_records(source_duplicates + annotation_duplicates)

    imported_table.to_csv(summary_dir / "imported_scenes.csv", index=False)
    skipped_table.to_csv(summary_dir / "skipped_scenes.csv", index=False)
    duplicate_table.to_csv(summary_dir / "duplicate_members.csv", index=False)

    summary_payload = {
        "source_zips": [str(path) for path in source_zips],
        "annotations_zip": str(annotations_zip),
        "output_dir": str(output_dir),
        "annotated_scene_count": len(annotations),
        "rgb_scene_count": len(rgb_index),
        "lidar_scene_count": len(lidar_index),
        "imported_scene_count": len(imported_rows),
        "skipped_scene_count": len(skipped_rows),
        "duplicate_count": len(source_duplicates) + len(annotation_duplicates),
    }
    with (summary_dir / "import_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
