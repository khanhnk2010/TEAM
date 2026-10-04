import os
import csv
import shutil
import argparse
import subprocess

from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


DATASET_CONFIG = {
    "UCF101": {
        "split_dir": "splits/ucf",
        "output_dir": "ucf101_FSAR",
        "source_layout": "class_subdir",
        "video_exts": [".avi", ".mp4"],
    },
    "HMDB51": {
        "split_dir": "splits/hmdb",
        "output_dir": "hmdb51_FSAR",
        "source_layout": "class_subdir",
        "video_exts": [".avi", ".mp4"],
    },
    "Kinetics": {
        "split_dir": "splits/kinetics",
        "output_dir": "kinetics_FSAR",
        "source_layout": "class_subdir",
        "video_exts": [".avi", ".mp4"],
    },
    "SSv2-Small": {
        "split_dir": "splits/ssv2_small",
        "output_dir": "ssv2_small_FSAR",
        "source_layout": "flat",
        "video_exts": [".webm", ".mp4"],
    },
}


@dataclass(frozen=True)
class VideoJob:
    split: str
    class_name: str
    video_id: str
    source_path: Path
    target_dir: Path


@dataclass
class JobResult:
    split: str
    class_name: str
    video_id: str
    source_path: str
    target_dir: str
    status: str
    num_frames: int
    message: str = ""


def normalize_extensions(exts):
    result = []

    for ext in exts:
        ext = ext.lower().strip()

        if not ext.startswith("."):
            ext = "." + ext

        if ext not in result:
            result.append(ext)

    return result


def parse_split_line(line):
    line = line.strip()

    if not line:
        return None

    line = line.replace("\\", "/")
    class_name, video_name = line.rsplit("/", 1)

    # Also supports lines such as: class/video01.avi 1
    video_name = video_name.split()[0]
    video_id = Path(video_name).stem

    return class_name.strip(), video_id


def candidate_source_paths(
    source_dir,
    class_name,
    video_id,
    source_layout,
    video_exts,
):
    source_dir = Path(source_dir)

    if source_layout == "flat":
        folders = [source_dir]
    else:
        folders = [
            source_dir / class_name,
            source_dir / class_name.replace(" ", "_"),
        ]

    paths = []

    for folder in folders:
        for ext in video_exts:
            paths.append(folder / f"{video_id}{ext}")

    return paths


def find_source_video(
    source_dir,
    class_name,
    video_id,
    source_layout,
    video_exts,
):
    paths = candidate_source_paths(
        source_dir,
        class_name,
        video_id,
        source_layout,
        video_exts,
    )

    for path in paths:
        if path.exists():
            return path

    return None


def count_frames(target_dir):
    return len(list(Path(target_dir).glob("*.jpg")))


def build_jobs(
    source_dir,
    split_dir,
    output_dir,
    source_layout,
    video_exts,
    splits=("train", "val", "test"),
):
    jobs = []
    missing = []

    source_dir = Path(source_dir)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)

    for split in splits:
        split_file = split_dir / f"{split}list.txt"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        lines = split_file.read_text(encoding="utf-8").splitlines()

        for line in lines:
            parsed = parse_split_line(line)

            if parsed is None:
                continue

            class_name, video_id = parsed

            source_path = find_source_video(
                source_dir,
                class_name,
                video_id,
                source_layout,
                video_exts,
            )

            target_dir = output_dir / split / class_name / video_id

            if source_path is None:
                missing.append(
                    JobResult(
                        split=split,
                        class_name=class_name,
                        video_id=video_id,
                        source_path="",
                        target_dir=str(target_dir),
                        status="missing",
                        num_frames=0,
                        message="Source video not found",
                    )
                )
                continue

            jobs.append(
                VideoJob(
                    split=split,
                    class_name=class_name,
                    video_id=video_id,
                    source_path=source_path,
                    target_dir=target_dir,
                )
            )

    return jobs, missing


def make_scale_filter(short_side):
    if short_side <= 0:
        return None

    return (
        "scale="
        f"'if(gt(iw,ih),-2,{short_side})':"
        f"'if(gt(iw,ih),{short_side},-2)'"
    )


def extract_one(
    job,
    short_side=0,
    overwrite=False,
    dry_run=False,
):
    existing_frames = count_frames(job.target_dir)

    if existing_frames > 0 and not overwrite:
        return JobResult(
            split=job.split,
            class_name=job.class_name,
            video_id=job.video_id,
            source_path=str(job.source_path),
            target_dir=str(job.target_dir),
            status="skipped",
            num_frames=existing_frames,
        )

    if dry_run:
        return JobResult(
            split=job.split,
            class_name=job.class_name,
            video_id=job.video_id,
            source_path=str(job.source_path),
            target_dir=str(job.target_dir),
            status="dry_run",
            num_frames=0,
        )

    if overwrite and job.target_dir.exists():
        shutil.rmtree(job.target_dir)

    job.target_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(job.source_path),
        "-threads",
        "1",
    ]

    scale_filter = make_scale_filter(short_side)

    if scale_filter:
        command += ["-vf", scale_filter]

    command += [
        "-q:v",
        "2",
        str(job.target_dir / "%06d.jpg"),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    frame_count = count_frames(job.target_dir)

    if result.returncode != 0 or frame_count == 0:
        return JobResult(
            split=job.split,
            class_name=job.class_name,
            video_id=job.video_id,
            source_path=str(job.source_path),
            target_dir=str(job.target_dir),
            status="failed",
            num_frames=frame_count,
            message=result.stderr.strip(),
        )

    return JobResult(
        split=job.split,
        class_name=job.class_name,
        video_id=job.video_id,
        source_path=str(job.source_path),
        target_dir=str(job.target_dir),
        status="ok",
        num_frames=frame_count,
    )


def write_manifest(results, manifest_path):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "split",
        "class_name",
        "video_id",
        "source_path",
        "target_dir",
        "status",
        "num_frames",
        "message",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {field: getattr(result, field) for field in fields}
            )


def print_summary(results):
    counts = {}

    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    print("\nSummary:")
    for status, count in counts.items():
        print(f"  {status}: {count}")


def run_preprocessing(
    jobs,
    missing,
    output_dir,
    num_workers=4,
    short_side=0,
    overwrite=False,
    dry_run=False,
):
    results = list(missing)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                extract_one,
                job,
                short_side,
                overwrite,
                dry_run,
            )
            for job in jobs
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)

            if i == 1 or i % 25 == 0 or i == len(jobs):
                print(
                    f"[{i}/{len(jobs)}] "
                    f"{result.status}: {result.video_id}"
                )

    manifest_path = Path(output_dir) / "preprocess_manifest.csv"

    write_manifest(results, manifest_path)
    print_summary(results)

    print(f"\nManifest: {manifest_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="TEAM-compatible video preprocessing"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_CONFIG.keys(),
    )

    parser.add_argument(
        "--source_dir",
        required=True,
        help="Directory containing the raw videos",
    )

    parser.add_argument(
        "--split_dir",
        default=None,
        help="Override the dataset split directory",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override the output frame directory",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--short_side",
        type=int,
        default=0,
        help="0 = keep original resolution",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not installed")

    config = DATASET_CONFIG[args.dataset]

    split_dir = args.split_dir or config["split_dir"]
    output_dir = args.output_dir or config["output_dir"]

    video_exts = normalize_extensions(config["video_exts"])
    source_layout = config["source_layout"]

    print("Dataset:", args.dataset)
    print("Source:", args.source_dir)
    print("Splits:", split_dir)
    print("Output:", output_dir)

    jobs, missing = build_jobs(
        source_dir=args.source_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        source_layout=source_layout,
        video_exts=video_exts,
        splits=args.splits,
    )

    print(f"\nJobs found: {len(jobs)}")
    print(f"Missing videos: {len(missing)}\n")

    run_preprocessing(
        jobs=jobs,
        missing=missing,
        output_dir=output_dir,
        num_workers=args.num_workers,
        short_side=args.short_side,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
