import argparse
import json
import os

from data_loading import (
    _load_records,
    _resolve_hd_epic_clip_window,
    _resolve_hd_epic_video_path,
    decode_mp4_frames,
)
from NextVQA_data_loading import (
    _load_records_from_path,
    _resolve_clip_window,
    _resolve_split_annotation_path,
    _resolve_video_path,
)


def parse_args():
    p = argparse.ArgumentParser(description="Scan dataset annotations and report video decode failures.")
    p.add_argument("--dataset_source", type=str, default="nextvqa", choices=["hd_epic", "nextvqa"])
    p.add_argument("--annotation_path", type=str, required=True)
    p.add_argument("--video_root", type=str, required=True)
    p.add_argument("--video_extension", type=str, default="mp4")
    p.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    p.add_argument("--video_frames", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--out_json", type=str, default="decode_failures.json")
    return p.parse_args()


def _iter_nextvqa_records(args):
    split_path = _resolve_split_annotation_path(args.annotation_path, args.split)
    records = _load_records_from_path(split_path)
    for i, r in enumerate(records):
        if args.max_samples > 0 and i >= args.max_samples:
            break
        yield i, r


def _iter_hdepic_records(args):
    records = _load_records(args)
    for i, r in enumerate(records):
        if args.max_samples > 0 and i >= args.max_samples:
            break
        yield i, r


def main():
    args = parse_args()
    failures = []
    total = 0
    ok = 0

    if args.dataset_source == "nextvqa":
        record_iter = _iter_nextvqa_records(args)
        resolver = _resolve_video_path
        clip_window = _resolve_clip_window
    else:
        record_iter = _iter_hdepic_records(args)
        resolver = _resolve_hd_epic_video_path
        clip_window = _resolve_hd_epic_clip_window

    for idx, record in record_iter:
        total += 1
        try:
            video_path = resolver(args, record)
            start_sec, end_sec = clip_window(record)
            _ = decode_mp4_frames(
                video_path=video_path,
                num_frames=args.video_frames,
                start_time_sec=start_sec,
                end_time_sec=end_sec,
            )
            ok += 1
        except Exception as exc:
            sample_id = (
                record.get("id")
                or record.get("qid")
                or record.get("sample_id")
                or record.get("uid")
                or f"idx_{idx}"
            )
            video_id = record.get("video_id") or record.get("vid") or record.get("video")
            failures.append(
                {
                    "sample_id": str(sample_id),
                    "video_id": "" if video_id is None else str(video_id),
                    "error": str(exc),
                }
            )
            print(f"[FAIL] sample_id={sample_id} video_id={video_id} err={exc}")

        if total % 200 == 0:
            print(f"processed={total} ok={ok} failed={len(failures)}")

    summary = {
        "dataset_source": args.dataset_source,
        "split": args.split,
        "total": total,
        "ok": ok,
        "failed": len(failures),
        "failures": failures,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"scan complete: total={total} ok={ok} failed={len(failures)} out={os.path.abspath(args.out_json)}")


if __name__ == "__main__":
    main()
