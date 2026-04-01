from .base import DatasetAdapter, ensure_local_media_path, get_first, normalize_text


class OCRBenchAdapter(DatasetAdapter):
    name = "ocrbench"
    media_type = "image"
    default_hf_repo = "LIME-DATA/ocrbench"
    default_hf_config = "default"
    default_train_split = "train"
    default_eval_split = "train"
    default_evaluator = "ocrbench"

    def format_sample(self, raw_item, split, args):
        question = normalize_text(get_first(raw_item, ["question", "prompt", "instruction", "query"]))
        answer = get_first(raw_item, ["answer", "answers", "target", "response"])
        if isinstance(answer, (list, tuple)):
            answer = answer[0] if answer else ""
        answer = normalize_text(answer)
        media = ensure_local_media_path(get_first(raw_item, ["image", "image_path"]), args.media_root or args.video_root)
        return {
            "id": str(get_first(raw_item, ["id", "sample_id", "uid"], default="")),
            "task_name": normalize_text(get_first(raw_item, ["dataset", "task_name", "subset"], default=self.name)),
            "media_type": "image",
            "media": media,
            "prompt": question,
            "target_text": answer,
            "choices": None,
            "correct_idx": None,
            "metadata": dict(raw_item),
        }
