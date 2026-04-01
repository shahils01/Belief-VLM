from .base import DatasetAdapter, build_mc_prompt, ensure_local_media_path, get_first, normalize_choices, normalize_text


class LocalManifestAdapter(DatasetAdapter):
    name = "local_manifest"
    supports_streaming = False
    default_evaluator = "generic_generation"

    def format_sample(self, raw_item, split, args):
        media_type = args.dataset_media_type or get_first(raw_item, ["media_type", "type"], default="image")
        media = get_first(
            raw_item,
            [args.media_column, args.image_column, args.video_column, "media", "image", "image_path", "video", "video_path"],
        )
        media = ensure_local_media_path(media, args.media_root or args.video_root)
        question = normalize_text(get_first(raw_item, [args.question_column, "question", "prompt", "instruction", "query"]))
        target_text = normalize_text(get_first(raw_item, [args.answer_column, "answer", "target_text", "response", "output"]))
        choices = normalize_choices(get_first(raw_item, [args.options_column, "choices", "options"]))
        correct_idx = get_first(raw_item, ["correct_idx", "answer_idx", "label_idx"])
        if correct_idx not in (None, ""):
            correct_idx = int(correct_idx)
        if choices:
            prompt = build_mc_prompt(question, choices)
            if target_text == "" and correct_idx is not None and 0 <= correct_idx < len(choices):
                target_text = normalize_text(choices[correct_idx])
        else:
            prompt = question
        return {
            "id": str(get_first(raw_item, [args.id_column, "id", "sample_id", "uid"], default="")),
            "task_name": normalize_text(get_first(raw_item, ["task_name", "task", "subset"], default=self.name)),
            "media_type": media_type,
            "media": media,
            "prompt": prompt,
            "target_text": target_text,
            "choices": choices,
            "correct_idx": correct_idx,
            "metadata": dict(raw_item),
        }
