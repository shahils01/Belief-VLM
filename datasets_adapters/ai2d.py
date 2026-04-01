from .base import DatasetAdapter, build_mc_prompt, ensure_local_media_path, get_first, letter_to_index, normalize_choices, normalize_text


class AI2DAdapter(DatasetAdapter):
    name = "ai2d"
    media_type = "image"
    default_hf_repo = "lmms-lab/ai2d"
    default_hf_config = "default"
    default_train_split = "test"
    default_eval_split = "test"
    default_evaluator = "multiple_choice"

    def format_sample(self, raw_item, split, args):
        question = normalize_text(get_first(raw_item, ["question", "prompt", "query"]))
        choices = normalize_choices(get_first(raw_item, ["options", "choices", "answers"]))
        if not choices:
            return None
        correct_idx = get_first(raw_item, ["correct_idx", "answer_idx", "label_idx", "answer"])
        if isinstance(correct_idx, str):
            maybe = letter_to_index(correct_idx)
            correct_idx = maybe if maybe is not None else None
        elif correct_idx not in (None, ""):
            correct_idx = int(correct_idx)
        target_text = ""
        if correct_idx is not None and 0 <= correct_idx < len(choices):
            target_text = normalize_text(choices[correct_idx])
        media = ensure_local_media_path(get_first(raw_item, ["image", "image_path"]), args.media_root or args.video_root)
        return {
            "id": str(get_first(raw_item, ["id", "question_id", "sample_id"], default="")),
            "task_name": normalize_text(get_first(raw_item, ["task", "task_name", "subset"], default=self.name)),
            "media_type": "image",
            "media": media,
            "prompt": build_mc_prompt(question, choices),
            "target_text": target_text,
            "choices": choices,
            "correct_idx": correct_idx,
            "metadata": dict(raw_item),
        }
