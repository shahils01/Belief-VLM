from .base import DatasetAdapter, build_mc_prompt, ensure_local_media_path, get_first, letter_to_index, normalize_choices, normalize_text


class MMMUAdapter(DatasetAdapter):
    name = "mmmu"
    media_type = "image"
    default_hf_repo = "MMMU/MMMU"
    default_train_split = "dev"
    default_eval_split = "validation"
    default_evaluator = "multiple_choice"

    def _collect_images(self, raw_item, args):
        candidates = []
        if raw_item.get("image") is not None:
            candidates.append(raw_item["image"])
        for idx in range(1, 8):
            value = raw_item.get(f"image_{idx}")
            if value is not None:
                candidates.append(value)
        if not candidates and raw_item.get("images") is not None:
            maybe = raw_item["images"]
            if isinstance(maybe, (list, tuple)):
                candidates.extend(maybe)
        if not candidates:
            return None
        return [ensure_local_media_path(item, args.media_root or args.video_root) for item in candidates[:1]]

    def format_sample(self, raw_item, split, args):
        question = normalize_text(get_first(raw_item, ["question", "prompt", "query"]))
        choices = normalize_choices(get_first(raw_item, ["options", "choices"]))
        if not choices:
            option_keys = [key for key in sorted(raw_item) if key.lower().startswith("option")]
            choices = normalize_choices([raw_item[key] for key in option_keys]) if option_keys else None
        answer = normalize_text(get_first(raw_item, ["answer", "target"]))
        correct_idx = get_first(raw_item, ["correct_idx", "answer_idx", "label_idx"])
        if correct_idx not in (None, ""):
            correct_idx = int(correct_idx)
        elif answer:
            maybe = letter_to_index(answer)
            if maybe is not None and choices and 0 <= maybe < len(choices):
                correct_idx = maybe
                answer = normalize_text(choices[maybe])
            elif choices and answer in choices:
                correct_idx = choices.index(answer)
        if choices and not answer and correct_idx is not None and 0 <= correct_idx < len(choices):
            answer = normalize_text(choices[correct_idx])
        media = self._collect_images(raw_item, args)
        return {
            "id": str(get_first(raw_item, ["id", "sample_id", "uid", "question_id"], default="")),
            "task_name": normalize_text(get_first(raw_item, ["subject", "task_name", "subset"], default=self.name)),
            "media_type": "image",
            "media": media,
            "prompt": build_mc_prompt(question, choices) if choices else question,
            "target_text": answer,
            "choices": choices,
            "correct_idx": correct_idx,
            "metadata": dict(raw_item),
        }
