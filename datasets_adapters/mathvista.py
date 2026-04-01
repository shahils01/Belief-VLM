from .base import DatasetAdapter, build_mc_prompt, ensure_local_media_path, get_first, letter_to_index, normalize_choices, normalize_text


class MathVistaAdapter(DatasetAdapter):
    name = "mathvista"
    media_type = "image"
    default_hf_repo = "AI4Math/MathVista"
    default_hf_config = "default"
    default_train_split = "testmini"
    default_eval_split = "testmini"
    default_evaluator = "mathvista"

    def format_sample(self, raw_item, split, args):
        question = normalize_text(get_first(raw_item, ["question", "query", "prompt"]))
        choices = normalize_choices(get_first(raw_item, ["choices", "options"]))
        answer = normalize_text(get_first(raw_item, ["answer", "unit_answer", "decoded_answer"]))
        correct_idx = get_first(raw_item, ["correct_idx", "answer_idx", "label_idx"])
        if correct_idx not in (None, ""):
            correct_idx = int(correct_idx)
        elif choices and answer:
            maybe_idx = letter_to_index(answer)
            if maybe_idx is not None and 0 <= maybe_idx < len(choices):
                correct_idx = maybe_idx
                answer = normalize_text(choices[maybe_idx])
        prompt = build_mc_prompt(question, choices) if choices else question
        media = ensure_local_media_path(get_first(raw_item, ["decoded_image", "image", "image_path"]), args.media_root or args.video_root)
        return {
            "id": str(get_first(raw_item, ["pid", "id", "sample_id"], default="")),
            "task_name": normalize_text(get_first(raw_item, ["question_type", "task_name", "subset"], default=self.name)),
            "media_type": "image",
            "media": media,
            "prompt": prompt,
            "target_text": answer,
            "choices": choices,
            "correct_idx": correct_idx,
            "metadata": dict(raw_item),
        }
