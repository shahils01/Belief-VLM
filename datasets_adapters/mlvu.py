from .base import DatasetAdapter, build_mc_prompt, ensure_local_media_path, get_first, letter_to_index, normalize_choices, normalize_text


class MLVUAdapter(DatasetAdapter):
    name = "mlvu"
    media_type = "video"
    supports_streaming = False
    default_hf_repo = "sy1998/MLVU_Test"
    default_hf_config = None
    default_train_split = "test"
    default_eval_split = "test"
    default_evaluator = "multiple_choice"

    def format_sample(self, raw_item, split, args):
        question = normalize_text(get_first(raw_item, ["question", "prompt", "query"]))
        choices = normalize_choices(get_first(raw_item, ["candidates", "choices", "options"]))
        answer = normalize_text(get_first(raw_item, ["answer", "target"]))
        correct_idx = get_first(raw_item, ["correct_idx", "answer_idx", "label_idx"])
        if correct_idx not in (None, ""):
            correct_idx = int(correct_idx)
        elif choices and answer:
            maybe = letter_to_index(answer)
            if maybe is not None and 0 <= maybe < len(choices):
                correct_idx = maybe
                answer = normalize_text(choices[maybe])
            elif answer in choices:
                correct_idx = choices.index(answer)
        if choices and not answer and correct_idx is not None and 0 <= correct_idx < len(choices):
            answer = normalize_text(choices[correct_idx])
        metadata = dict(raw_item)
        media = ensure_local_media_path(get_first(raw_item, ["video", "video_path", "video_name", "video_id"]), args.video_root or args.media_root)
        return {
            "id": str(get_first(raw_item, ["question_id", "id", "sample_id"], default="")),
            "task_name": normalize_text(get_first(raw_item, ["task_type", "task_name", "subset"], default=self.name)),
            "media_type": "video",
            "media": media,
            "prompt": build_mc_prompt(question, choices) if choices else question,
            "target_text": answer,
            "choices": choices,
            "correct_idx": correct_idx,
            "metadata": metadata,
        }
