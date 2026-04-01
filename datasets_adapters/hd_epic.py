from data_loading import _resolve_hd_epic_clip_window, _resolve_hd_epic_video_path

from .base import DatasetAdapter, build_mc_prompt, get_first, normalize_choices, normalize_text


class HDEpicAdapter(DatasetAdapter):
    name = "hd_epic"
    media_type = "video"
    supports_streaming = False
    default_evaluator = "multiple_choice"

    def format_sample(self, raw_item, split, args):
        question = normalize_text(get_first(raw_item, [args.question_column, "question", "prompt"]))
        choices = normalize_choices(get_first(raw_item, [args.options_column, "choices", "options"]))
        answer = normalize_text(get_first(raw_item, [args.answer_column, "answer"]))
        correct_idx = get_first(raw_item, ["correct_idx", "answer_idx", "label_idx"])
        if correct_idx not in (None, ""):
            correct_idx = int(correct_idx)
            if choices and 0 <= correct_idx < len(choices):
                answer = normalize_text(choices[correct_idx])
        video_path = _resolve_hd_epic_video_path(args, raw_item)
        start_time_sec, end_time_sec = _resolve_hd_epic_clip_window(raw_item)
        metadata = dict(raw_item)
        metadata["start_time_sec"] = start_time_sec
        metadata["end_time_sec"] = end_time_sec
        return {
            "id": str(get_first(raw_item, [args.id_column, "id", "sample_id"], default="")),
            "task_name": normalize_text(get_first(raw_item, ["task_name", "task"], default=self.name)),
            "media_type": "video",
            "media": video_path,
            "prompt": build_mc_prompt(question, choices) if choices else question,
            "target_text": answer,
            "choices": choices,
            "correct_idx": correct_idx,
            "metadata": metadata,
        }
