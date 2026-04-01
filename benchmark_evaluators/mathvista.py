import re

from .base import BenchmarkEvaluator, normalize_answer_text, strip_punctuation


def _parse_number(text):
    if text is None:
        return None
    cleaned = strip_punctuation(normalize_answer_text(text)).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


class MathVistaEvaluator(BenchmarkEvaluator):
    name = "mathvista"
    task_type = "generative"

    def score(self, prediction, sample):
        pred_text = normalize_answer_text(prediction)
        target_text = normalize_answer_text(sample.get("target_text", ""))
        pred_num = _parse_number(pred_text)
        target_num = _parse_number(target_text)
        numeric_match = pred_num is not None and target_num is not None and abs(pred_num - target_num) < 1e-5
        return {
            "correct": numeric_match or pred_text == target_text,
            "prediction": pred_text,
            "target": target_text,
        }
