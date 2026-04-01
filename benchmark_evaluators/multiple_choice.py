from .base import BenchmarkEvaluator, normalize_answer_text


class MultipleChoiceEvaluator(BenchmarkEvaluator):
    name = "multiple_choice"
    task_type = "multiple_choice"

    def match_choice(self, prediction, sample):
        pred = normalize_answer_text(prediction)
        choices = sample.get("choices") or []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for idx, choice in enumerate(choices):
            choice_text = normalize_answer_text(choice)
            if pred == choice_text:
                return idx
            prefix = f"{letters[idx].lower()}."
            if pred.startswith(prefix) and pred[len(prefix):].strip() == choice_text:
                return idx
            if pred == letters[idx].lower():
                return idx
        return None

    def score(self, prediction, sample):
        match_idx = self.match_choice(prediction, sample)
        correct_idx = sample.get("correct_idx")
        return {
            "correct": match_idx is not None and correct_idx is not None and int(match_idx) == int(correct_idx),
            "prediction": normalize_answer_text(prediction),
            "target": normalize_answer_text(sample.get("target_text", "")),
            "pred_idx": match_idx,
            "correct_idx": correct_idx,
        }
