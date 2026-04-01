from .base import BenchmarkEvaluator, normalize_answer_text, strip_punctuation


class OCRBenchEvaluator(BenchmarkEvaluator):
    name = "ocrbench"
    task_type = "generative"

    def score(self, prediction, sample):
        pred = strip_punctuation(normalize_answer_text(prediction))
        target = strip_punctuation(normalize_answer_text(sample.get("target_text", "")))
        return {
            "correct": pred == target,
            "prediction": pred,
            "target": target,
        }
