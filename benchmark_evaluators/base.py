import re
import string
from abc import ABC


def normalize_answer_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_punctuation(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


class BenchmarkEvaluator(ABC):
    name = "base"
    task_type = "generative"

    def prepare_prompt(self, sample):
        return sample["prompt"]

    def postprocess_prediction(self, text):
        return str(text).strip()

    def normalize(self, text):
        return normalize_answer_text(text)

    def score(self, prediction, sample):
        pred = self.normalize(self.postprocess_prediction(prediction))
        target = self.normalize(sample.get("target_text", ""))
        return {
            "correct": pred == target,
            "prediction": pred,
            "target": target,
        }
