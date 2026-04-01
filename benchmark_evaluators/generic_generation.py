from .base import BenchmarkEvaluator


class GenericGenerationEvaluator(BenchmarkEvaluator):
    name = "generic_generation"
    task_type = "generative"
