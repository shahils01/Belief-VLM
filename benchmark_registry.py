from benchmark_evaluators import (
    GenericGenerationEvaluator,
    MathVistaEvaluator,
    MultipleChoiceEvaluator,
    OCRBenchEvaluator,
)


_EVALUATORS = {
    "generic_generation": GenericGenerationEvaluator,
    "multiple_choice": MultipleChoiceEvaluator,
    "mathvista": MathVistaEvaluator,
    "ocrbench": OCRBenchEvaluator,
}


def list_evaluators():
    return sorted(_EVALUATORS)


def get_evaluator(name):
    key = name.lower()
    if key not in _EVALUATORS:
        raise KeyError(f"Unknown benchmark evaluator `{name}`. Available: {', '.join(list_evaluators())}")
    return _EVALUATORS[key]()
