from .inference import InferenceExecutor
from .training import TrainingExecutor
from .evaluation import EvaluationExecutor


class ExecutionType:

    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
