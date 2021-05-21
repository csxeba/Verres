import argparse
from typing import Dict, Any

import verres as V


def get_args():
    parser = argparse.ArgumentParser(prog="Verres")
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--execution_type", "-e", type=str, default="_unset")
    parser.add_argument("--model_weights", "-w", type=str, default="_unset")
    parser.add_argument("--config_updates", "-u", nargs="+")
    return parser.parse_args()


def execute_training(cfg: V.Config):
    print(" [Verres] - Execution type: training")
    trainer = V.execution.TrainingExecutor.factory(cfg)
    trainer.execute()


def execute_inference(cfg: V.Config):
    print(" [Verres] - Execution type: inference")
    inference = V.execution.InferenceExecutor.factory(cfg)
    inference.execute()


def execute_evaluation(cfg: V.Config):
    print(" [Verres] - Execution type: evaluation")
    evaluators = V.execution.EvaluationExecutor.factory(cfg)
    for evaluator in evaluators:
        evaluator.execute()


def update_config(config: V.Config, field_path: str, value):
    field_name_list = field_path.split(".")
    field = config
    for field_name in field_name_list[:-1]:
        if isinstance(field, dict):
            field = field.get(field_name, None)
        else:
            field = getattr(field, field_name, None)
        if field is None:
            raise RuntimeError(f"No such config field: {field_path}")
    if isinstance(field, dict):
        field[field_name_list[-1]] = value
    else:
        setattr(field, field_path[-1], value)
    print(f" [Verres] - Set config.{field_path} to {value}")


def main(config_path: str = None,
         execution_type: str = None,
         model_weights: str = None,
         config_updates: Dict[str, Any] = None):

    if config_path is None:
        args = get_args()
        config_path = args.config
        execution_type = args.execution_type
        model_weights = args.model_weights
        config_updates = dict(line.split("=") for line in args.config_updates)

    cfg = V.Config(config_path)

    if config_updates:
        for field_name, value in config_updates.items():
            update_config(cfg, field_name, value)

    if execution_type in {"_unset", None}:
        execution_type = cfg.context.execution_type

    if model_weights not in {"_unset", None}:
        cfg.model.weights = model_weights

    if execution_type == V.execution.ExecutionType.TRAINING:
        execute_training(cfg)
    elif execution_type == V.execution.ExecutionType.INFERENCE:
        execute_inference(cfg)
    elif execution_type == V.execution.ExecutionType.EVALUATION:
        execute_evaluation(cfg)
    else:
        assert False


if __name__ == '__main__':
    main()
