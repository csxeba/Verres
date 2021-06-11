import argparse
import os
from typing import Dict, Any

import tensorflow as tf

import verres as V


def get_args():
    parser = argparse.ArgumentParser(prog="Verres")
    parser.add_argument("--config", "-c", type=str, default="_unset")
    parser.add_argument("--continue_training", type=str, default="_unset")
    parser.add_argument("--execution_type", "-e", type=str, default="_unset")
    parser.add_argument("--model_weights", "-w", type=str, default="_unset")
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--config_updates", "-u", nargs="+", default={})
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
        setattr(field, field_name_list[-1], value)
    print(f" [Verres] - Set config.{field_path} to {value}")


def main(config_path: str = None,
         continue_train: str = None,
         execution_type: str = None,
         model_weights: str = None,
         debug: bool = False,
         config_updates: Dict[str, Any] = None):

    on_cluster = "COLAB_GPU" in os.environ

    if not on_cluster:
        args = get_args()
        config_path = args.config
        continue_train = args.continue_training
        execution_type = args.execution_type
        model_weights = args.model_weights
        debug = args.debug
        config_updates = dict(line.split("=") for line in args.config_updates)

    if continue_train not in {None, "_unset"}:
        config_path = os.path.join(continue_train, "config.yml")
        config_updates["model.weights"] = os.path.join(continue_train, "checkpoints", "latest.h5")
        config_updates["training.initial_epoch"] = V.utils.logging_utils.extract_last_epoch(continue_train)

    cfg = V.Config(config_path)

    if config_updates:
        for field_name, value in config_updates.items():
            update_config(cfg, field_name, value)

    cfg.context.debug = cfg.context.debug or debug

    if execution_type in {"_unset", None}:
        execution_type = cfg.context.execution_type

    if model_weights not in {"_unset", None}:
        cfg.model.weights = model_weights

    tf.keras.backend.set_floatx(cfg.context.float_precision)
    if cfg.context.verbose:
        print(" [Verres] - Float precision set to", cfg.context.float_precision)

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
