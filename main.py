import argparse

import tensorflow as tf

import verres as V


def get_args():
    parser = argparse.ArgumentParser(prog="Verres")
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--execution_type", "-e", type=str, default="_unset")
    parser.add_argument("--model_weights", "-w", type=str, default="_unset")
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


def main(config_path: str = None,
         execution_type: str = None,
         model_weights: str = None):

    if config_path is None:
        args = get_args()
        config_path = args.config
        execution_type = args.execution_type
        model_weights = args.model_weights

    cfg = V.Config(config_path)

    if execution_type in {"_unset", None}:
        execution_type = cfg.context.execution_type

    if model_weights not in {"_unset", None}:
        cfg.model.weights = model_weights

    if cfg.context.debug:
        tf.config.run_functions_eagerly(True)

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
