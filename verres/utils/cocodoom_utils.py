def apply_image_filters(meta_iterator, config):
    if config.run_number is not None:
        criterion = "run{}".format(config.run_number)
        meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
    if config.level_number is not None:
        criterion = "map{}".format(config.level_number)
        meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
    return meta_iterator


