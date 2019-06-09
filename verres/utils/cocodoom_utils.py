from verres.data import cocodoom


def filter_by_path(meta_iterator, config: cocodoom.COCODoomStreamConfig):
    if config.run_number is not None:
        criterion = "run{}".format(config.run_number)
        meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
    if config.level_number is not None:
        criterion = "map{:0>2}".format(config.level_number)
        meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
    return meta_iterator


def filter_by_objects(meta_iterator,
                      config: cocodoom.COCODoomStreamConfig,
                      loader: cocodoom.COCODoomLoader):
    if config.min_no_visible_objects > 1:
        meta_iterator = (meta for meta in meta_iterator if
                         len(loader.index[meta["id"]]) >= config.min_no_visible_objects)
    return meta_iterator


def apply_filters(meta_iterator,
                  config: cocodoom.COCODoomStreamConfig,
                  loader: cocodoom.COCODoomLoader):

    return filter_by_objects(filter_by_path(meta_iterator, config), config, loader)
