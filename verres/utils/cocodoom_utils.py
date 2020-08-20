import os
import json
from verres.data import cocodoom


def fetch(to_root: str = "/data/Datasets", full=False):
    from urllib import request
    import tarfile

    URL = "http://www.robots.ox.ac.uk/~vgg/share/cocodoom-v1.0.tar.gz"
    if full:
        URL = "http://www.robots.ox.ac.uk/~vgg/share/cocodoom-full-v1.0.tar.gz"

    os.makedirs(to_root, exist_ok=True)
    filename = os.path.split(URL)[-1]
    output_path = os.path.join(to_root, filename)
    request.urlretrieve(URL, output_path)

    cwd = os.getcwd()

    os.chdir(to_root)
    tarfile.open(output_path).extractall()
    os.chdir(cwd)


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


def generate_enemy_dataset(root="/data/Datasets/cocodoom"):

    ENEMY_TYPES = [
        "POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD", "BRUISER",
        "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN", "WOLFSS"
    ]

    def convert(source, target):
        data = json.load(open(source))
        enemy_ids = set(cat["id"] for cat in data["categories"] if cat["name"] in ENEMY_TYPES)
        data["annotations"] = [anno for anno in data["annotations"] if anno["category_id"] in enemy_ids]
        with open(target, "w") as handle:
            json.dump(data, handle)

    files = ["map-train.json", "map-full-train.json",
             "map-val.json", "map-full-val.json",
             "map-test.json", "map-full-test.json",
             "run-train.json", "run-full-train.json",
             "run-val.json", "run-full-val.json",
             "run-test.json", "run-full-test.json"]

    for file in files:
        file_path = os.path.join(root, file)
        if not os.path.exists(file_path):
            print(" [Verres] - Non-existent annotation file:", file_path)
            continue
        target_file = "enemy-" + file
        target_path = os.path.join(root, target_file)
        if os.path.exists(target_path):
            print(" [Verres] - Target file already exists:", target_path)
            continue
        print(f" [Verres] {file} -> {target_file}")
        convert(file_path, target_path)
