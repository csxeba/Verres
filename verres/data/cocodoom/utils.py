import os
import json
import pathlib
from collections import defaultdict

import numpy as np

from verres.data import cocodoom

ENEMY_TYPES = [
    "POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD", "BRUISER",
    "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN", "WOLFSS"
]


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


def filter_by_path(meta_iterator, config: cocodoom.StreamConfig):
    if config.run_number is not None:
        criterion = "run{}".format(config.run_number)
        meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
    if config.level_number is not None:
        criterion = "map{:0>2}".format(config.level_number)
        meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
    return meta_iterator


def filter_by_objects(meta_iterator,
                      config: cocodoom.StreamConfig,
                      loader: cocodoom.Loader):
    if config.min_no_visible_objects > 0:
        meta_iterator = (meta for meta in meta_iterator if
                         len(loader.index[meta["id"]]) >= config.min_no_visible_objects)
    return meta_iterator


def apply_filters(meta_iterator,
                  config: cocodoom.StreamConfig,
                  loader: cocodoom.Loader):

    return filter_by_objects(filter_by_path(meta_iterator, config), config, loader)


def generate_enemy_dataset(root="/data/Datasets/cocodoom"):

    print(" [Verres] - Generating enemy-only dataset")

    def convert(source, target):
        data = json.load(open(source))
        enemy_ids = set(cat["id"] for cat in data["categories"] if cat["name"] in ENEMY_TYPES)
        data["annotations"] = [anno for anno in data["annotations"] if anno["category_id"] in enemy_ids]
        with open(target, "w") as handle:
            json.dump(data, handle)

    files = ["-".join([t1, t2]) + ".json" for t2 in ["train", "val", "test"] for t1 in ["map", "run"]]

    with_full = [file.replace("-", "-full-") for file in files]
    files.extend(with_full)
    with_time = ["time-" + file for file in files]
    files.extend(with_time)

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


def deconstruct_path(file_name: str):
    # run_str, map_str, frame_str = file_name.split("/")
    parts = file_name.split("/")
    run_no = int(parts[0].split("run")[-1])
    map_no = int(parts[1].split("map")[-1])
    frame_no = int(parts[-1].split(".")[0])
    return run_no, map_no, frame_no


def reconstruct_path(run_no: int, map_no: int, frame_no: int):
    return os.path.join(f"{run_no:0>2}", f"{map_no:0>2}", "rgb", f"{frame_no:0>6}")


def generate_time_priorized_dataset(full_dataset_json_path: str,
                                    output_json_path: str,
                                    enemy_only: bool = True,
                                    overwrite: bool = False):

    if not overwrite and os.path.exists(output_json_path):
        print(" [Verres] - Dataset already exists:", output_json_path)
        return

    data = json.load(open(full_dataset_json_path))

    print(" [Verres] - Generating Time dataset...")

    for prev_meta, meta in zip(data["images"][:-1], data["images"][1:]):
        run_now, map_now, frame_now = deconstruct_path(meta["file_name"])
        run_then, map_then, frame_then = deconstruct_path(prev_meta["file_name"])
        prev_meta_id = prev_meta["id"]
        if map_now != map_then:
            prev_meta_id = None
        meta["prev_image_id"] = prev_meta_id

    pathlib.Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    print(f" [Verres] - Done! Writing to {output_json_path}")
    with open(output_json_path, "w") as handle:
        json.dump(data, handle)

    return data


def subsample_coco_dataset(input_dataset: str,
                           output_dataset: str,
                           subsampling_factor: int,
                           shuffle: bool = True):

    print(" [Verres] - Building indices for dataset subsampling")
    data = json.load(open(input_dataset))
    annotation_index = defaultdict(list)
    for anno in data["annotations"]:
        annotation_index[anno["image_id"]].append(anno)
    image_index = {meta["id"]: meta for meta in data["images"]}
    IDs = np.array(list(image_index.keys()))
    if shuffle:
        np.random.shuffle(IDs)

    print(" [Verres] - Generating new annotation set")
    new_annotations = []
    for i, ID in enumerate(IDs, start=1):
        if i % subsampling_factor == 0:
            continue
        new_annotations.extend(annotation_index[ID])
    data["annotations"] = new_annotations

    print(" [Verres] - Dumping subsampled dataset")
    with open(output_dataset, "w") as handle:
        json.dump(data, handle)
