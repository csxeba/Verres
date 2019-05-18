from collections import defaultdict


def ignore_categories(data, frequency_threshold=0.01, ignored_ids=None, ignored_names=()):
    category_index = {cat["id"]: cat for cat in data["categories"]}
    class_freqs = defaultdict(int)
    ignored_names = set(ignored_names)

    N = 0
    dropped = 0
    if ignored_ids is None:
        ignored_ids = set()

    for anno in data["annotations"]:
        class_freqs[anno["category_id"]] += 1
        N += 1

    for ID, freq in class_freqs.items():
        percent = freq / N
        category = category_index[ID]
        if percent < frequency_threshold or ID in ignored_ids or category["name"] in ignored_names:
            ignored_ids.add(ID)
            category_index[ID]["ignore"] = True
            dropped += 1
        else:
            category_index[ID]["ignore"] = False

    print(f"Dropped {dropped} classes due to low frequency.")
    return ignored_ids
