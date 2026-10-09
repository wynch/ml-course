"""Solution B — exact overlap."""


def overlapping_ids(train_ids, test_ids):
    return sorted(set(map(str, train_ids)) & set(map(str, test_ids)))
