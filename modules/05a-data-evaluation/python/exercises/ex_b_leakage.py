"""Exercise B — detect exact duplicate leakage."""


def overlapping_ids(train_ids, test_ids):
    """Return sorted string IDs present in both collections."""
    # TODO(you): compare sets, not row positions.
    raise NotImplementedError("find the overlap")


if __name__ == "__main__":
    train = ["doc-01", "doc-02", "doc-03", "doc-04"]
    test = ["doc-05", "doc-02", "doc-06"]
    print(overlapping_ids(train, test))
