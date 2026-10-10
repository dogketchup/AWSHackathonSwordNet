"""Point every test at a throwaway data dir and FiftyOne dataset, set before swordnet is imported."""
import os
import tempfile
import uuid

TMP = tempfile.mkdtemp(prefix="swordnet-test-")
os.environ["SWORDNET_HOME"] = TMP
os.environ["SWORDNET_FO_DATASET"] = f"swordnet_test_{uuid.uuid4().hex[:8]}"
os.environ["SWORDNET_DB"] = "local"  # tests run the database operations in-process, never on the cluster

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup():
    yield
    import fiftyone as fo
    from swordnet.storage import store
    if fo.dataset_exists(store.DATASET):
        fo.delete_dataset(store.DATASET)
    for name in store.mongo().list_collection_names():
        if name.startswith(store.DATASET + "_"):
            store.mongo().drop_collection(name)


@pytest.fixture
def make_image():
    """Write a synthetic image into INGEST_DIR and register it; returns the sample id."""
    import cv2
    import numpy as np
    from swordnet import config
    from swordnet.storage import store

    def _make(collection="test", origin=None, w=320, h=240):
        config.ensure_dirs()
        path = config.INGEST_DIR / f"t{uuid.uuid4().hex[:10]}.jpg"
        img = np.random.default_rng().integers(60, 120, (h, w, 3), dtype=np.uint8)  # unique bytes per image
        cv2.line(img, (40, 200), (280, 40), (230, 230, 230), 6)  # a "blade"
        cv2.imwrite(str(path), img)
        return store.add_image(path, collection, origin or f"upload:{path.name}")
    return _make
