from splits import LevelBook, assign, group_key, split_of


def imgs(n, prefix="p", video=None):
    out = []
    for i in range(n):
        origin = f"video:{video}#{i}" if video else "upload"
        out.append({"id": f"{prefix}{i}", "filename": f"{prefix}{i}.jpg", "origin": origin})
    return out


def test_video_frames_share_a_split():
    frames = imgs(40, "f", video="clip1.mp4")
    assert len({split_of(f) for f in frames}) == 1
    assert group_key(frames[0]) == "video:clip1.mp4"


def test_split_is_stable_as_data_grows():
    a = imgs(200)
    before = {i["id"]: split_of(i) for i in a}
    a += imgs(200, "q")
    assert all(split_of(i) == before[i["id"]] for i in a if i["id"] in before)


def test_rough_fractions_on_many_groups():
    s = assign(imgs(5000))
    assert 0.17 < len(s["val"]) / 5000 < 0.23
    assert 0.06 < len(s["dev"]) / 5000 < 0.10


def test_first_level_waits_for_enough_heldout():
    book = LevelBook(min_new=10, min_heldout=30)
    level, reason = book.maybe_new_level(imgs(20), now=0)
    assert level is None and "held-out" in reason
    level, _ = book.maybe_new_level(imgs(400), now=1)
    assert level is not None and level.level_id == 0 and len(book.heldout) >= 30


def test_levels_are_nested_and_heldout_frozen():
    book = LevelBook(min_new=10, min_heldout=5)
    data = imgs(300)
    l0, _ = book.maybe_new_level(data, now=0)
    frozen = list(book.heldout)
    data2 = data + imgs(300, "q")
    l1, reason = book.maybe_new_level(data2, now=1)
    assert l1 is not None and set(l0.train) <= set(l1.train) and not l1.missing_from_previous
    assert book.heldout == frozen                       # later held-out images don't join the test
    assert len(book.unused_heldout(data2)) > 0
    assert not set(book.heldout) & set(l1.train + l1.dev)  # no held-out image ever reaches a trainer


def test_needs_min_new_for_next_level():
    book = LevelBook(min_new=50, min_heldout=5)
    data = imgs(300)
    book.maybe_new_level(data, now=0)
    level, reason = book.maybe_new_level(data + imgs(10, "q"), now=1)
    assert level is None and "need 50" in reason


def test_removed_images_are_reported_not_hidden():
    book = LevelBook(min_new=1, min_heldout=5)
    data = imgs(300)
    l0, _ = book.maybe_new_level(data, now=0)
    gone = l0.train[0]
    data2 = [i for i in data if i["id"] != gone] + imgs(100, "q")
    l1, reason = book.maybe_new_level(data2, now=1)
    assert gone in l1.missing_from_previous and "NOT NESTED" in reason


def test_roundtrip_json():
    book = LevelBook(min_new=1, min_heldout=5)
    book.maybe_new_level(imgs(300), now=0)
    again = LevelBook.from_json(book.to_json())
    assert again.to_json() == book.to_json()
