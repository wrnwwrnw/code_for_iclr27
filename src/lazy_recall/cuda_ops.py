from functools import lru_cache


@lru_cache(maxsize=1)
def extension():
    try:
        from lazy_recall import _C
    except ImportError as error:
        raise RuntimeError(
            "LazyRecall CUDA extension missing. Run: "
            "python -m pip install --no-build-isolation --no-deps -e ."
        ) from error
    return _C

