#!/usr/bin/env python3
"""S37 scratch probe of the pinned harness's HFLM construction. Deleted before commit."""
import inspect
import sys


def main() -> int:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    print("lm_eval", lm_eval.__version__)
    print("HFLM init signature:", inspect.signature(HFLM.__init__))
    source = inspect.getsource(HFLM._create_model)
    print("--- _create_model ---")
    print(source[:4000])
    print("--- _model use ---")
    src = inspect.getsource(HFLM)
    for needle in ["self._model", "_model =", "def forward", "def loglikelihood"]:
        print(needle, src.count(needle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
