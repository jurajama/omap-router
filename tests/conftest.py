import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).parent))

from omap_router.config import Settings  # noqa: E402
from synthetic import make_map  # noqa: E402


@pytest.fixture
def cfg() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def synth():
    return make_map(seed=1)
