"""Suite-wide options. `--catalog-layout columnar` (N1F-6, parent docs/N1F6_PROTOCOL.md §3 C2) runs
every test with the columnar catalog as the default layout, to show it answers as the resident
one does; without the option nothing changes. Every test closes the stores it opened (below)."""

import weakref

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--catalog-layout",
        default=None,
        choices=("resident", "columnar"),
        help="the default catalog layout for this run (N1F-6)",
    )


def pytest_configure(config):
    layout = config.getoption("--catalog-layout")
    if layout:
        from rvg.storage import catalog

        catalog.DEFAULT_LAYOUT = layout


@pytest.fixture(autouse=True)
def _close_what_the_test_opened(monkeypatch):
    """0.1.0 (parent PUB-4): the state stores and local exact indexes a test opened in this
    process are closed after it, so their files and memory maps do not wait for the garbage
    collector (a store and its storage refer to each other) and the suite runs within macOS's
    default `ulimit -n 256`. Only references are kept; what closing releases is checked by the
    tests themselves (tests/test_fd_bounds.py)."""
    from rvg.storage.local_exact import LocalExactIndex
    from rvg.storage.sqlite import SQLiteState

    opened = []
    for cls in (SQLiteState, LocalExactIndex):

        def tracked(self, *args, _init=cls.__init__, **kwargs):
            _init(self, *args, **kwargs)
            opened.append(weakref.ref(self))

        monkeypatch.setattr(cls, "__init__", tracked)
    yield
    for ref in reversed(opened):
        target = ref()
        if target is not None:
            target.close()
