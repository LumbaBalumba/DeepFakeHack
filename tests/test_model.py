"""This module pertforms PyTest testing for model implemented in main package."""

import os


def test_nothing(tmp_path: os.PathLike):
    """Dummy test just to check if workflows are correctly set up."""
    assert tmp_path is not None
