import pytest
from streamlit.testing.v1 import AppTest


def test_app_ui_loads_and_runs():
    at = AppTest.from_file("../app_streamlit.py", default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an exception: {at.exception}"
