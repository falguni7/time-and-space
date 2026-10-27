import importlib
import os
import site

import pytest

SUBPACKAGES = ["sim", "engine", "stream", "agents", "mcp_server", "web"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    module = importlib.import_module(f"presslens.{name}")
    assert module.__doc__


@pytest.mark.skipif("APPTAINER_CONTAINER" not in os.environ, reason="only meaningful inside the dev container")
def test_runs_in_isolated_container_env():
    assert os.environ.get("CONDA_DEFAULT_ENV") == "presslens_3_11_helma"
    assert not site.ENABLE_USER_SITE


def test_core_dependencies_available():
    import duckdb
    import numpy
    import pandas
    import pyarrow  # noqa: F401
    import sklearn  # noqa: F401

    assert numpy.__version__.startswith("1.")
    con = duckdb.connect()
    con.register("t", pandas.DataFrame({"x": [1, 2, 3]}))
    assert con.execute("select sum(x) from t").fetchone()[0] == 6
