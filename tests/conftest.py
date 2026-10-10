# Imported before the test modules, which call `pytest.importorskip("tensorflow")` first:
# importing verres here sets TF_USE_LEGACY_KERAS before tensorflow is loaded.
import verres  # noqa: F401
