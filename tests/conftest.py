# Imported before the test modules, which call `pytest.importorskip("tensorflow")` first:
# importing verres here pins KERAS_BACKEND and fails early on a non-Keras-3 environment.
import verres  # noqa: F401
