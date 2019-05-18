import os


def ensure_working_directory():
    if os.path.join("verres", "experiments") in os.getcwd():
        os.chdir("..")