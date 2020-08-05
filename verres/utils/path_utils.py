import os


def ensure_working_directory():
    if os.path.split(os.getcwd())[-1] in ["experiments", "keepers"]:
        os.chdir("..")
