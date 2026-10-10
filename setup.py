from setuptools import setup
from setuptools import find_packages

setup(
    name='verres',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/csxeba/Verres.git',
    license='MIT',
    author='Csxeba',
    author_email='csxeba@gmail.com',
    description='Curiosity',
    install_requires=["opencv-python>=4.0",
                      "numpy<2",
                      "matplotlib",
                      "tensorflow==2.15.*",
                      "IPython",
                      "pycocotools",
                      "pyyaml",
                      "pydantic",
                      "artifactorium @ git+https://github.com/csxeba/Artifactorium.git"]

)
