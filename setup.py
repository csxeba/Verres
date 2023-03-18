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
                      "numpy",
                      "scipy",
                      "matplotlib",
                      "tensorflow>=2.2",
                      "IPython",
                      "tqdm",
                      "pyyaml",
                      "pydantic",
                      "git+https://github.com/csxeba/Artifactorium.git"],
    extras_require={"all": ["tfkerassurgeon"],
                    "pruning": ["tfkerassurgeon"],
                    "gpu": ["tensorflow-gpu>=2.2"]}

)
