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
                      "tensorflow>=2.2"],
    extras_require={"all": ["torch", "tfkerassurgeon"],
                    "torch": ["torch"],
                    "pruning": ["tfkerassurgeon"]}

)
