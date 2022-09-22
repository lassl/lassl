from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

with open(here / "requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")
install_requires = [p.strip() for p in install_requires]

setup(
    name="lassl",
    version="0.2.0",
    author="seopbo, bzantium, iron-ij, monologg",
    author_email="bsk0130@gmail.com, ryumin93@gmail.com, yij1126@gmail.com, adieujw@gmail.com",
    description="Easy framework for pre-training language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="NLP transformer lm llm",
    license="Apache",
    url="https://github.com/lassl/lassl",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"lassl": ["includes/tqdm.h", "csrc/dataset_utils.cpp"]},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=install_requires,  # External packages as dependencies
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
