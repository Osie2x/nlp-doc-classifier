from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="documind",
    version="1.0.0",
    author="DocuMind Contributors",
    description="NLP-Powered Document Intelligence & Classification Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/osie2x/nlp-doc-classifier",
    packages=find_packages(exclude=["tests*", "scripts*", "notebooks*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "documind-train=scripts.train_model:main",
            "documind-evaluate=scripts.evaluate_model:main",
            "documind-serve=documind.api.app:main",
            "documind-generate=scripts.generate_sample_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "documind": ["config/*.yaml"],
    },
)
