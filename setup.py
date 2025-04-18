from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sparktts",
    version="0.3.0",
    author="Ashraff Hathibelagal",
    description="A fork of Spark-TTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hathibelagal-dev/Spark-TTS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "sparktts=cli.inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ai text-to-speech speech-synthesis nlp transformer voice",
    project_urls={
        "Source": "https://github.com/hathibelagal-dev/Spark-TTS",
        "Tracker": "https://github.com/hathibelagal-dev/Spark-TTS/issues",
    },
)
