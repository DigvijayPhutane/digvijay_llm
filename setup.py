from setuptools import setup, find_packages

setup(
    name="digvijay_llm",
    version="0.1.0",
    description="Run large LLMs (incl. 70B-class) on low-RAM machines by streaming weights from disk instead of holding the full model in RAM.",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Digvijay",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "psutil>=5.9.0",
    ],
    extras_require={
        "gguf": ["llama-cpp-python>=0.2.0"],
        "hf": ["torch>=2.0.0", "transformers>=4.38.0", "accelerate>=0.27.0", "safetensors>=0.4.0"],
        "all": [
            "llama-cpp-python>=0.2.0",
            "torch>=2.0.0",
            "transformers>=4.38.0",
            "accelerate>=0.27.0",
            "safetensors>=0.4.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
