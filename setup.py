"""Setup configuration for LLM Multilingual Safety Evaluation Framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-multilingual-safety-eval",
    version="1.0.0",
    author="Safety Evaluation Team",
    author_email="safety-eval@example.org",
    description="A comprehensive framework for evaluating LLM safety across multiple languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/llm-multilingual-safety-eval",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "rich>=10.0.0",
        "requests>=2.26.0",
        "aiohttp>=3.8.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
        "langdetect>=1.0.9",
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "google-generativeai>=0.1.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "jsonschema>=4.0.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.18.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.961",
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
        ],
        "viz": [
            "bokeh>=2.4.0",
            "altair>=4.2.0",
            "streamlit>=1.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lmse=lmse.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "lmse": ["data/*.json", "data/*.yaml"],
    },
)