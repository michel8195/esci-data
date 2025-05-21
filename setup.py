from setuptools import setup, find_packages

setup(
    name="esci_ranking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.30.0",
        "pandas>=1.5.3",
        "scikit-learn>=1.5.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
    ],
    author="ESCI Ranking Team",
    author_email="your.email@example.com",
    description="A package for query-product ranking using two-tower models",
    keywords="ranking, e-commerce, search, machine learning",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "esci-train=esci_ranking.train.train_model:main",
            "esci-infer=esci_ranking.inference.run_inference:main",
        ],
    },
)
