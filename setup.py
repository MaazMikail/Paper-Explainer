from setuptools import setup, find_packages

setup(
    name="ai_engine",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
    ],
    extras_require={
        "dev": ["pytest", "jupyter"]
    }
)