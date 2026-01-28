from setuptools import setup, find_packages

setup(
    name="plotlybrain",
    version="0.1.0",
    description="Allen Brain Atlas visualization and scoring utilities",
    author="Anna Teruel-Sanchis & Konrad Danielewski",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "h5py",
        "plotly",
    ],
)