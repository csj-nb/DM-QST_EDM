from setuptools import setup, find_packages

setup(
    name="dm-qst",
    version="0.1.0",
    description="Denoising Diffusion Probabilistic Models for Quantum State Tomography",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<2.0",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "torch>=2.2",
        "tqdm>=4.65",
        "pyyaml>=6.0",
        "qiskit>=1.0",
        "qutip>=5.0",
    ],
)
