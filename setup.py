from setuptools import setup, find_packages


setup(
    name="choreo-mini",  # Replace with your project name
    version="0.1.0",
    description="A short description of your package",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/sivasathivelkandasamy/choreo-mini",
    packages=find_packages(include=["choreo_mini", "choreo_mini.*"]),
    include_package_data=True,
    install_requires=[
        # put your runtime dependencies here, e.g. "requests>=2.25.1",
        "jinja2>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",  # development/test dependencies
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        # console scripts provide a CLI command when the package is installed
        "console_scripts": [
            "choreo-mini=choreo_mini.cli:main",
            "choreo_mini=choreo_mini.cli:main",
        ],
    },
)
