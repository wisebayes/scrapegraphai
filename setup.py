from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scrapegraphai",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for intelligent web scraping using graph-based approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scrapegraphai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Browsers",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Add your dependencies here
        # "requests>=2.25.0",
        # "beautifulsoup4>=4.9.0",
        # "networkx>=2.5",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/scrapegraphai/issues",
        "Source": "https://github.com/yourusername/scrapegraphai",
    },
) 