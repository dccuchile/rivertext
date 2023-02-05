from distutils.core import setup

setup(
    name="rivertext",  # How you named your package folder (MyLib)
    packages=["rivertext"],  # Chose the same as "name"
    version="0.0.1",  # Start with a small number and increase it with every change you make
    license="MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="A Library for Training and Evaluating Incremental Word Embeddings from Text Data Streams",  # Give a short description about your library
    author="Gabriel Iturra",  # Type in your name
    author_email="gabrieliturrab@ug.uchile.cl   ",  # Type in your E-Mail
    url="https://github.com/dccuchile/rivertext",  # Provide either the link to your github or to your website
    download_url="https://github.com/dccuchile/rivertext/archive/refs/tags/v.0.0.1.tar.gz",  # I explain this later on
    keywords=[
        "word-embedding",
        "incremental-learning",
        "streaming-data",
    ],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        "nltk",
        "numpy",
        "river",
        "scikit_learn",
        "scipy",
        "torch",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3.10",  # Specify which pyhton versions that you want to support
    ],
)
