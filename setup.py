import setuptools

setuptools.setup(
        name = 'ponask',
        packages = setuptools.find_packages(),
        install_requires=[
            'numpy', 'torch', 'tqdm'],)

