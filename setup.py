from setuptools import find_packages, setup

__version__ = '0.0.0'

with open('requirements.txt', 'r') as req:
    install_requires = req.readlines()

setup(
    name='MotiFiesta',
    version=__version__,
    description='Neural Approximate Motif Mining (user-friendly version)',
    author = "Carlos Oliver",
    author_email = "carlos.oliver@bsse.ethz.ch",
    keywords = ['pattern-mining',
                'deep-learning',
                'graphs',
                'pytorch',
                'torch-geometric',
                ],
    python_requires='>=3.7',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    scripts=['scripts/motifiesta', 'scripts/build_data_motifiesta']
)

