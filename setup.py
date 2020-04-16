import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='ucimlr',
    version='0.3.0',
    author='Isac Arnekvist',
    author_email='isac.arnekvist@gmail.com',
    description='Easy access to datasets from the UCI Machine Learning Repository',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isacarnekvist/ucimlr",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18.2',
        'pandas>=1.0.3',
        'sklearn>=0.0',
        'unlzw>=0.1.1',
        'xlrd >= 1.0.0',
    ]
)
