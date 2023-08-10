from setuptools import setup, find_packages

setup(
    name='HyperModule',
    version='0.1.0',
    description='Wrapper for managing neural network training processes',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kevinkevin556/HyperModule',
    author='Zhen-Lun Hong',
    author_email='your.email@example.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib',
        'numpy',
        'tqdm',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
