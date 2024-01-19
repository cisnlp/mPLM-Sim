#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Peiqin Lin, Chengzhi Hu",
    author_email='lpq29743@gmail.com, Chengzhi.Hu@campus.lmu.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="mPLM-Sim: Better Cross-Lingual Similarity and Transfer in Multilingual Pretrained Language Models",
    entry_points={
        'console_scripts': [
            'mplm_sim=mplm_sim.cli:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='mplm_sim',
    name='mplm_sim',
    packages=find_packages(include=['mplm_sim', 'mplm_sim.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/cisnlp/mplm_sim',
    version='0.1.0',
    zip_safe=False,
)
