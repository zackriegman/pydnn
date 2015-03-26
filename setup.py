import os
from setuptools import find_packages
from setuptools import setup

root = os.path.abspath(os.path.dirname(__file__))

setup(
    name='pydnn',
    version='0.0.dev',
    description='deep neural network library in Python',
    long_description=open(os.path.join(root, 'README.rst')).read(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries'
        ],
    keywords='neural network deep learning AI machine learning',
    author='Isaac Kriegman',
    author_email='zackriegman@gmail.com',
    url='https://github.com/zackriegman/pydnn',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'Theano>=0.7.0rc1.dev', 'pyyaml', 'boto', 'pandas'],
    extras_require={'docs': ['Sphinx']},
    )