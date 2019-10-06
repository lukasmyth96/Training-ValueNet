try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='training-value-net',
    version='1.0.0',
    url='https://github.com/lukasmyth96/Training-ValueNet',
    author='Luka Smyth',
    author_email='lukasmyth@msn.com',
    license='MIT',
    description='A tool for performing automated label cleaning on weakly-supervised classification data',
    packages=['tv_net'],
    install_requires=['tqdm', 'keras', 'scikit-image']
    , dependency_links=[
                        ],
    include_package_data=True,
    python_requires='>=3.4',
    long_description='''Training-ValueNet is a tool that allows users to perform label cleaning on weakly-supervised 
                        data for any classification task. ''',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 3'
    ],
    keywords='weakly-supervised learning label noise',
)
