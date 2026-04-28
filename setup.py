from setuptools import setup, find_packages

setup(
    name='SpectrumReconstruction',
    version='0.2',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'numpy>=2.0.1, <3.0.0',
        'scipy>=1.14.0, <2.0.0',
        'scikit-learn>=1.5.2, <2.0.0',
        'pandas>=2.2.3, <3.0.0',
        'plotly>= 5.24.1, <7.0.0',
        'numba>=0.61.0, <1.0.0'
    ],
    author='hcy2206',
    author_email='97283537+hcy2206@users.noreply.github.com.',
    description='A Python package for spectrum reconstruction and simulation of photodetector responses',
    long_description_content_type='markdown',
    long_description='''
    # SpectrumReconstruction
    
    A Python package for spectrum reconstruction and simulation, providing tools for modeling photodetector responses and reconstructing unknown spectra from detector signals.
    
    ## Features
    
    - *Advanced photodetector simulation* with adjustable parameters (bandgap, quantum efficiency, bias modes)
    - *Flexible spectrum modeling* using Gaussian and blackbody radiation models
    - *Multiple reconstruction methods* including:
        - Direct linear regression
        - L1 regularization (LASSO)
        - L2 regularization (Ridge)
        - ElasticNet regularization
    - *Interactive visualization* of responsivity curves, spectra, and reconstruction results
    - *Noise simulation* for realistic testing scenarios
    - *Physics-based models* incorporating semiconductor physics and optical principles
    
    This package is designed for researchers and engineers working in spectroscopy, optical sensing, semiconductor photodetectors, and similar fields requiring advanced spectral analysis and reconstruction capabilities.
    ''',
    license='MIT',
    keywords='spectrum reconstruction photodetector simulation',
    python_requires='>=3.10',
    url='https://github.com/hcy2206/SpectrumReconstruction'
)
