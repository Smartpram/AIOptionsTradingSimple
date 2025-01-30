from setuptools import setup, find_packages

setup(
    name='options_trading',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'yfinance',
        'TA-Lib',
        'keras',
        'tensorflow',
        'scipy',
        'matplotlib',
    ],
    author='SmartPram',
    description='Options Trading with Technical Analysis',
    url='https://github.com/Smartpram/AIOptionsTradingSimple',  # Update with your GitHub URL
)
