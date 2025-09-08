"""
Setup script for Bot trading bot package.
"""

from setuptools import setup, find_packages

setup(
    name="Bot",
    version="1.0.0",
    description="Cryptocurrency Trading Bot with ML and RL capabilities",
    author="Jesse",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "pyarrow>=12.0.0",
        # Machine Learning
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "lightgbm>=4.0.0",
        "stable-baselines3[extra]>=2.7.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        # Data processing and APIs
        "aiohttp>=3.8.0",
        "aiosqlite>=0.19.0",
        "psutil>=5.9.0",
        "requests>=2.31.0",
        "httpx>=0.24.0",
        # Trading and APIs
        "python-binance>=1.0.0",
        "ccxt>=4.0.0",
        "yfinance>=0.2.18",
        # Notifications
        "python-telegram-bot>=20.0",
        # Experiment tracking
        "mlflow>=2.5.0",
        "wandb>=0.15.0",
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        # Reinforcement Learning Environment
        "gymnasium>=0.29.0",
        # Utilities
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
        "click>=8.1.0",
        "rich>=13.4.0",
        "arrow>=1.2.3",
        "python-dateutil>=2.8.0",
        "pytz>=2023.3",
        # Hyperparameter optimization
        "optuna>=3.6.0",
        # Data analysis and statistics
        "statsmodels>=0.14.0",
        "ta>=0.10.2",
        "pandas-ta>=0.3.14b0",
        # Additional utilities for production
        "schedule>=1.2.0",
        # Configuration management
        "pydantic>=2.0.0",
        "environs>=9.5.0",
        # Database support
        "sqlalchemy>=2.0.0",
        # Better JSON handling
        "orjson>=3.9.0",
        # WebSocket support
        "websockets>=11.0.3",
        # Better error handling
        "tenacity>=8.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "gpu": [
            "torchaudio>=2.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
