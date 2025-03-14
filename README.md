# Expert-Level Backtesting System for Market-Timing Strategies

A sophisticated backtesting system designed for market-timing strategies, utilizing `vectorbt` and focusing on complex financial time series analysis.

## Features

- Data-driven market timing strategies
- Complex signal generation using multiple indicators
- Performance evaluation against buy-and-hold benchmarks
- Comprehensive statistical analysis
- Frequency-aware calculations and adjustments

## Key Components

- Technical Indicators
- Statistical Features
- Momentum/Mean Reversion Indicators
- Regime Detection
- Advanced Data Transformations
- Time Series Features

## Requirements

- Python 3.x
- vectorbt
- pandas
- numpy
- Other dependencies specified in poetry.lock

## Usage

The system is designed to work with financial time series data, particularly focusing on the `cad_ig_er_index` for market timing strategies.

## Performance Metrics

- Annualized Returns
- Sharpe Ratio
- Additional metrics via vectorbt's Portfolio Stats

## Project Structure

```
backtrader/
├── pulling_data/         # Data storage and retrieval
├── notebooks/           # Jupyter notebooks for analysis
├── general_notes/       # Documentation and notes
└── README.md           # This file
```

## Setup

This project uses Poetry for dependency management. To get started:

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

4. Install the Jupyter kernel for the poetry environment:
```bash
poetry run python -m ipykernel install --user --name tajana-MtNpgr4y-py3.11 --display-name "Poetry (backtester)"
```

## License

[Add your license here]
