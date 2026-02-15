"""
Entry point for gold backtesting.

Runs the full pipeline:
Polygon data -> indicators -> signals -> backtest -> plots
"""

from src.gold_backtest.bot import main


if __name__ == "__main__":
    main()
