
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class OptimizationStrategy(Enum):
    EQUAL_WEIGHT = "equal_weight"
    YIELD_WEIGHTED = "yield_weighted"
    RISK_ADJUSTED = "risk_adjusted"
    DIVIDEND_GROWTH = "dividend_growth"
    BALANCED = "balanced"

@dataclass
class PortfolioConstraints:
    min_position_size: float = 0.02  # Minimum 2% position
    max_position_size: float = 0.15  # Maximum 15% position
    min_yield: float = 0.03  # Minimum 3% yield
    max_yield: float = 0.20  # Maximum 20% yield (avoid yield traps)
    max_payout_ratio: float = 0.80  # Maximum 80% payout ratio
    min_market_cap: float = 1e9  # Minimum 1B market cap
    max_debt_to_equity: float = 2.0  # Maximum debt-to-equity ratio

@dataclass
class StockMetrics:
    ticker: str
    price: float
    dividend_yield: float
    payout_ratio: float
    market_cap: float
    debt_to_equity: float
    beta: float
    pe_ratio: float
    dividend_growth_5y: float
    free_cash_flow: float
    roe: float

class DividendStockPortfolio:
    def __init__(self, wig_analysis: 'WIGCompanyAnalysis', available_cash: float, 
                 constraints: Optional[PortfolioConstraints] = None) -> None:
        self.wig_analysis = wig_analysis
        self.available_cash = available_cash
        self.constraints = constraints or PortfolioConstraints()

        # Portfolio data
        self.stock_metrics: Dict[str, StockMetrics] = {}
        self.portfolio: Dict[str, int] = {}  # ticker -> shares
        self.selected_tickers: List[str] = []
        self.optimization_strategy = OptimizationStrategy.BALANCED

        # Portfolio analytics
        self.portfolio_stats: Dict[str, float] = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_stock_metrics(self, tickers: List[str]) -> None:
        """Fetch comprehensive stock metrics for analysis"""
        self.logger.info(f'Fetching comprehensive metrics for {len(tickers)} tickers')

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1y")

                if hist.empty:
                    self.logger.warning(f'No price history for {ticker}')
                    continue

                current_price = hist['Close'].iloc[-1]

                # Extract metrics with proper error handling
                dividend_yield = info.get('dividendYield', 0.0)  # Already decimal
                if dividend_yield is None:
                    dividend_yield = 0.0

                payout_ratio = info.get('payoutRatio', 0.0)
                if payout_ratio is None:
                    payout_ratio = 0.0

                market_cap = info.get('marketCap', 0.0)
                if market_cap is None:
                    market_cap = 0.0

                debt_to_equity = info.get('debtToEquity', 0.0)
                if debt_to_equity is None:
                    debt_to_equity = 0.0
                else:
                    debt_to_equity = debt_to_equity / 100.0  # Convert percentage to ratio

                beta = info.get('beta', 1.0)
                if beta is None:
                    beta = 1.0

                pe_ratio = info.get('trailingPE', 0.0)
                if pe_ratio is None:
                    pe_ratio = 0.0

                # Calculate dividend growth (simplified)
                dividend_growth = info.get('earningsGrowth', 0.0)
                if dividend_growth is None:
                    dividend_growth = 0.0

                free_cash_flow = info.get('freeCashflow', 0.0)
                if free_cash_flow is None:
                    free_cash_flow = 0.0

                roe = info.get('returnOnEquity', 0.0)
                if roe is None:
                    roe = 0.0

                # Create StockMetrics object
                metrics = StockMetrics(
                    ticker=ticker,
                    price=current_price,
                    dividend_yield=dividend_yield,
                    payout_ratio=payout_ratio,
                    market_cap=market_cap,
                    debt_to_equity=debt_to_equity,
                    beta=beta,
                    pe_ratio=pe_ratio,
                    dividend_growth_5y=dividend_growth,
                    free_cash_flow=free_cash_flow,
                    roe=roe
                )

                self.stock_metrics[ticker] = metrics
                self.logger.info(f'{ticker}: Price={current_price:.2f}, Yield={dividend_yield*100:.2f}%')

            except Exception as e:
                self.logger.error(f'Error fetching metrics for {ticker}: {e}')

    def apply_screening_filters(self) -> List[str]:
        """Apply screening filters based on constraints"""
        filtered_tickers = []

        for ticker, metrics in self.stock_metrics.items():
            # Apply constraints
            if (metrics.dividend_yield >= self.constraints.min_yield and
                metrics.dividend_yield <= self.constraints.max_yield and
                metrics.payout_ratio <= self.constraints.max_payout_ratio and
                metrics.market_cap >= self.constraints.min_market_cap and
                metrics.debt_to_equity <= self.constraints.max_debt_to_equity and
                metrics.price > 0):

                filtered_tickers.append(ticker)
                self.logger.info(f'{ticker} passed screening filters')
            else:
                self.logger.info(f'{ticker} filtered out - failed constraints')

        return filtered_tickers

    def select_tickers_with_strategy(self, scenario_key: str, strategy: OptimizationStrategy, 
                                   max_positions: int = 10) -> None:
        """Select tickers based on scenario and optimization strategy"""
        scenarios = self.wig_analysis.get_scenarios()

        if scenario_key not in scenarios:
            raise ValueError(f"Scenario key '{scenario_key}' not found")

        candidate_tickers = scenarios[scenario_key]

        if not candidate_tickers:
            self.logger.warning(f"No tickers found for scenario: {scenario_key}")
            return

        # Fetch metrics for all candidates
        self.fetch_stock_metrics(candidate_tickers)

        # Apply screening filters
        filtered_tickers = self.apply_screening_filters()

        if not filtered_tickers:
            self.logger.warning("No tickers passed screening filters")
            return

        # Select based on strategy
        self.optimization_strategy = strategy
        self.selected_tickers = self._select_by_strategy(filtered_tickers, strategy, max_positions)

        self.logger.info(f'Selected {len(self.selected_tickers)} tickers using {strategy.value} strategy')

    def _select_by_strategy(self, tickers: List[str], strategy: OptimizationStrategy, 
                           max_positions: int) -> List[str]:
        """Select tickers based on optimization strategy"""
        metrics_df = pd.DataFrame([
            {
                'ticker': ticker,
                'yield': self.stock_metrics[ticker].dividend_yield,
                'payout_ratio': self.stock_metrics[ticker].payout_ratio,
                'beta': self.stock_metrics[ticker].beta,
                'roe': self.stock_metrics[ticker].roe,
                'dividend_growth': self.stock_metrics[ticker].dividend_growth_5y,
                'pe_ratio': self.stock_metrics[ticker].pe_ratio,
                'market_cap': self.stock_metrics[ticker].market_cap
            }
            for ticker in tickers
        ])

        if strategy == OptimizationStrategy.EQUAL_WEIGHT:
            # Simple random selection up to max_positions
            selected = tickers[:max_positions]

        elif strategy == OptimizationStrategy.YIELD_WEIGHTED:
            # Select highest yielding stocks
            selected = metrics_df.nlargest(max_positions, 'yield')['ticker'].tolist()

        elif strategy == OptimizationStrategy.DIVIDEND_GROWTH:
            # Select stocks with best dividend growth
            selected = metrics_df.nlargest(max_positions, 'dividend_growth')['ticker'].tolist()

        elif strategy == OptimizationStrategy.RISK_ADJUSTED:
            # Select based on yield/beta ratio
            metrics_df['risk_adjusted_yield'] = metrics_df['yield'] / (metrics_df['beta'] + 0.1)
            selected = metrics_df.nlargest(max_positions, 'risk_adjusted_yield')['ticker'].tolist()

        elif strategy == OptimizationStrategy.BALANCED:
            # Balanced approach using multiple factors
            metrics_df['quality_score'] = (
                metrics_df['yield'] * 0.3 +
                (1 - metrics_df['payout_ratio']) * 0.2 +  # Lower payout is better
                metrics_df['roe'] * 0.2 +
                metrics_df['dividend_growth'] * 0.2 +
                (1 / (metrics_df['beta'] + 0.1)) * 0.1  # Lower beta is better
            )
            selected = metrics_df.nlargest(max_positions, 'quality_score')['ticker'].tolist()

        return selected

    def calculate_optimal_weights(self) -> Dict[str, float]:
        """Calculate optimal portfolio weights based on strategy"""
        if not self.selected_tickers:
            return {}

        weights = {}

        if self.optimization_strategy == OptimizationStrategy.EQUAL_WEIGHT:
            # Equal weights
            weight = 1.0 / len(self.selected_tickers)
            weights = {ticker: weight for ticker in self.selected_tickers}

        elif self.optimization_strategy == OptimizationStrategy.YIELD_WEIGHTED:
            # Weight by dividend yield
            total_yield = sum(self.stock_metrics[ticker].dividend_yield 
                            for ticker in self.selected_tickers)
            if total_yield > 0:
                weights = {ticker: self.stock_metrics[ticker].dividend_yield / total_yield 
                          for ticker in self.selected_tickers}
            else:
                weights = self.calculate_optimal_weights()  # Fallback to equal weight

        elif self.optimization_strategy in [OptimizationStrategy.RISK_ADJUSTED, 
                                          OptimizationStrategy.DIVIDEND_GROWTH,
                                          OptimizationStrategy.BALANCED]:
            # Modified equal weight with constraints
            base_weight = 1.0 / len(self.selected_tickers)
            weights = {}

            for ticker in self.selected_tickers:
                # Apply position size constraints
                weight = max(self.constraints.min_position_size,
                           min(self.constraints.max_position_size, base_weight))
                weights[ticker] = weight

            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            weights = {ticker: weight / total_weight for ticker, weight in weights.items()}

        return weights

    def optimize_portfolio(self) -> None:
        """Optimize portfolio allocation"""
        self.logger.info(f'Optimizing portfolio with PLN {self.available_cash:,.2f}')

        if not self.selected_tickers:
            self.logger.warning('No tickers selected for optimization')
            return

        # Calculate optimal weights
        weights = self.calculate_optimal_weights()

        # Allocate cash based on weights
        self.portfolio.clear()
        total_invested = 0.0

        for ticker, weight in weights.items():
            if ticker not in self.stock_metrics:
                continue

            allocated_cash = self.available_cash * weight
            price = self.stock_metrics[ticker].price

            if price > 0:
                shares = int(allocated_cash // price)
                if shares > 0:
                    self.portfolio[ticker] = shares
                    invested = shares * price
                    total_invested += invested

                    self.logger.info(f'{ticker}: {shares} shares @ PLN {price:.2f} '
                                   f'(PLN {invested:,.2f}, {weight*100:.1f}% target)')

        remaining_cash = self.available_cash - total_invested
        self.logger.info(f'Total invested: PLN {total_invested:,.2f}, '
                        f'Remaining: PLN {remaining_cash:,.2f}')

        # Calculate portfolio statistics
        self._calculate_portfolio_stats()

    def _calculate_portfolio_stats(self) -> None:
        """Calculate comprehensive portfolio statistics"""
        if not self.portfolio:
            return

        total_value = sum(shares * self.stock_metrics[ticker].price 
                         for ticker, shares in self.portfolio.items())

        # Portfolio metrics
        weighted_yield = 0.0
        weighted_payout_ratio = 0.0
        weighted_beta = 0.0
        weighted_roe = 0.0

        for ticker, shares in self.portfolio.items():
            if ticker in self.stock_metrics:
                weight = (shares * self.stock_metrics[ticker].price) / total_value
                metrics = self.stock_metrics[ticker]

                weighted_yield += weight * metrics.dividend_yield
                weighted_payout_ratio += weight * metrics.payout_ratio
                weighted_beta += weight * metrics.beta
                weighted_roe += weight * metrics.roe

        self.portfolio_stats = {
            'total_value': total_value,
            'cash_utilization': total_value / self.available_cash,
            'weighted_dividend_yield': weighted_yield,
            'weighted_payout_ratio': weighted_payout_ratio,
            'weighted_beta': weighted_beta,
            'weighted_roe': weighted_roe,
            'number_of_positions': len(self.portfolio),
            'average_position_size': total_value / len(self.portfolio) if self.portfolio else 0
        }

    def calculate_dividend_income(self) -> Dict[str, float]:
        """Calculate expected dividend income with detailed breakdown"""
        self.logger.info("\n=== Dividend Income Analysis ===")

        dividend_breakdown = {}
        total_annual_dividends = 0.0
        total_quarterly_dividends = 0.0

        for ticker, shares in self.portfolio.items():
            if ticker in self.stock_metrics:
                metrics = self.stock_metrics[ticker]
                position_value = shares * metrics.price
                annual_dividend_per_share = metrics.dividend_yield * metrics.price
                total_dividends = annual_dividend_per_share * shares
                quarterly_dividends = total_dividends / 4  # Assuming quarterly payments

                dividend_breakdown[ticker] = {
                    'shares': shares,
                    'price': metrics.price,
                    'position_value': position_value,
                    'dividend_yield': metrics.dividend_yield,
                    'annual_dividend_per_share': annual_dividend_per_share,
                    'total_annual_dividends': total_dividends,
                    'quarterly_dividends': quarterly_dividends
                }

                total_annual_dividends += total_dividends
                total_quarterly_dividends += quarterly_dividends

                self.logger.info(f'{ticker}: {shares} shares @ {metrics.dividend_yield*100:.2f}% '
                               f'→ PLN {total_dividends:,.2f} annually')

        # Portfolio dividend summary
        portfolio_yield = (total_annual_dividends / self.portfolio_stats.get('total_value', 1)) * 100

        summary = {
            'total_annual_dividends': total_annual_dividends,
            'total_quarterly_dividends': total_quarterly_dividends,
            'portfolio_dividend_yield': portfolio_yield,
            'monthly_estimate': total_annual_dividends / 12,
            'breakdown': dividend_breakdown
        }

        self.logger.info(f"\nPortfolio Dividend Summary:")
        self.logger.info(f"Total Annual Dividends: PLN {total_annual_dividends:,.2f}")
        self.logger.info(f"Portfolio Dividend Yield: {portfolio_yield:.2f}%")
        self.logger.info(f"Monthly Estimate: PLN {summary['monthly_estimate']:,.2f}")

        return summary

    def generate_portfolio_report(self) -> str:
        """Generate comprehensive portfolio report"""
        if not self.portfolio:
            return "No portfolio to report"

        report = []
        report.append("=" * 60)
        report.append("DIVIDEND PORTFOLIO ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Available Cash: PLN {self.available_cash:,.2f}")
        report.append(f"Optimization Strategy: {self.optimization_strategy.value}")
        report.append("")

        # Portfolio Overview
        report.append("PORTFOLIO OVERVIEW")
        report.append("-" * 30)
        stats = self.portfolio_stats
        report.append(f"Total Portfolio Value: PLN {stats.get('total_value', 0):,.2f}")
        report.append(f"Cash Utilization: {stats.get('cash_utilization', 0)*100:.1f}%")
        report.append(f"Number of Positions: {stats.get('number_of_positions', 0)}")
        report.append(f"Average Position Size: PLN {stats.get('average_position_size', 0):,.2f}")
        report.append("")

        # Portfolio Metrics
        report.append("PORTFOLIO METRICS")
        report.append("-" * 30)
        report.append(f"Weighted Dividend Yield: {stats.get('weighted_dividend_yield', 0)*100:.2f}%")
        report.append(f"Weighted Payout Ratio: {stats.get('weighted_payout_ratio', 0)*100:.1f}%")
        report.append(f"Weighted Beta: {stats.get('weighted_beta', 0):.2f}")
        report.append(f"Weighted ROE: {stats.get('weighted_roe', 0)*100:.1f}%")
        report.append("")

        # Holdings Detail
        report.append("HOLDINGS DETAIL")
        report.append("-" * 50)
        report.append(f"{'Ticker':<8} {'Shares':<8} {'Price':<10} {'Value':<12} {'Yield':<8} {'Weight'}")
        report.append("-" * 50)

        total_value = stats.get('total_value', 1)
        for ticker, shares in self.portfolio.items():
            if ticker in self.stock_metrics:
                metrics = self.stock_metrics[ticker]
                value = shares * metrics.price
                weight = value / total_value * 100

                report.append(f"{ticker:<8} {shares:<8} {metrics.price:<10.2f} "
                            f"{value:<12,.0f} {metrics.dividend_yield*100:<8.1f}% {weight:.1f}%")

        report.append("")

        # Risk Analysis
        report.append("RISK ANALYSIS")
        report.append("-" * 30)

        # Sector concentration (if data available)
        report.append("Position Concentration:")
        max_position = max((shares * self.stock_metrics[ticker].price / total_value) * 100 
                          for ticker, shares in self.portfolio.items() 
                          if ticker in self.stock_metrics)
        report.append(f"Largest Position: {max_position:.1f}%")

        return "\n".join(report)

    def get_portfolio(self) -> Dict[str, int]:
        """Get current portfolio holdings"""
        return self.portfolio.copy()

    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get portfolio statistics"""
        return self.portfolio_stats.copy()

    def get_remaining_cash(self) -> float:
        """Calculate remaining cash after investments"""
        total_invested = sum(shares * self.stock_metrics[ticker].price 
                           for ticker, shares in self.portfolio.items() 
                           if ticker in self.stock_metrics)
        return self.available_cash - total_invested

# Example usage and testing
if __name__ == "__main__":
    # This would typically be used with a WIGCompanyAnalysis instance
    # analyzer = WIGCompanyAnalysis(wig_tickers)
    # analyzer.fetch_data()

    # portfolio = DividendStockPortfolio(analyzer, 100000.0)  # 100k PLN
    # portfolio.select_tickers_with_strategy(
    #     "Dividend Yield above 5% and Positive Earnings Growth", 
    #     OptimizationStrategy.BALANCED, 
    #     max_positions=8
    # )
    # portfolio.optimize_portfolio()
    # dividend_summary = portfolio.calculate_dividend_income()
    # print(portfolio.generate_portfolio_report())
    pass
