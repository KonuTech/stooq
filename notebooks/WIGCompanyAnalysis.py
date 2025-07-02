
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import List, Dict
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WIGCompanyAnalysis:
    def __init__(self, tickers: List[str], palette_name: str = "husl") -> None:
        self.tickers = tickers
        self.palette_name = palette_name
        self.dividend_yields_df = pd.DataFrame()
        self.metrics_df = pd.DataFrame()
        self.data_fetch_date = datetime.now().strftime('%Y-%m-%d')
        self.scenarios = {}

    def fetch_data(self) -> None:
        logging.info('Fetching data for tickers: %s', self.tickers)
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                self._process_dividend_data(stock, ticker)
                self._process_metrics_data(stock, ticker)
            except Exception as e:
                logging.error('Error fetching data for %s: %s', ticker, e)
        self._calculate_scenarios()

    def _process_dividend_data(self, stock: yf.Ticker, ticker: str) -> None:
        """Process dividend data with corrected yield calculations"""
        try:
            # Get dividend history
            dividends = stock.dividends
            if dividends.empty:
                logging.warning(f'No dividend data found for {ticker}')
                return

            # Get price history
            price_history = stock.history(period="max")

            # Convert to DataFrame and reset index
            div_df = dividends.reset_index()
            div_df.columns = ['Ex-Dividend Date', 'Dividends']

            # Calculate annualized dividend yield for each ex-dividend date
            results = []

            for _, row in div_df.iterrows():
                ex_date = row['Ex-Dividend Date']
                dividend = row['Dividends']

                # Get stock price on or closest to ex-dividend date
                try:
                    # Find closest price date (within 5 days)
                    price_window = price_history[
                        (price_history.index >= ex_date - timedelta(days=5)) & 
                        (price_history.index <= ex_date + timedelta(days=5))
                    ]

                    if not price_window.empty:
                        # Use closing price closest to ex-dividend date
                        closest_price = price_window.iloc[0]['Close']

                        # Calculate annualized dividend yield
                        # Get trailing 12-month dividends as of this date
                        trailing_dividends = dividends[
                            (dividends.index >= ex_date - timedelta(days=365)) & 
                            (dividends.index <= ex_date)
                        ].sum()

                        if trailing_dividends > 0 and closest_price > 0:
                            annualized_yield = (trailing_dividends / closest_price) * 100

                            results.append({
                                'Ticker': ticker,
                                'Ex-Dividend Date': ex_date,
                                'Dividends': dividend,
                                'Close': closest_price,
                                'Trailing_12M_Dividends': trailing_dividends,
                                'Dividend Yield (%)': annualized_yield
                            })

                except Exception as e:
                    logging.warning(f'Error processing dividend date {ex_date} for {ticker}: {e}')
                    continue

            if results:
                result_df = pd.DataFrame(results)
                self.dividend_yields_df = pd.concat([self.dividend_yields_df, result_df], ignore_index=True)

        except Exception as e:
            logging.error(f'Error processing dividend data for {ticker}: {e}')

    def _process_metrics_data(self, stock: yf.Ticker, ticker: str) -> None:
        """Process current metrics data with corrected calculations"""
        try:
            info = stock.info

            # Fix dividend yield calculation - yfinance returns as decimal (0.05 = 5%)
            dividend_yield = info.get('dividendYield', None)
            if dividend_yield is not None:
                # dividend_yield_pct = dividend_yield * 100  # Convert to percentage
                dividend_yield_pct = dividend_yield
            else:
                dividend_yield_pct = float('nan')

            # Fix payout ratio calculation
            payout_ratio = info.get('payoutRatio', None)
            if payout_ratio is not None:
                payout_ratio_pct = payout_ratio * 100  # Convert to percentage
                # payout_ratio_pct = payout_ratio
            else:
                payout_ratio_pct = float('nan')

            # Fix earnings growth calculation
            earnings_growth = info.get('earningsGrowth', None)
            if earnings_growth is not None:
                earnings_growth_pct = earnings_growth * 100  # Convert to percentage
                # earnings_growth_pct = earnings_growth
            else:
                earnings_growth_pct = float('nan')

            # Fix ROE calculation
            roe = info.get('returnOnEquity', None)
            if roe is not None:
                roe_pct = roe * 100  # Convert to percentage
                # roe_pct = roe
            else:
                roe_pct = float('nan')

            # Free cash flow in billions
            free_cashflow = info.get('freeCashflow', None)
            if free_cashflow is not None:
                fcf_billions = free_cashflow / 1e9
            else:
                fcf_billions = float('nan')

            metrics = {
                'Ticker': ticker,
                'Dividend Yield (%)': dividend_yield_pct,
                'Payout Ratio (%)': payout_ratio_pct,
                'Free Cash Flow (Billion $)': fcf_billions,
                'Earnings Growth (%)': earnings_growth_pct,
                'ROE (%)': roe_pct,
                'Debt-to-Equity Ratio': info.get('debtToEquity', float('nan')),
                'Interest Coverage Ratio': info.get('interestCoverage', float('nan')),
            }

            self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([metrics])], ignore_index=True)

        except Exception as e:
            logging.error(f'Error processing metrics for {ticker}: {e}')

    def _calculate_scenarios(self) -> None:
        """Calculate investment scenarios with data validation"""
        # Filter out rows with invalid data
        valid_data = self.metrics_df.dropna(subset=['Dividend Yield (%)', 'Earnings Growth (%)'])

        self.scenarios = {
            'Dividend Yield above 5% and Positive Earnings Growth': valid_data[
                (valid_data['Dividend Yield (%)'] > 5) & 
                (valid_data['Dividend Yield (%)'] < 50) &  # Sanity check
                (valid_data['Earnings Growth (%)'] > 0)
            ]['Ticker'].tolist(),
            'Dividend Yield above 10% and Positive Earnings Growth': valid_data[
                (valid_data['Dividend Yield (%)'] > 10) & 
                (valid_data['Dividend Yield (%)'] < 50) &  # Sanity check
                (valid_data['Earnings Growth (%)'] > 0)
            ]['Ticker'].tolist(),
            'Dividend Yield above 15% and Positive Earnings Growth': valid_data[
                (valid_data['Dividend Yield (%)'] > 15) & 
                (valid_data['Dividend Yield (%)'] < 50) &  # Sanity check
                (valid_data['Earnings Growth (%)'] > 0)
            ]['Ticker'].tolist()
        }

    def validate_data(self) -> None:
        """Validate calculated dividend yields"""
        logging.info("Data validation results:")

        # Check for unrealistic dividend yields
        if not self.dividend_yields_df.empty:
            high_yields = self.dividend_yields_df[self.dividend_yields_df['Dividend Yield (%)'] > 50]
            if not high_yields.empty:
                logging.warning(f"Found {len(high_yields)} dividend yields above 50%:")
                for _, row in high_yields.iterrows():
                    logging.warning(f"  {row['Ticker']}: {row['Dividend Yield (%)']:.2f}% on {row['Ex-Dividend Date'].date()}")

        # Check current metrics
        if not self.metrics_df.empty:
            high_current_yields = self.metrics_df[self.metrics_df['Dividend Yield (%)'] > 50]
            if not high_current_yields.empty:
                logging.warning(f"Found {len(high_current_yields)} current dividend yields above 50%:")
                for _, row in high_current_yields.iterrows():
                    logging.warning(f"  {row['Ticker']}: {row['Dividend Yield (%)']:.2f}%")

    def plot_data(self) -> None:
        """Plot data with improved formatting and validation"""
        logging.info('Plotting data')

        # Validate data first
        self.validate_data()

        sns.set(style="whitegrid")
        fig, axes = plt.subplots(6, 1, figsize=(12, 24), sharex=False)
        palette = sns.color_palette(self.palette_name, len(self.tickers))
        color_map = dict(zip(self.tickers, palette))
        sorted_metrics_df = self.metrics_df.sort_values(by='Dividend Yield (%)', ascending=False)

        # Filter out unrealistic yields for plotting
        plot_dividend_df = self.dividend_yields_df[
            (self.dividend_yields_df['Dividend Yield (%)'] <= 50) & 
            (self.dividend_yields_df['Dividend Yield (%)'] >= 0)
        ]

        if not plot_dividend_df.empty:
            sns.lineplot(ax=axes[0], data=plot_dividend_df, x="Ex-Dividend Date", y="Dividend Yield (%)", 
                        hue="Ticker", palette=color_map, marker="o", legend=False)
        axes[0].set_title("Annualized Dividend Yield (%) Over Time")
        axes[0].set_ylabel("Dividend Yield (%)")
        axes[0].axhline(y=5, color='r', linestyle='--', label='5% Dividend Yield')
        axes[0].axhline(y=10, color='b', linestyle='--', label='10% Dividend Yield')
        axes[0].axhline(y=15, color='g', linestyle='--', label='15% Dividend Yield')
        axes[0].grid(True)
        axes[0].set_xlabel(f"Data fetched on: {self.data_fetch_date}")

        if not self.dividend_yields_df.empty:
            sns.lineplot(ax=axes[1], data=self.dividend_yields_df, x="Ex-Dividend Date", y="Dividends", 
                        hue="Ticker", palette=color_map, marker="o", legend=False)
        axes[1].set_title("Individual Dividend Payments Over Time")
        axes[1].set_ylabel("Dividends")
        axes[1].grid(True)
        axes[1].set_xlabel(f"Data fetched on: {self.data_fetch_date}")

        # Add annotations for dividend amounts
        for _, row in self.dividend_yields_df.iterrows():
            if pd.notna(row['Dividends']):
                axes[1].annotate(f"{row['Dividends']:.2f}",
                                (row['Ex-Dividend Date'], row['Dividends']),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha='center',
                                fontsize=8)

        # Current dividend yield bar chart
        valid_metrics = sorted_metrics_df[
            (sorted_metrics_df['Dividend Yield (%)'] <= 50) & 
            (sorted_metrics_df['Dividend Yield (%)'] >= 0)
        ]

        if not valid_metrics.empty:
            sns.barplot(ax=axes[2], data=valid_metrics, x='Ticker', y='Dividend Yield (%)', 
                       hue='Ticker', palette=color_map, legend=False)
        axes[2].set_title("Current Dividend Yield (%) by Ticker")
        axes[2].set_ylabel("Dividend Yield (%)")
        axes[2].set_xlabel(f"Data fetched on: {self.data_fetch_date}")
        axes[2].grid(True)

        # Add value labels on bars
        for container in axes[2].containers:
            axes[2].bar_label(container, fmt='%.1f%%')

        # Continue with other plots...
        valid_payout = sorted_metrics_df[
            (sorted_metrics_df['Payout Ratio (%)'] <= 200) & 
            (sorted_metrics_df['Payout Ratio (%)'] >= 0)
        ]

        if not valid_payout.empty:
            sns.barplot(ax=axes[3], data=valid_payout, x='Ticker', y='Payout Ratio (%)', 
                       hue='Ticker', palette=color_map, legend=False)
        axes[3].set_title("Payout Ratio (%) by Ticker")
        axes[3].set_ylabel("Payout Ratio (%)")
        axes[3].set_xlabel(f"Data fetched on: {self.data_fetch_date}")
        axes[3].grid(True)

        for container in axes[3].containers:
            axes[3].bar_label(container, fmt='%.1f%%')

        sns.barplot(ax=axes[4], data=sorted_metrics_df, x='Ticker', y='Free Cash Flow (Billion $)', 
                   hue='Ticker', palette=color_map, legend=False)
        axes[4].set_title("Free Cash Flow (Billion $) by Ticker")
        axes[4].set_ylabel("Free Cash Flow (Billion $)")
        axes[4].set_xlabel(f"Data fetched on: {self.data_fetch_date}")
        axes[4].grid(True)

        for container in axes[4].containers:
            axes[4].bar_label(container, fmt='%.2f')

        sns.barplot(ax=axes[5], data=sorted_metrics_df, x='Ticker', y='Earnings Growth (%)', 
                   hue='Ticker', palette=color_map, legend=False)
        axes[5].set_title("Earnings Growth (%) by Ticker")
        axes[5].set_ylabel("Earnings Growth (%)")
        axes[5].set_xlabel(f"Data fetched on: {self.data_fetch_date}")
        axes[5].grid(True)

        for container in axes[5].containers:
            axes[5].bar_label(container, fmt='%.1f%%')

        for ax in axes:
            plt.setp(ax.get_xticklabels(), rotation=45)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 1), title="Ticker")

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    def get_scenarios(self) -> Dict[str, List[str]]:
        return self.scenarios

    def print_summary(self) -> None:
        """Print summary of analysis results"""
        print(f"\n=== WIG Company Dividend Analysis Summary ===")
        print(f"Data fetched on: {self.data_fetch_date}")
        print(f"Companies analyzed: {len(self.tickers)}")

        if not self.metrics_df.empty:
            print(f"\nCurrent Dividend Yields:")
            for _, row in self.metrics_df.sort_values('Dividend Yield (%)', ascending=False).iterrows():
                if pd.notna(row['Dividend Yield (%)']):
                    print(f"  {row['Ticker']}: {row['Dividend Yield (%)']:.2f}%")

        print(f"\nInvestment Scenarios:")
        for scenario, tickers in self.scenarios.items():
            print(f"  {scenario}: {tickers}")

# Example usage:
if __name__ == "__main__":
    # Example WIG companies (you should replace with actual WIG tickers)
    wig_tickers = ["PKN.WA", "PZU.WA", "PKO.WA", "PEO.WA", "LPP.WA"]

    analyzer = WIGCompanyAnalysis(wig_tickers)
    analyzer.fetch_data()
    analyzer.print_summary()
    analyzer.plot_data()
