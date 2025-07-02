
from OptimizationStrategy import DividendStockPortfolio, OptimizationStrategy, PortfolioConstraints
from WIGCompanyAnalysis import WIGCompanyAnalysis

# Selected tickers with dividend yield information
selected_tickers = [
    "KTY.WA",  # Grupa Kęty SA (KTY) - DY: 7.7%
    "BDX.WA",  # Budimex SA (BDX) - DY: 5.9%
    "PZU.WA",  # Powszechny Zakład Ubezpieczeń SA (PZU) - DY: 5.0%
    "DVL.WA",  # Develia SA (DVL) - DY: 8.8%
    "XTB.WA",  # X-Trade Brokers SA (XTB) - DY: 7.5%
    "GPW.WA",  # Giełda Papierów Wartościowych w Warszawie SA (GPW) - DY: 7.0%
    "RBW.WA",  # Rainbow Tours SA (RBW) - DY: 5.3%
    "ELT.WA",  # Elektrotim SA (ELT) - DY: 8.0%
    "PLW.WA",  # PlayWay SA (PLW) - DY: 7.4%
]

# Initialize WIGCompanyAnalysis with selected tickers
wig_analysis = WIGCompanyAnalysis(tickers=selected_tickers)

# Fetch data for analysis
wig_analysis.fetch_data()

# Plot data to visualize the analysis
wig_analysis.plot_data()

# Print summary of the analysis
wig_analysis.print_summary()

# Define available cash
available_cash = 45000  # Example amount in PLN

# Define custom constraints (optional - uses defaults if not specified)
constraints = PortfolioConstraints(
    min_position_size=0.05,    # Minimum 5% position
    max_position_size=0.20,    # Maximum 20% position
    min_yield=0.04,            # Minimum 4% yield
    max_yield=0.15,            # Maximum 15% yield (avoid yield traps)
    max_payout_ratio=0.80,     # Maximum 80% payout ratio
    min_market_cap=1e8,        # Minimum 100M market cap (adjusted for Polish market)
    max_debt_to_equity=2.0     # Maximum debt-to-equity ratio
)

# Initialize DividendStockPortfolio with updated constructor
portfolio = DividendStockPortfolio(
    wig_analysis=wig_analysis, 
    available_cash=available_cash,
    constraints=constraints  # Optional custom constraints
)

# Define optimization strategies to test
strategies_to_test = [
    OptimizationStrategy.BALANCED,
    OptimizationStrategy.YIELD_WEIGHTED,
    OptimizationStrategy.RISK_ADJUSTED,
    OptimizationStrategy.DIVIDEND_GROWTH,
    OptimizationStrategy.EQUAL_WEIGHT
]

# Process each scenario with different optimization strategies
for scenario_key in wig_analysis.get_scenarios():
    print(f"\n{'='*60}")
    print(f"*** SCENARIO: {scenario_key} ***")
    print(f"{'='*60}")

    # Test different optimization strategies for this scenario
    for strategy in strategies_to_test:
        print(f"\n{'-'*50}")
        print(f"Strategy: {strategy.value.upper()}")
        print(f"{'-'*50}")

        try:
            # Select tickers based on the scenario and strategy
            portfolio.select_tickers_with_strategy(
                scenario_key=scenario_key,
                strategy=strategy,
                max_positions=8  # Limit to 8 positions for diversification
            )

            # Check if any tickers were selected
            if not portfolio.selected_tickers:
                print(f"No tickers selected for scenario '{scenario_key}' with strategy '{strategy.value}'")
                continue

            print(f"Selected tickers: {portfolio.selected_tickers}")

            # Optimize the portfolio
            portfolio.optimize_portfolio()

            # Calculate dividend income
            dividend_summary = portfolio.calculate_dividend_income()

            # Get portfolio results
            optimized_portfolio = portfolio.get_portfolio()
            portfolio_stats = portfolio.get_portfolio_stats()
            remaining_cash = portfolio.get_remaining_cash()

            # Display results
            print(f"\nOptimized Portfolio: {optimized_portfolio}")
            print(f"Remaining Cash: PLN {remaining_cash:,.2f}")
            print(f"Portfolio Value: PLN {portfolio_stats.get('total_value', 0):,.2f}")
            print(f"Expected Annual Dividends: PLN {dividend_summary['total_annual_dividends']:,.2f}")
            print(f"Portfolio Dividend Yield: {dividend_summary['portfolio_dividend_yield']:.2f}%")

            # Generate and display comprehensive report
            if optimized_portfolio:  # Only generate report if portfolio exists
                print(f"\n{portfolio.generate_portfolio_report()}")

        except Exception as e:
            print(f"Error processing scenario '{scenario_key}' with strategy '{strategy.value}': {e}")
            continue

    print(f"\n{'='*60}\n")

# Example: Focus on a specific scenario and strategy
print(f"\n{'#'*60}")
print("FOCUSED ANALYSIS - BEST PERFORMING SCENARIO")
print(f"{'#'*60}")

# Get available scenarios
scenarios = wig_analysis.get_scenarios()
if scenarios:
    # Pick the first scenario with tickers
    target_scenario = next((k for k, v in scenarios.items() if v), None)

    if target_scenario:
        print(f"Analyzing scenario: {target_scenario}")

        # Use BALANCED strategy for comprehensive analysis
        portfolio.select_tickers_with_strategy(
            scenario_key=target_scenario,
            strategy=OptimizationStrategy.BALANCED,
            max_positions=6
        )

        portfolio.optimize_portfolio()
        dividend_summary = portfolio.calculate_dividend_income()

        # Display comprehensive results
        print(portfolio.generate_portfolio_report())

        # Display dividend breakdown
        print(f"\nDETAILED DIVIDEND BREAKDOWN:")
        print(f"{'-'*40}")
        for ticker, details in dividend_summary['breakdown'].items():
            print(f"{ticker}: {details['shares']} shares @ {details['dividend_yield']*100:.2f}% "
                  f"→ PLN {details['total_annual_dividends']:,.2f}/year")
