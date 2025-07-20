# GitHub Copilot Implementation Guide

This guide provides step-by-step instructions for implementing GitHub Copilot in your development workflow for this cryptocurrency trading bot project.

## Quick Start

### 1. Install GitHub Copilot

**VS Code:**
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "GitHub Copilot"
4. Install the extension by GitHub
5. Restart VS Code
6. Sign in to GitHub when prompted

**JetBrains IDEs (PyCharm, IntelliJ):**
1. Go to File → Settings → Plugins
2. Search for "GitHub Copilot"
3. Install and restart IDE
4. Sign in to GitHub when prompted

### 2. Configure Your Development Environment

**Create workspace settings for VS Code:**
```bash
mkdir -p .vscode
```

Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.analysis.extraPaths": ["./paper_trader"],
  "github.copilot.enable": {
    "*": true,
    "python": true,
    "yaml": true,
    "markdown": true
  },
  "github.copilot.inlineSuggest.enable": true,
  "github.copilot.suggestions.count": 3
}
```

**Set up Python environment:**
```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\\Scripts\\activate

# Linux/Mac  
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_paper_trader.txt
```

### 3. Configure Environment Variables

Create `.env` file in the project root:
```env
# Copy from .env example and fill in your values
BITVAVO_API_KEY=your_api_key_here
BITVAVO_API_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Using Copilot Effectively

### 1. Writing Descriptive Comments

**Good:** Comments that give Copilot context
```python
# Calculate RSI (Relative Strength Index) for the last 14 periods
def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
```

**Better:** Include expected behavior
```python
# Calculate RSI indicator for momentum analysis
# Returns value between 0-100 where >70 indicates overbought, <30 oversold
def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
```

### 2. Project-Specific Prompting

**For Trading Logic:**
```python
# Implement position sizing based on Kelly Criterion and risk tolerance
def calculate_position_size(self, signal_confidence: float, portfolio_value: float) -> float:
```

**For Data Processing:**
```python
# Process 1-minute OHLCV candles and calculate technical indicators
async def process_market_data(self, symbol: str, candle_data: dict) -> dict:
```

**For Risk Management:**
```python
# Check if opening new position violates risk management rules
# Consider max positions, correlation, and portfolio heat
def validate_new_position(self, symbol: str, side: str, size: float) -> bool:
```

### 3. Common Patterns in This Project

**Async Error Handling:**
```python
try:
    # Copilot will suggest appropriate async trading logic here
    pass
except Exception as e:
    self.logger.error(f"Trading operation failed: {e}")
    await self.telegram_notifier.send_error(f"Error in {symbol}: {str(e)}")
    return False
```

**Configuration Access:**
```python
# Access trading parameters from settings
max_position_size = self.settings.max_position_size
stop_loss_percentage = self.settings.stop_loss_pct
```

**Feature Engineering:**
```python
# Calculate multiple technical indicators for ML model features
features = {}
features['rsi'] = ta.rsi(df['close'], length=14)
features['macd'] = ta.macd(df['close'])['MACD_12_26_9']
# Copilot will suggest more indicators based on this pattern
```

## Advanced Usage

### 1. Testing with Copilot

Start test functions with descriptive names:
```python
def test_signal_generator_with_strong_bullish_pattern():
    """Test signal generation when RSI < 30 and MACD crosses above signal line."""
    # Copilot will suggest appropriate test setup and assertions
```

### 2. Docstring-Driven Development

Write docstrings first, let Copilot implement:
```python
def backtest_strategy(self, start_date: str, end_date: str, initial_capital: float) -> dict:
    """
    Backtest the trading strategy over a specified period.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format  
        initial_capital: Starting portfolio value in EUR
        
    Returns:
        dict: Backtest results including total return, Sharpe ratio, max drawdown
    """
    # Copilot will suggest the implementation based on the docstring
```

### 3. Code Completion Shortcuts

- **Tab**: Accept current suggestion
- **Esc**: Dismiss suggestion
- **Alt + ]**: Next suggestion
- **Alt + [**: Previous suggestion
- **Ctrl + Enter**: Open Copilot suggestions panel

## Troubleshooting

### Common Issues

**Copilot not suggesting relevant code:**
- Add more context in comments
- Include relevant imports at the top
- Use descriptive variable names
- Specify the trading domain in comments

**Suggestions don't match project structure:**
- Ensure proper imports are present
- Check that class and method names follow project conventions
- Verify the file is in the correct directory

**AsyncIO patterns not working:**
- Include `async`/`await` keywords in comments
- Import asyncio at the top of the file
- Use async function definitions

### Performance Tips

1. **Write comments before code** - Gives Copilot context
2. **Use type hints** - Improves suggestion accuracy  
3. **Include imports** - Helps Copilot understand available libraries
4. **Be specific** - "Calculate RSI" is better than "Calculate indicator"

## Security Considerations

### What to Watch For

- **Never commit API keys** - Review suggestions for hardcoded secrets
- **Validate external data** - Don't trust Copilot suggestions for input validation
- **Review financial calculations** - Double-check math in trading logic
- **Test error handling** - Ensure graceful failure in trading operations

### Best Practices

1. Use environment variables for sensitive data
2. Add input validation to all trading functions
3. Include comprehensive error handling
4. Log all trading decisions for audit trails
5. Test with paper trading before live implementation

## Integration with Existing Code

### Working with the Paper Trader

When modifying existing components:

**Data Collection (`paper_trader/data/`):**
```python
# Extend BitvavoDataCollector to support new timeframes
class ExtendedDataCollector(BitvavoDataCollector):
```

**Strategy Development (`paper_trader/strategy/`):**
```python
# Create new trading strategy inheriting from base strategy
class MyCustomStrategy(BaseStrategy):
```

**Model Integration (`paper_trader/models/`):**
```python
# Add new ML model for price prediction
class TransformerPricePredictor:
```

## Getting Help

- **GitHub Copilot Docs**: https://docs.github.com/en/copilot
- **VS Code Copilot Guide**: https://code.visualstudio.com/docs/editor/github-copilot
- **Project Documentation**: See README files in this repository
- **Community**: GitHub Copilot Community Discussions

Remember: Copilot is a tool to enhance your productivity, not replace your understanding of trading systems and risk management. Always review and test generated code thoroughly.