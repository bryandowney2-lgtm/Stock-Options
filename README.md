# Options Scanner

A directional options screener (calls + puts) that pulls live option chains via
[yfinance](https://github.com/ranaroussi/yfinance) and ranks contracts by a
blended score.

> **This is a research aid, not financial advice.** A high score means a contract
> with high leverage that is actually tradeable — **not** one likely to profit.
> Options can expire worthless. Verify all quotes against your broker before trading.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python options_scanner.py AAPL MSFT NVDA
python options_scanner.py SPY  --side call --max-dte 30
python options_scanner.py TSLA --side put  --min-dte 7 --max-dte 21 --top 15
```

### Options

| Flag        | Default | Description                      |
|-------------|---------|----------------------------------|
| `--side`    | `both`  | `call`, `put`, or `both`         |
| `--min-dte` | `7`     | Minimum days to expiry           |
| `--max-dte` | `45`    | Maximum days to expiry           |
| `--top`     | `10`    | How many candidates to display   |

## How scoring works

Each contract is scored on five normalized factors:

- **Leverage** (30%) — delta-adjusted bang per dollar
- **Liquidity** (25%) — volume, open interest, spread tightness
- **Unusual volume** (15%) — volume relative to open interest
- **IV value** (15%) — implied vol vs realized vol
- **Moneyness** (15%) — convexity sweet spot (~2–8% OTM)

Edit the `WEIGHTS` dict in `options_scanner.py` to change the blend.

## License

MIT
