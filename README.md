# Stacey Burke Forex Scanner 📊

A Python-based scanner that implements Stacey Burke's mechanical forex trading methodology for identifying FRD (First Red Day) and FGD (First Green Day) patterns.

## 🎯 Features

- **Multi-Pair Scanning**: Analyzes all major forex pairs (EUR/USD, GBP/USD, AUD/USD, USD/JPY, NZD/USD, USD/CAD, USD/CHF)
- **Pattern Recognition**: Identifies FRD/FGD patterns with consecutive day counts
- **H1/D1 Alignment**: Checks alignment between hourly and daily timeframes
- **Setup Recommendations**: Provides entry zones and stop loss levels
- **Session Context**: Considers optimal trading sessions for Manila timezone
- **MT5 Integration**: Direct connection to MetaTrader 5 (Windows only)

## 📁 Project Structure

```
SBStrat/
├── sb_scanner.py           # Basic Stacey Burke scanner (MVP)
├── sb_scanner_v2.py        # Enhanced scanner with H1/D1 alignment
├── test_real_scenario.py   # Test with your actual trading scenario
├── mt5_integration_guide.py  # CSV export method (cross-platform)
├── mt5_live_scanner.py     # Direct MT5 connection (Windows only)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Option 1: CSV Export Method (Cross-Platform)

1. Export data from MT5:
   - Open chart → Right-click → Save As → CSV
   - Save as `{SYMBOL}_H1.csv` (e.g., `EURUSD_H1.csv`)

2. Run the scanner:
   ```bash
   pip install -r requirements.txt
   python3 mt5_integration_guide.py
   ```

### Option 2: Direct MT5 API (Windows Only)

1. Install MetaTrader5 Python library:
   ```bash
   pip install MetaTrader5
   ```

2. Update your credentials in `mt5_live_scanner.py`:
   ```python
   self.account = YOUR_ACCOUNT_NUMBER
   self.password = "YOUR_PASSWORD"
   self.server = "YOUR_BROKER_SERVER"
   ```

3. Run live scanner:
   ```bash
   python mt5_live_scanner.py
   ```

## 📊 Stacey Burke Methodology

### Core Concepts

The market only does 3 things:
1. **Breakout and Trend** - Price breaks out and continues
2. **Breakout and Reverse** - False breakout leading to reversal
3. **Trading Range** - Price consolidates within bounds

### Pattern Definitions

- **FRD (First Red Day)**: First bearish candle after 3+ consecutive bullish candles
  - **Next Day Action**: Look for shorts at previous day's high

- **FGD (First Green Day)**: First bullish candle after 3+ consecutive bearish candles
  - **Next Day Action**: Look for longs at previous day's low

- **3DL (3 Days of Longs)**: 3 consecutive days of buying pressure
- **3DS (3 Days of Shorts)**: 3 consecutive days of selling pressure

### Trading Sessions (Manila Time)

- **2:00-4:00 PM PHT**: London Open
- **8:00 PM-12:00 AM PHT**: NY Overlap (Prime time)
- **12:00-2:00 PM PHT**: Late NY Session

## 🛠️ Dependencies

```txt
pandas==2.0.3
numpy==1.24.3
yfinance==0.2.18
plotly==5.15.0
requests==2.31.0
```

For Windows MT5 integration:
```txt
MetaTrader5
```

## 📈 Example Output

```
============================================================
🎯 STACEY BURKE SCANNER - EURUSD
============================================================
🚨 SETUP: FRD_CONFIRMED
🎯 ACTION: LOOK_FOR_SHORTS
📍 Entry Zone: 1.09200
⛔ Stop Loss: Above 1.09746
💪 Strength: NORMAL
📈 Consecutive Days: 4
```

## 🔄 Continuous Development

This is a work in progress. Planned features:
- [ ] Alert system (email/Telegram)
- [ ] Web dashboard
- [ ] Backtesting module
- [ ] Risk management calculator
- [ ] Multi-broker support

## ⚠️ Disclaimer

This software is for educational purposes only. Forex trading involves substantial risk of loss. Always use proper risk management and never trade with money you cannot afford to lose.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For questions or support regarding the Stacey Burke methodology or this scanner, please refer to the original trading materials and documentation.

---

Built with ❤️ for the trading community