# Windows Setup Instructions for MT5 Integration

## 🔧 Requirements

1. Windows 10 or 11
2. Python 3.8+ installed from [python.org](https://python.org)
3. MetaTrader 5 installed with your ACG Markets account

## 📋 Setup Steps

### 1. Install Python on Windows

1. Download Python from [python.org](https://python.org)
2. During installation:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install for all users"

3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

### 2. Install Required Packages

Open Command Prompt or PowerShell and run:

```cmd
# Install all required packages
pip install pandas numpy plotly requests

# Install MetaTrader5 library
pip install MetaTrader5
```

### 3. Configure MT5

1. Open MetaTrader 5
2. Login to your ACG Markets demo account:
   - Account: 2428164
   - Password: 3b6%StTrQj
   - Server: ACGMarkets-Main

3. Enable automated trading in MT5:
   - Tools → Options → Expert Advisors
   - ✅ Check "Allow automated trading"

### 4. Run the Live Scanner

1. Pull the repository:
   ```cmd
   git clone https://github.com/YOUR_USERNAME/SBStrat.git
   cd SBStrat
   ```

2. Run the scanner:
   ```cmd
   python mt5_live_scanner.py
   ```

## 🚀 Features on Windows

- Real-time data from MT5
- No rate limits
- All 7 major pairs
- H1/D1 alignment checking
- Continuous scanning mode

## 🔍 Testing the Connection

To test if MT5 is properly connected:

```python
import MetaTrader5 as mt5

# Initialize MT5
if mt5.initialize():
    print("MT5 initialized successfully")

    # Get terminal info
    terminal_info = mt5.terminal_info()
    print(f"Terminal: {terminal_info.name}")
    print(f"Build: {terminal_info.build}")

    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"Account: {account_info.login}")
        print(f"Balance: {account_info.balance}")

    mt5.shutdown()
else:
    print("Failed to initialize MT5")
```

## ⚠️ Troubleshooting

### Common Issues:

1. **"Cannot import MetaTrader5"**
   - Solution: Reinstall the package: `pip install --force-reinstall MetaTrader5`

2. **"MT5 initialize failed"**
   - Solution: Make sure MT5 is running and you're logged in

3. **"Login failed"**
   - Solution: Verify account credentials and server name
   - Check if demo account is still active

4. **"No data for symbol"**
   - Solution: Enable symbols in MT5 Market Watch
   - Right-click → Show all symbols

### Debug Mode:

Add these lines to see detailed error messages:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

If you encounter issues:
1. Check MT5 is running
2. Verify you're logged into the correct account
3. Ensure Python is properly installed
4. Check firewall/antivirus settings

---

Happy scanning! 🎯