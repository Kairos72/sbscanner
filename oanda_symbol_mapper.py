"""
OANDA Symbol Mapper
===================

Handles OANDA symbol naming conventions.
OANDA adds suffixes to symbols: .sml for standard, .lot for mini lots
"""

def get_oanda_symbol(base_symbol):
    """Convert standard symbol to OANDA format"""
    # EURUSD, GBPUSD, USDJPY, and AUDUSD use .sml suffix
    sml_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

    if base_symbol in sml_symbols:
        return base_symbol + '.sml'
    else:
        return base_symbol

def test_oanda_symbols():
    """Test various symbols with OANDA format"""
    common_pairs = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
        'USDCAD', 'AUDUSD', 'NZDUSD',
        'EURGBP', 'EURJPY', 'GBPJPY',
        'AUDJPY', 'NZDJPY', 'EURCHF'
    ]

    print("OANDA Symbol Mapping:")
    print("-" * 30)
    for symbol in common_pairs:
        oanda_symbol = get_oanda_symbol(symbol)
        print(f"{symbol:8} -> {oanda_symbol}")

if __name__ == "__main__":
    test_oanda_symbols()