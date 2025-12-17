# ACB Enhanced Scanner Implementation Plan

## Phase 1: Core ACB Detection

### 1.1 ACB Level Detection Module
```python
class ACBDetector:
    def identify_acb_levels(self, df):
        """
        Identify levels that broke and never returned
        - Find significant breakouts
        - Validate with time-distance rules (>24 hours)
        - Confirm no retests occurred
        - Mark as ACB (Ain't Coming Back)
        """
```

### 1.2 DMR Level Calculator
```python
class DMRLevelCalculator:
    def calculate_dmr_levels(self, df):
        """
        Calculate all active DMR levels
        - Previous Day High/Low (PDH/PDL)
        - 3-Day High/Low
        - Weekly High/Low
        - Return dictionary with levels and strength
        """
```

### 1.3 Session Analysis Engine
```python
class SessionAnalyzer:
    def identify_sessions(self, candle_time):
        """
        Determine session for each candle
        - Asian: 00:00-06:00 UTC
        - London: 06:00-14:00 UTC
        - NY: 13:00-22:00 UTC
        """

    def analyze_session_behavior(self, df):
        """
        Identify session-specific patterns
        - Asian: Range building
        - London: Breakout/manipulation
        - NY: Confirmation/distribution
        """
```

## Phase 2: Pattern Recognition

### 2.1 Enhanced FRD/FGD with ACB Context
```python
def detect_frd_fgd_with_acb(self, daily_df, dmr_levels, acb_levels):
    """
    Enhanced FRD/FGD detection
    - Traditional FRD/FGD rules
    - Check proximity to DMR levels
    - Validate with session context
    - Grade signals (A+ or A)
    - Identify manipulation phase
    """
```

### 2.2 Pump & Dump Pattern Detector
```python
class PumpDumpDetector:
    def detect_equilibrium_pd(self, df):
        """Price pushes to DMR level and reverses"""

    def detect_m_top_w_bottom(self, df):
        """M-formation for tops, W-formation for bottoms"""

    def detect_lower_range_pd(self, df):
        """Extreme sell-off followed by violent reversal"""

    def detect_entry_phase_pd(self, df):
        """Clever H&S, running trend continuation"""

    def detect_coil_spring(self, df):
        """Compression phase before explosive breakout"""
```

## Phase 3: Smart Money Manipulation Analysis

### 3.1 Liquidity Hunt Detector
```python
def detect_liquidity_hunt(self, df, dmr_levels):
    """
    Identify liquidity hunting
    - Asian low hunts
    - Stop runs below key levels
    - Volume spike analysis
    - Wicking patterns
    """
```

### 3.2 Market Phase Identifier
```python
def identify_market_phase(self, df):
    """
    Determine current market phase
    - ACCUMULATION: Smart money building positions
    - MANIPULATION: Stop hunting, false breakouts
    - DISTRIBUTION: Smart money exiting positions
    - ROTATION: Return to DMR levels
    """
```

## Phase 4: Enhanced Signal Generation

### 4.1 ACB-Aware Signal Generator
```python
def generate_acb_signals(self, df):
    """
    Generate comprehensive trading signals
    - FRD/FGD with ACB validation
    - Pump & Dump pattern triggers
    - Session-specific opportunities
    - Confidence scoring (A+, A, B+)
    """
```

### 4.2 5-Star Setup Prioritizer
```python
def prioritize_setups(self, signals):
    """
    Prioritize high-probability setups:
    1. Daily Equilibrium P&D
    2. Inside Day B/L
    3. Daily M-Top/W-Bottom P&D
    4. 3-Day Market Cycle (FRD/FGD)
    5. Coil/Spring compression

    Ranking criteria:
    - Pattern quality
    - DMR level proximity
    - Session alignment
    - Market structure context
    """
```

## Phase 5: Enhanced Trade Management

### 5.1 Dynamic Stop Loss Calculator
```python
def calculate_acb_stop(self, entry_price, pattern_type, recent_low):
    """
    Calculate optimal stop placement:
    - Liquidity hunt: Below manipulation zone
    - ACB breakout: At ACB level
    - Standard: ATR-based (2x)
    - Always outside manipulation zones
    """
```

### 5.2 DMR-Aware Target Calculator
```python
def calculate_dmr_targets(self, entry_price, direction):
    """
    Calculate profit targets using DMR levels:
    1. Nearest DMR (PDH/PDL)
    2. 3-Day DMR level
    3. Weekly DMR level
    4. Next ACB level
    """
```

## Phase 6: Visualization & Alerts

### 6.1 ACB Chart Markers
```python
def plot_acb_markers(self, ax, df):
    """
    Visual enhancement functions:
    - Mark ACB levels (red)
    - Mark DMR levels (blue)
    - Show session boundaries
    - Highlight manipulation zones
    - Mark entry/exit areas
    """
```

### 6.2 Smart Alert System
```python
def send_acb_alerts(self):
    """
    Context-aware alerts:
    - "Asian low hunt in progress"
    - "FGD confirmed, waiting for liquidity hunt"
    - "5-Star setup at DMR level"
    - "ACB level breach detected"
    - "DMR rotation approaching"
    """
```

## Phase 7: Market Structure Analysis

### 7.1 Smart Money Position Tracker
```python
def track_smart_money(self, df):
    """
    Track smart money activity:
    - Accumulation zones
    - Distribution zones
    - Protected levels
    - Trap patterns
    """
```

### 7.2 Time-Price Relationship Analyzer
```python
def analyze_time_price(self, df):
    """
    Analyze time-price patterns:
    - 60/40 rule validation
    - Session-specific behaviors
    - Time-based manipulations
    - Optimal entry timing windows
    """
```

## Phase 8: Integration & Optimization

### 8.1 Unified ACB Scanner
```python
class EnhancedACBScanner(EnhancedStaceyBurkeScanner):
    def __init__(self):
        super().__init__()
        self.acb_detector = ACBDetector()
        self.dmr_calculator = DMRLevelCalculator()
        self.session_analyzer = SessionAnalyzer()
        self.pd_detector = PumpDumpDetector()

    def scan_markets_acb(self):
        """
        Main scanning function
        - Run all detectors
        - Correlate signals
        - Generate unified analysis
        - Return actionable insights
        """
```

### 8.2 Backtesting & Validation
```python
def backtest_acb_strategy(self, historical_data):
    """
    Backtest ACB-enhanced strategy
    - Validate ACB level success rate
    - Test pattern accuracy
    - Optimize parameters
    - Calculate expectancy by pattern
    - Compare with traditional FRD/FGD
    """
```

## Implementation Priority

1. **Phase 1** (Core ACB) - Foundation level
2. **Phase 4** (Signal Generation) - Immediate trading value
3. **Phase 5** (Trade Management) - Practical application
4. **Phase 2** (Patterns) - Advanced setup recognition
5. **Phase 3** (Manipulation) - Deep market insights
6. **Phase 6** (Visualization) - User interface improvements
7. **Phase 7** (Market Structure) - Expert level analysis
8. **Phase 8** (Integration) - Final optimization

## Key Design Principles

- **Modular Architecture**: Each component independent
- **Configuration Parameters**: Easy adjustment of rules
- **Educational Output**: Explain why signals trigger
- **Actionable Insights**: Clear entry/exit rules
- **Adaptive Learning**: Improve from market behavior
- **Performance Metrics**: Track success rates
- **Risk Management**: Built-in safety mechanisms

## Success Metrics

- Signal accuracy > 70%
- ACB level validation > 90%
- Risk/Reward improvement > 2.0
- Drawdown reduction > 30%
- Win rate improvement > 15%

## File Structure

```
sbscanner/
├── enhanced_sb_scanner.py (existing)
├── acb_enhanced_scanner.py (new)
├── acb/
│   ├── __init__.py
│   ├── detector.py
│   ├── dmr_calculator.py
│   ├── session_analyzer.py
│   ├── patterns/
│   │   ├── pump_dump.py
│   │   ├── frd_fgd.py
│   │   └── five_star.py
│   ├── management/
│   │   ├── stops.py
│   │   └── targets.py
│   └── visualization/
│       ├── charts.py
│       └── alerts.py
└── backtest/
    ├── validator.py
    └── metrics.py
```

## Next Steps

1. Create base ACB detector
2. Implement DMR calculator
3. Add session analysis
4. Enhance FRD/FGD with ACB context
5. Test with historical data
6. Iterate and optimize