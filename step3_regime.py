"""Step 3: 시장 국면 감지기"""
import pandas as pd
import numpy as np

def detect_market_regime(kospi_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    KOSPI 일별 수익률 기반 시장 국면 분류
    - 상승장(bull): 20일 수익률 > 2%
    - 하락장(bear): 20일 수익률 < -2%
    - 횡보장(sideways): 그 외
    """
    returns = kospi_df["종가"].pct_change()
    rolling_ret = returns.rolling(window).sum()

    regime = pd.Series("sideways", index=kospi_df.index)
    regime[rolling_ret > 0.02] = "bull"
    regime[rolling_ret < -0.02] = "bear"
    return regime

# 국면별 전략 활성화 맵
REGIME_STRATEGY_MAP = {
    "bull": {
        "MA_Cross": 1.0,
        "MACD": 1.0,
        "RSI": 0.5,
        "Bollinger": 0.7,
        "Momentum": 1.2,
        "Volume": 1.0,
    },
    "bear": {
        "MA_Cross": 0.7,
        "MACD": 0.8,
        "RSI": 1.2,
        "Bollinger": 1.0,
        "Momentum": 0.5,
        "Volume": 0.8,
    },
    "sideways": {
        "MA_Cross": 0.8,
        "MACD": 0.7,
        "RSI": 1.0,
        "Bollinger": 1.2,
        "Momentum": 0.6,
        "Volume": 0.9,
    },
}

if __name__ == "__main__":
    kospi = pd.read_csv("data/kospi_index.csv", index_col=0, parse_dates=True)
    regime = detect_market_regime(kospi)
    counts = regime.value_counts()
    print("Step 3: 시장 국면 감지 완료")
    for r, c in counts.items():
        print(f"  {r}: {c}일 ({c/len(regime)*100:.1f}%)")
