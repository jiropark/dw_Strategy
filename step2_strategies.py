"""Step 2: 전략 구현 - 6개 트레이딩 전략"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.weight = 1.0 / 6  # 균등 가중치

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """매수=1, 매도=-1, 관망=0 시그널 반환"""
        pass

class MAStrategy(BaseStrategy):
    """이동평균 교차 전략: 5일/20일"""
    def __init__(self):
        super().__init__("MA_Cross")

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        ma5 = df["종가"].rolling(5).mean()
        ma20 = df["종가"].rolling(20).mean()
        signal = pd.Series(0, index=df.index)
        signal[ma5 > ma20] = 1   # 골든크로스
        signal[ma5 < ma20] = -1  # 데드크로스
        return signal

class MACDStrategy(BaseStrategy):
    """MACD 전략: 12/26/9"""
    def __init__(self):
        super().__init__("MACD")

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        ema12 = df["종가"].ewm(span=12).mean()
        ema26 = df["종가"].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        signal = pd.Series(0, index=df.index)
        signal[macd > signal_line] = 1
        signal[macd < signal_line] = -1
        return signal

class RSIStrategy(BaseStrategy):
    """RSI 전략: 14일, 기준 30/70"""
    def __init__(self):
        super().__init__("RSI")

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        delta = df["종가"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        signal = pd.Series(0, index=df.index)
        signal[rsi < 30] = 1   # 과매도 → 매수
        signal[rsi > 70] = -1  # 과매수 → 매도
        return signal

class BollingerStrategy(BaseStrategy):
    """볼린저밴드 전략: 20일, 2표준편차"""
    def __init__(self):
        super().__init__("Bollinger")

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        ma20 = df["종가"].rolling(20).mean()
        std20 = df["종가"].rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        signal = pd.Series(0, index=df.index)
        signal[df["종가"] < lower] = 1   # 하단 이탈 → 매수
        signal[df["종가"] > upper] = -1  # 상단 이탈 → 매도
        return signal

class MomentumStrategy(BaseStrategy):
    """모멘텀 전략: 5일/20일 수익률"""
    def __init__(self):
        super().__init__("Momentum")

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        mom5 = df["종가"].pct_change(5)
        mom20 = df["종가"].pct_change(20)
        signal = pd.Series(0, index=df.index)
        signal[(mom5 > 0) & (mom20 > 0)] = 1   # 단기+장기 상승
        signal[(mom5 < 0) & (mom20 < 0)] = -1  # 단기+장기 하락
        return signal

class VolumeStrategy(BaseStrategy):
    """거래량 전략: 20일 평균 150% 이상"""
    def __init__(self):
        super().__init__("Volume")

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        vol_ma20 = df["거래량"].rolling(20).mean()
        vol_ratio = df["거래량"] / vol_ma20
        price_change = df["종가"].pct_change()
        signal = pd.Series(0, index=df.index)
        # 거래량 폭증 + 가격 상승 → 매수
        signal[(vol_ratio > 1.5) & (price_change > 0)] = 1
        # 거래량 폭증 + 가격 하락 → 매도
        signal[(vol_ratio > 1.5) & (price_change < 0)] = -1
        return signal

def get_all_strategies():
    return [
        MAStrategy(),
        MACDStrategy(),
        RSIStrategy(),
        BollingerStrategy(),
        MomentumStrategy(),
        VolumeStrategy(),
    ]

if __name__ == "__main__":
    strategies = get_all_strategies()
    print("Step 2: 전략 구현 완료")
    for s in strategies:
        print(f"  {s.name}: 초기 가중치 = {s.weight:.4f}")
