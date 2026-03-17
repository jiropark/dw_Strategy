"""Step 5: 백테스터"""
import pandas as pd
import numpy as np

TRADING_COST = 0.003  # 0.3% (수수료 + 슬리피지)
HOLDING_DAYS = 3      # 3거래일
TARGET_RETURN = 0.003 # 0.3% 목표

def calculate_3day_return(df: pd.DataFrame, entry_idx: int) -> float:
    """3거래일 누적 수익률 계산 (거래비용 반영)"""
    if entry_idx + HOLDING_DAYS >= len(df):
        return None
    entry_price = df.iloc[entry_idx]["종가"]
    exit_price = df.iloc[entry_idx + HOLDING_DAYS]["종가"]
    raw_return = (exit_price - entry_price) / entry_price
    net_return = raw_return - TRADING_COST  # 거래비용 차감
    return net_return

def judge_trade(net_return: float) -> bool:
    """판정: 3거래일 수익률 >= 0.3% → 승(True)"""
    return net_return >= TARGET_RETURN

def update_weights(strategies, results: list, learning_rate: float = 0.02):
    """전략별 가중치 업데이트
    results: list of (strategy_name, is_win) tuples
    최소 가중치 0.03 보장 (전략 소멸 방지)
    """
    MIN_WEIGHT = 0.03
    strategy_map = {s.name: s for s in strategies}

    for name, is_win in results:
        if name in strategy_map:
            s = strategy_map[name]
            if is_win:
                s.weight *= (1 + learning_rate)
            else:
                s.weight *= (1 - learning_rate)

    # 최소 가중치 보장
    for s in strategies:
        s.weight = max(s.weight, MIN_WEIGHT)

    # 가중치 정규화
    total = sum(s.weight for s in strategies)
    if total > 0:
        for s in strategies:
            s.weight /= total

def check_convergence(old_weights: dict, new_weights: dict, threshold: float = 0.01) -> bool:
    """수렴 감지: 모든 전략의 가중치 변화가 threshold 미만이면 수렴"""
    for name in old_weights:
        if abs(old_weights[name] - new_weights.get(name, 0)) >= threshold:
            return False
    return True

if __name__ == "__main__":
    print("Step 5: 백테스터 모듈 완료")
    print(f"  거래비용: {TRADING_COST*100}%")
    print(f"  보유기간: {HOLDING_DAYS}거래일")
    print(f"  목표수익률: {TARGET_RETURN*100}%")
