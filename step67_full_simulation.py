"""Step 6+7: 전체 시뮬레이션 + 500만원 투자 수익 시뮬레이션"""
import sys
import os
import json
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime

# stdout 버퍼 비활성화
sys.stdout.reconfigure(line_buffering=True)

from step2_strategies import get_all_strategies, BaseStrategy
from step3_regime import detect_market_regime, REGIME_STRATEGY_MAP
from step4_risk import RiskManager
from step5_backtest import (
    calculate_3day_return, judge_trade, update_weights,
    check_convergence, HOLDING_DAYS, TRADING_COST, TARGET_RETURN
)

os.makedirs("results", exist_ok=True)
random.seed(42)
np.random.seed(42)

print("=" * 60)
print("Step 6+7: 전체 시뮬레이션 + 500만원 투자")
print("=" * 60)

# === 데이터 로드 ===
split_info = json.load(open("data/split_info.json", encoding="utf-8"))
kospi = pd.read_csv("data/kospi_index.csv", index_col=0, parse_dates=True)
ticker_info = pd.read_csv("data/top100_tickers.csv", encoding="utf-8-sig")
tickers = ticker_info["ticker"].astype(str).str.zfill(6).tolist()
names = ticker_info["name"].tolist()

stock_data = {}
for t in tickers:
    path = f"data/stocks/{t}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > 30:
            stock_data[t] = df

print(f"로드된 종목 수: {len(stock_data)}")
print(f"전체 기간: {split_info['train_start']} ~ {split_info['test_end']}")
print(f"  Train:      {split_info['train_start']} ~ {split_info['train_end']} ({split_info['train_days']}일)")
print(f"  Validation: {split_info['val_start']} ~ {split_info['val_end']} ({split_info['val_days']}일)")
print(f"  Test:       {split_info['test_start']} ~ {split_info['test_end']} ({split_info['test_days']}일)")

# 시장 국면 감지
regime = detect_market_regime(kospi)
regime_counts = regime.value_counts()
print("\n시장 국면 분포:")
for r, c in regime_counts.items():
    print(f"  {r}: {c}일 ({c/len(regime)*100:.1f}%)")

def get_period_data(df, start_str, end_str):
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    mask = (df.index >= start) & (df.index <= end)
    return df[mask]

# ========================================================
# Step 6: 시뮬레이션 루프
# ========================================================
def run_simulation(stock_data, regime, split_info, period="train", max_iterations=10000, strategies=None):
    if period == "train":
        start, end = split_info["train_start"], split_info["train_end"]
    elif period == "val":
        start, end = split_info["val_start"], split_info["val_end"]
    else:
        start, end = split_info["test_start"], split_info["test_end"]

    if strategies is None:
        strategies = get_all_strategies()

    risk_mgr = RiskManager()
    ticker_list = list(stock_data.keys())

    history = {"iteration": [], "win_rate": [], "weights": {s.name: [] for s in strategies}}
    regime_stats = {"bull": {"win": 0, "total": 0}, "bear": {"win": 0, "total": 0}, "sideways": {"win": 0, "total": 0}}

    total_wins = 0
    total_trades = 0
    converged = False
    is_train = (period == "train")
    iterations = max_iterations if is_train else 1

    for iteration in range(1, iterations + 1):
        old_weights = {s.name: s.weight for s in strategies}
        sample_size = min(20, len(ticker_list))
        sampled = random.sample(ticker_list, sample_size)

        for ticker in sampled:
            df = stock_data[ticker]
            period_df = get_period_data(df, start, end)
            if len(period_df) < HOLDING_DAYS + 30:
                continue

            signals = {}
            for strat in strategies:
                try:
                    sig = strat.generate_signal(period_df)
                    signals[strat.name] = sig
                except:
                    signals[strat.name] = pd.Series(0, index=period_df.index)

            # 벡터화된 시그널 결합
            regime_series = regime.reindex(period_df.index).fillna("sideways")
            combined = pd.Series(0.0, index=period_df.index)
            for strat in strategies:
                sig = signals[strat.name].reindex(period_df.index).fillna(0)
                mult = regime_series.map(lambda r: REGIME_STRATEGY_MAP.get(r, {}).get(strat.name, 1.0))
                combined += sig * strat.weight * mult

            buy_dates = combined[combined > 0.3].index.tolist()
            if not buy_dates:
                continue
            if is_train and len(buy_dates) > 5:
                buy_dates = random.sample(buy_dates, 5)

            for buy_date in buy_dates:
                buy_idx = period_df.index.get_loc(buy_date)
                ret = calculate_3day_return(period_df, buy_idx)
                if ret is None:
                    continue
                should_exit, reason = risk_mgr.should_exit(ret)
                if reason == "trading_halted":
                    continue
                is_win = judge_trade(ret)
                total_wins += int(is_win)
                total_trades += 1

                r = regime.loc[buy_date] if buy_date in regime.index else "sideways"
                regime_stats[r]["total"] += 1
                if is_win:
                    regime_stats[r]["win"] += 1

                risk_mgr.update_loss_count(not is_win)

                if is_train:
                    trade_results = []
                    for strat in strategies:
                        sig_val = signals[strat.name].get(buy_date, 0)
                        if sig_val > 0:
                            trade_results.append((strat.name, is_win))
                    if trade_results:
                        update_weights(strategies, trade_results)
            risk_mgr.reset()

        win_rate = total_wins / total_trades if total_trades > 0 else 0
        history["iteration"].append(iteration)
        history["win_rate"].append(win_rate)
        for s in strategies:
            history["weights"][s.name].append(s.weight)

        if is_train and iteration % 500 == 0:
            print(f"\n--- {iteration}회 중간 결과 ---")
            print(f"  누적 달성률: {win_rate*100:.2f}% ({total_wins}/{total_trades})")
            for s in strategies:
                print(f"  {s.name:12s}: {s.weight:.4f}")

        if is_train and iteration > 100:
            new_weights = {s.name: s.weight for s in strategies}
            if check_convergence(old_weights, new_weights, threshold=0.005):
                if iteration > 500:
                    print(f"\n수렴 감지! {iteration}회에서 종료")
                    converged = True
                    break

    final_win_rate = total_wins / total_trades if total_trades > 0 else 0
    return {
        "period": period, "iterations": iteration if is_train else 1,
        "total_trades": total_trades, "total_wins": total_wins,
        "win_rate": final_win_rate, "converged": converged,
        "regime_stats": regime_stats, "history": history,
        "strategies": {s.name: s.weight for s in strategies},
    }

# === Train ===
print("\n" + "=" * 60)
print("Train 시뮬레이션 시작 (최대 10,000회)")
print("=" * 60)

strategies = get_all_strategies()
train_result = run_simulation(stock_data, regime, split_info, "train", 10000, strategies)
optimal_weights = train_result["strategies"]

print(f"\nTrain 결과: 달성률 {train_result['win_rate']*100:.2f}% ({train_result['total_trades']}거래, {train_result['iterations']}회)")
print("최적 가중치:")
for name, w in sorted(optimal_weights.items(), key=lambda x: -x[1]):
    print(f"  {name:12s}: {w:.4f}")

# === Validation ===
print("\n" + "=" * 60)
print("Validation 검증")
print("=" * 60)
val_strategies = get_all_strategies()
for s in val_strategies:
    s.weight = optimal_weights.get(s.name, 1/6)
val_result = run_simulation(stock_data, regime, split_info, "val", strategies=val_strategies)
print(f"Validation: {val_result['win_rate']*100:.2f}% ({val_result['total_trades']}거래)")

# === Test ===
test_result = None
if val_result["win_rate"] >= 0.55:
    print("\nValidation 55%+ → Test 실행")
else:
    print(f"\nValidation {val_result['win_rate']*100:.1f}% < 55% → Test 참고용 실행")

test_strategies = get_all_strategies()
for s in test_strategies:
    s.weight = optimal_weights.get(s.name, 1/6)
test_result = run_simulation(stock_data, regime, split_info, "test", strategies=test_strategies)
print(f"Test: {test_result['win_rate']*100:.2f}% ({test_result['total_trades']}거래)")

# ========================================================
# Step 7: 500만원 투자 수익 시뮬레이션
# ========================================================
print("\n" + "=" * 60)
print("Step 7: 500만원 투자 수익 시뮬레이션")
print("=" * 60)

INITIAL_CAPITAL = 5_000_000  # 500만원

def simulate_investment(stock_data, regime, split_info, strategy_list, use_ensemble=False, ensemble_weights=None):
    """
    실제 투자 시뮬레이션: 매수 시그널 발생 시 투자, 3거래일 후 청산
    전체 기간(train+val+test) 통합 실행
    """
    start = split_info["train_start"]
    end = split_info["test_end"]

    capital = INITIAL_CAPITAL
    capital_history = [(pd.Timestamp(start), capital)]
    monthly_balance = {}
    trades = []
    max_capital = capital
    mdd = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    risk_mgr = RiskManager()

    ticker_list = list(stock_data.keys())

    # 모든 거래일 수집
    all_dates = set()
    for t in ticker_list:
        df = get_period_data(stock_data[t], start, end)
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    i = 0
    while i < len(all_dates):
        current_date = all_dates[i]

        if risk_mgr.is_halted:
            # 3일 대기 후 재개
            risk_mgr.reset()
            i += 3
            continue

        # 각 종목 스캔
        best_signal = None
        best_ticker = None
        best_signal_strength = 0

        for ticker in random.sample(ticker_list, min(30, len(ticker_list))):
            df = stock_data[ticker]
            if current_date not in df.index:
                continue
            loc = df.index.get_loc(current_date)
            if loc < 30 or loc + HOLDING_DAYS >= len(df):
                continue

            # 시그널 계산
            window_df = df.iloc[max(0, loc-50):loc+HOLDING_DAYS+1]
            total_signal = 0

            for strat in strategy_list:
                try:
                    sig = strat.generate_signal(window_df)
                    sig_val = sig.iloc[-HOLDING_DAYS-1] if len(sig) > HOLDING_DAYS else 0

                    r = regime.loc[current_date] if current_date in regime.index else "sideways"
                    regime_mult = REGIME_STRATEGY_MAP.get(r, {}).get(strat.name, 1.0)

                    if use_ensemble and ensemble_weights:
                        w = ensemble_weights.get(strat.name, 1/len(strategy_list))
                    else:
                        w = 1.0 / len(strategy_list)

                    total_signal += sig_val * w * regime_mult
                except:
                    pass

            if total_signal > best_signal_strength:
                best_signal_strength = total_signal
                best_ticker = ticker
                best_signal = total_signal

        # 매수 실행
        if best_ticker and best_signal_strength > 0.3:
            df = stock_data[best_ticker]
            loc = df.index.get_loc(current_date)
            ret = calculate_3day_return(df, loc)

            if ret is not None:
                # 리스크 관리
                if ret <= -0.01:  # stop loss
                    actual_ret = -0.01 - TRADING_COST
                elif ret >= 0.003:  # take profit
                    actual_ret = ret - TRADING_COST
                else:
                    actual_ret = ret

                profit = capital * actual_ret
                capital += profit
                is_win = actual_ret > 0

                trades.append({
                    "date": str(current_date.date()),
                    "ticker": best_ticker,
                    "return": actual_ret,
                    "capital": capital,
                    "is_win": is_win,
                })

                # MDD 계산
                if capital > max_capital:
                    max_capital = capital
                drawdown = (max_capital - capital) / max_capital
                if drawdown > mdd:
                    mdd = drawdown

                # 연속 손실
                if not is_win:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0

                risk_mgr.update_loss_count(not is_win)

                capital_history.append((current_date, capital))

                # 월별 기록
                month_key = current_date.strftime("%Y-%m")
                monthly_balance[month_key] = capital

                # 3거래일 skip
                i += HOLDING_DAYS
                continue

        i += 1

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    win_count = sum(1 for t in trades if t["is_win"])
    win_rate = win_count / len(trades) if trades else 0

    return {
        "final_capital": round(capital),
        "total_profit": round(capital - INITIAL_CAPITAL),
        "total_return_pct": round(total_return * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "mdd": round(mdd * 100, 2),
        "max_consecutive_losses": max_consecutive_losses,
        "total_trades": len(trades),
        "monthly_balance": monthly_balance,
        "capital_history": [(str(d.date()) if hasattr(d, 'date') else str(d), c) for d, c in capital_history],
    }

def simulate_kospi_hold(kospi_df, split_info):
    """KOSPI 단순 보유 수익"""
    start = pd.Timestamp(split_info["train_start"])
    end = pd.Timestamp(split_info["test_end"])
    period = kospi_df[(kospi_df.index >= start) & (kospi_df.index <= end)]
    if len(period) < 2 or "종가" not in period.columns:
        return {"final_capital": INITIAL_CAPITAL, "total_return_pct": 0, "mdd": 0}

    start_price = period["종가"].iloc[0]
    end_price = period["종가"].iloc[-1]
    total_return = (end_price - start_price) / start_price

    # MDD 계산
    cummax = period["종가"].cummax()
    drawdown = (cummax - period["종가"]) / cummax
    mdd = drawdown.max()

    final_capital = INITIAL_CAPITAL * (1 + total_return)

    # 월별
    monthly = {}
    for date, row in period.iterrows():
        mk = date.strftime("%Y-%m")
        ret = (row["종가"] - start_price) / start_price
        monthly[mk] = round(INITIAL_CAPITAL * (1 + ret))

    return {
        "final_capital": round(final_capital),
        "total_profit": round(final_capital - INITIAL_CAPITAL),
        "total_return_pct": round(total_return * 100, 2),
        "mdd": round(mdd * 100, 2),
        "monthly_balance": monthly,
        "total_trades": 0,
        "win_rate": 0,
        "max_consecutive_losses": 0,
    }

# === 각 전략별 투자 시뮬레이션 ===
from step2_strategies import MAStrategy, MACDStrategy, RSIStrategy, BollingerStrategy, MomentumStrategy, VolumeStrategy

strategy_configs = {
    "MA 전략": [MAStrategy()],
    "RSI 전략": [RSIStrategy()],
    "볼린저밴드 전략": [BollingerStrategy()],
    "MACD 전략": [MACDStrategy()],
    "모멘텀 전략": [MomentumStrategy()],
    "거래량 전략": [VolumeStrategy()],
    "최적 앙상블": get_all_strategies(),
}

investment_results = {}

print(f"\n초기 투자금: {INITIAL_CAPITAL:,}원")
print(f"투자 기간: {split_info['train_start']} ~ {split_info['test_end']}\n")

for strat_name, strat_list in strategy_configs.items():
    print(f"  {strat_name} 시뮬레이션 중...", end="", flush=True)
    use_ensemble = (strat_name == "최적 앙상블")
    result = simulate_investment(
        stock_data, regime, split_info, strat_list,
        use_ensemble=use_ensemble,
        ensemble_weights=optimal_weights if use_ensemble else None
    )
    investment_results[strat_name] = result
    print(f" 완료 → {result['final_capital']:,}원 ({result['total_return_pct']:+.2f}%)")

# KOSPI 단순 보유
print(f"  KOSPI 단순보유 계산 중...", end="", flush=True)
kospi_result = simulate_kospi_hold(kospi, split_info)
investment_results["KOSPI 단순보유"] = kospi_result
print(f" 완료 → {kospi_result['final_capital']:,}원 ({kospi_result['total_return_pct']:+.2f}%)")

# ========================================================
# 최종 출력
# ========================================================
print("\n" + "=" * 60)
print("최종 결과")
print("=" * 60)

# 1. 국면별 최적 전략
print("\n[1] 국면별 최적 전략 가중치:")
for r in ["bull", "bear", "sideways"]:
    stats = train_result["regime_stats"][r]
    wr = stats["win"] / stats["total"] * 100 if stats["total"] > 0 else 0
    print(f"  {r}: 달성률 {wr:.1f}% ({stats['win']}/{stats['total']})")
    regime_w = {}
    for name, base_w in optimal_weights.items():
        regime_w[name] = base_w * REGIME_STRATEGY_MAP[r].get(name, 1.0)
    total_rw = sum(regime_w.values())
    for name, rw in sorted(regime_w.items(), key=lambda x: -x[1]):
        print(f"    {name:12s}: {rw/total_rw:.4f}")

# 2. 전체 달성률
print(f"\n[2] 전체 달성률:")
print(f"  Train:      {train_result['win_rate']*100:.2f}%")
print(f"  Validation: {val_result['win_rate']*100:.2f}%")
print(f"  Test:       {test_result['win_rate']*100:.2f}%")

# 3. 500만원 투자 비교표
print(f"\n[3] 500만원 투자 시 전략별 최종 수익 비교:")
print(f"{'전략':<20s} {'최종잔고':>12s} {'수익금':>12s} {'수익률':>8s} {'달성률':>8s} {'MDD':>7s} {'최대연속손실':>8s}")
print("-" * 80)
for sname in ["MA 전략", "MACD 전략", "RSI 전략", "볼린저밴드 전략", "모멘텀 전략", "거래량 전략", "최적 앙상블", "KOSPI 단순보유"]:
    r = investment_results[sname]
    print(f"{sname:<20s} {r['final_capital']:>12,}원 {r['total_profit']:>+12,}원 {r['total_return_pct']:>+7.2f}% {r['win_rate']:>7.1f}% {r['mdd']:>6.2f}% {r['max_consecutive_losses']:>6d}회")

# 4. JSON 저장
result_json = {
    "optimal_weights": optimal_weights,
    "train": {
        "win_rate": train_result["win_rate"],
        "total_trades": train_result["total_trades"],
        "iterations": train_result["iterations"],
        "converged": train_result["converged"],
        "regime_stats": train_result["regime_stats"],
    },
    "validation": {
        "win_rate": val_result["win_rate"],
        "total_trades": val_result["total_trades"],
        "regime_stats": val_result["regime_stats"],
    },
    "test": {
        "win_rate": test_result["win_rate"],
        "total_trades": test_result["total_trades"],
        "regime_stats": test_result["regime_stats"],
    },
    "investment_simulation": {},
    "split_info": split_info,
}
for sname, r in investment_results.items():
    result_json["investment_simulation"][sname] = {
        "final_capital": r["final_capital"],
        "total_profit": r["total_profit"],
        "total_return_pct": r["total_return_pct"],
        "win_rate": r["win_rate"],
        "mdd": r["mdd"],
        "max_consecutive_losses": r["max_consecutive_losses"],
        "total_trades": r.get("total_trades", 0),
        "monthly_balance": r.get("monthly_balance", {}),
    }

with open("results/simulation_result.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)
print(f"\n[4] 결과 저장: results/simulation_result.json")

# 5. HTML 리포트
print(f"\n[5] HTML 리포트 생성 중...")

history = train_result["history"]

# 월별 잔고 데이터 준비
all_months = sorted(set().union(*[set(r.get("monthly_balance", {}).keys()) for r in investment_results.values()]))

html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Trading Simulation Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body { font-family: 'Segoe UI', sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 40px; }
.card { background: white; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.metric { display: inline-block; text-align: center; padding: 15px 25px; margin: 8px; background: #ecf0f1; border-radius: 8px; min-width: 120px; }
.metric .value { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
.metric .label { color: #7f8c8d; font-size: 0.85em; }
.metric.green .value { color: #27ae60; }
.metric.red .value { color: #e74c3c; }
.chart { width: 100%%; height: 450px; }
table { width: 100%%; border-collapse: collapse; margin: 15px 0; }
th, td { padding: 10px 12px; text-align: center; border: 1px solid #ddd; }
th { background: #3498db; color: white; }
tr:nth-child(even) { background: #f9f9f9; }
.highlight { background: #ffffcc !important; font-weight: bold; }
</style>
</head><body>
<h1>Trading Simulation Report</h1>
<p>기간: """ + split_info["train_start"] + " ~ " + split_info["test_end"] + "</p>"

# 요약 카드
html += '<div class="card"><h2>Performance Summary</h2>'
html += '<div class="metric"><div class="value">{:.1f}%</div><div class="label">Train 달성률</div></div>'.format(train_result["win_rate"]*100)
html += '<div class="metric"><div class="value">{:.1f}%</div><div class="label">Validation</div></div>'.format(val_result["win_rate"]*100)
html += '<div class="metric"><div class="value">{:.1f}%</div><div class="label">Test</div></div>'.format(test_result["win_rate"]*100)
html += '<div class="metric"><div class="value">{}</div><div class="label">학습 횟수</div></div>'.format(train_result["iterations"])

best_strat = max([(k,v) for k,v in investment_results.items() if k != "KOSPI 단순보유"], key=lambda x: x[1]["final_capital"])
kospi_r = investment_results["KOSPI 단순보유"]
css_class = "green" if best_strat[1]["total_return_pct"] > 0 else "red"
html += f'<div class="metric {css_class}"><div class="value">{best_strat[1]["final_capital"]:,}원</div><div class="label">최적전략 최종잔고</div></div>'
html += f'<div class="metric"><div class="value">{kospi_r["final_capital"]:,}원</div><div class="label">KOSPI 보유 잔고</div></div>'
html += '</div>'

# 500만원 투자 비교표
html += '<div class="card"><h2>500만원 투자 전략별 수익 비교</h2>'
html += '<table><tr><th>전략</th><th>최종잔고</th><th>수익금</th><th>수익률</th><th>달성률</th><th>MDD</th><th>최대연속손실</th><th>거래수</th></tr>'
for sname in ["MA 전략", "MACD 전략", "RSI 전략", "볼린저밴드 전략", "모멘텀 전략", "거래량 전략", "최적 앙상블", "KOSPI 단순보유"]:
    r = investment_results[sname]
    cls = ' class="highlight"' if sname == best_strat[0] else ""
    profit_color = "color:#27ae60" if r["total_profit"] >= 0 else "color:#e74c3c"
    html += f'<tr{cls}><td>{sname}</td><td>{r["final_capital"]:,}원</td>'
    html += f'<td style="{profit_color}">{r["total_profit"]:+,}원</td>'
    html += f'<td style="{profit_color}">{r["total_return_pct"]:+.2f}%</td>'
    html += f'<td>{r["win_rate"]:.1f}%</td><td>{r["mdd"]:.2f}%</td>'
    html += f'<td>{r["max_consecutive_losses"]}회</td><td>{r.get("total_trades",0)}</td></tr>'
html += '</table></div>'

# 차트 1: 회차별 달성률
iters = history["iteration"]
win_rates = [w * 100 for w in history["win_rate"]]
step = max(1, len(iters) // 500)
html += """
<div class="card"><h2>Train 회차별 달성률 추이</h2>
<div id="chart1" class="chart"></div>
<script>
Plotly.newPlot('chart1', [{{x:{x},y:{y},type:'scatter',mode:'lines',name:'Win Rate',line:{{color:'#3498db',width:2}}}}],
{{xaxis:{{title:'Iteration'}},yaxis:{{title:'Win Rate (%%)'}},margin:{{t:20}}}});
</script></div>
""".format(x=json.dumps(iters[::step]), y=json.dumps(win_rates[::step]))

# 차트 2: 전략별 가중치 변화
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
html += '<div class="card"><h2>전략별 가중치 변화</h2><div id="chart2" class="chart"></div><script>\nvar traces2=['
for i, (sname, weights) in enumerate(history["weights"].items()):
    sw = [w * 100 for w in weights[::step]]
    si = iters[::step]
    html += '{{x:{x},y:{y},type:"scatter",mode:"lines",name:"{n}",line:{{color:"{c}",width:2}}}},'.format(
        x=json.dumps(si), y=json.dumps(sw), n=sname, c=colors[i % len(colors)])
html += """]; Plotly.newPlot('chart2',traces2,{xaxis:{title:'Iteration'},yaxis:{title:'Weight (%)'},margin:{t:20}});
</script></div>"""

# 차트 3: 국면별 성과
html += '<div class="card"><h2>국면별 성과 비교</h2>'
html += '<div id="chart3" class="chart"></div><script>'
regime_wr = []
for r in ["bull", "bear", "sideways"]:
    s = train_result["regime_stats"][r]
    regime_wr.append(s["win"] / s["total"] * 100 if s["total"] > 0 else 0)
html += """
Plotly.newPlot('chart3',[{{x:['상승장(Bull)','하락장(Bear)','횡보장(Sideways)'],y:{wr},type:'bar',
marker:{{color:['#2ecc71','#e74c3c','#f39c12']}}}}],{{yaxis:{{title:'Win Rate (%%)'}},margin:{{t:20}}}});
</script></div>""".format(wr=json.dumps(regime_wr))

# 차트 4: 500만원 → 최종 잔고 (전략별 비교 바 차트)
strat_names_chart = ["MA", "MACD", "RSI", "볼린저", "모멘텀", "거래량", "앙상블", "KOSPI보유"]
strat_keys = ["MA 전략", "MACD 전략", "RSI 전략", "볼린저밴드 전략", "모멘텀 전략", "거래량 전략", "최적 앙상블", "KOSPI 단순보유"]
final_caps = [investment_results[k]["final_capital"] for k in strat_keys]
bar_colors = ['#e74c3c' if c < INITIAL_CAPITAL else '#2ecc71' for c in final_caps]

html += """
<div class="card"><h2>500만원 투자 → 최종 잔고 비교</h2>
<div id="chart4" class="chart"></div>
<script>
Plotly.newPlot('chart4',[{{x:{names},y:{vals},type:'bar',marker:{{color:{colors}}},
text:{vals}.map(v=>v.toLocaleString()+'원'),textposition:'outside'}}],
{{yaxis:{{title:'최종 잔고 (원)'}},margin:{{t:20,b:80}},
shapes:[{{type:'line',y0:5000000,y1:5000000,x0:-0.5,x1:7.5,line:{{color:'#34495e',width:2,dash:'dash'}}}}]}});
</script></div>""".format(
    names=json.dumps(strat_names_chart),
    vals=json.dumps(final_caps),
    colors=json.dumps(bar_colors)
)

# 차트 5: 월별 잔고 변화
html += '<div class="card"><h2>월별 잔고 변화</h2><div id="chart5" class="chart"></div><script>\nvar traces5=['
chart_strats = ["RSI 전략", "최적 앙상블", "KOSPI 단순보유"]
chart_colors5 = ['#2ecc71', '#3498db', '#95a5a6']
for i, sname in enumerate(chart_strats):
    mb = investment_results[sname].get("monthly_balance", {})
    months = sorted(mb.keys())
    vals = [mb[m] for m in months]
    html += '{{x:{x},y:{y},type:"scatter",mode:"lines+markers",name:"{n}",line:{{color:"{c}",width:2}}}},'.format(
        x=json.dumps(months), y=json.dumps(vals), n=sname, c=chart_colors5[i])
html += """]; Plotly.newPlot('chart5',traces5,{xaxis:{title:'월'},yaxis:{title:'잔고 (원)'},margin:{t:20},
shapes:[{type:'line',y0:5000000,y1:5000000,x0:0,x1:1,xref:'paper',line:{color:'#34495e',width:1,dash:'dash'}}]});
</script></div>"""

# 차트 6: KOSPI 대비 초과 수익
html += '<div class="card"><h2>KOSPI 대비 초과 수익</h2>'
kospi_mb = investment_results["KOSPI 단순보유"].get("monthly_balance", {})
ensemble_mb = investment_results["최적 앙상블"].get("monthly_balance", {})
common_months = sorted(set(kospi_mb.keys()) & set(ensemble_mb.keys()))
excess = [(ensemble_mb.get(m, INITIAL_CAPITAL) - kospi_mb.get(m, INITIAL_CAPITAL)) for m in common_months]
excess_colors = ['#2ecc71' if e >= 0 else '#e74c3c' for e in excess]

html += """<div id="chart6" class="chart"></div>
<script>
Plotly.newPlot('chart6',[{{x:{months},y:{excess},type:'bar',marker:{{color:{colors}}}}}],
{{yaxis:{{title:'초과 수익 (원)'}},margin:{{t:20}}}});
</script></div>""".format(
    months=json.dumps(common_months),
    excess=json.dumps(excess),
    colors=json.dumps(excess_colors)
)

# 최적 가중치 테이블
html += '<div class="card"><h2>최적 전략 가중치</h2>'
html += '<table><tr><th>전략</th><th>기본 가중치</th><th>상승장</th><th>하락장</th><th>횡보장</th></tr>'
for name, w in sorted(optimal_weights.items(), key=lambda x: -x[1]):
    html += f'<tr><td>{name}</td><td>{w:.4f}</td>'
    for r in ["bull", "bear", "sideways"]:
        rw = w * REGIME_STRATEGY_MAP[r].get(name, 1.0)
        html += f'<td>{rw:.4f}</td>'
    html += '</tr>'
html += '</table></div>'

html += '</body></html>'

with open("results/report.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"리포트 저장: results/report.html")
print("\n" + "=" * 60)
print("모든 작업 완료!")
print("=" * 60)
