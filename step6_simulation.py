"""Step 6: 시뮬레이션 루프 + 최종 출력"""
import os
import json
import random
import pandas as pd
import numpy as np
from copy import deepcopy

from step2_strategies import get_all_strategies
from step3_regime import detect_market_regime, REGIME_STRATEGY_MAP
from step4_risk import RiskManager
from step5_backtest import (
    calculate_3day_return, judge_trade, update_weights,
    check_convergence, HOLDING_DAYS
)

os.makedirs("results", exist_ok=True)

print("=" * 60)
print("Step 6: 시뮬레이션 시작")
print("=" * 60)

# 데이터 로드
split_info = json.load(open("data/split_info.json", encoding="utf-8"))
kospi = pd.read_csv("data/kospi_index.csv", index_col=0, parse_dates=True)
ticker_info = pd.read_csv("data/top100_tickers.csv", encoding="utf-8-sig")
tickers = ticker_info["ticker"].tolist()
names = ticker_info["name"].tolist()

# 종목 데이터 로드
stock_data = {}
for t in tickers:
    path = f"data/stocks/{t}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > 30:  # 최소 데이터
            stock_data[t] = df

print(f"로드된 종목 수: {len(stock_data)}")

# 시장 국면 감지
regime = detect_market_regime(kospi)
regime_counts = regime.value_counts()
print("\n시장 국면 분포:")
for r, c in regime_counts.items():
    print(f"  {r}: {c}일 ({c/len(regime)*100:.1f}%)")

# 날짜 분리
train_end = pd.Timestamp(split_info["train_end"])
val_end = pd.Timestamp(split_info["val_end"])

def get_period_data(df, start_str, end_str):
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    mask = (df.index >= start) & (df.index <= end)
    return df[mask]

# ========== 시뮬레이션 함수 ==========
def run_simulation(stock_data, regime, split_info, period="train", max_iterations=10000, strategies=None):
    """한 기간에 대한 시뮬레이션 실행"""
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

    # 기록용
    history = {
        "iteration": [],
        "win_rate": [],
        "weights": {s.name: [] for s in strategies},
    }
    regime_stats = {"bull": {"win": 0, "total": 0},
                    "bear": {"win": 0, "total": 0},
                    "sideways": {"win": 0, "total": 0}}

    total_wins = 0
    total_trades = 0
    converged = False

    is_train = (period == "train")
    iterations = max_iterations if is_train else 1

    for iteration in range(1, iterations + 1):
        old_weights = {s.name: s.weight for s in strategies}
        iter_wins = 0
        iter_trades = 0

        # 랜덤 종목 샘플링 (매 이터레이션)
        sample_size = min(20, len(ticker_list))
        sampled = random.sample(ticker_list, sample_size)

        for ticker in sampled:
            df = stock_data[ticker]
            period_df = get_period_data(df, start, end)

            if len(period_df) < HOLDING_DAYS + 30:
                continue

            # 전략 시그널 생성
            signals = {}
            for strat in strategies:
                try:
                    sig = strat.generate_signal(period_df)
                    signals[strat.name] = sig
                except:
                    signals[strat.name] = pd.Series(0, index=period_df.index)

            # 시그널 결합 (가중 합)
            combined = pd.Series(0.0, index=period_df.index)
            for strat in strategies:
                sig = signals[strat.name]
                # 국면별 조정
                for date_idx in period_df.index:
                    if date_idx in regime.index:
                        r = regime.loc[date_idx]
                    else:
                        r = "sideways"
                    regime_mult = REGIME_STRATEGY_MAP.get(r, {}).get(strat.name, 1.0)
                    combined.loc[date_idx] += sig.loc[date_idx] * strat.weight * regime_mult

            # 매수 시그널이 있는 날짜 찾기
            buy_dates = combined[combined > 0.3].index.tolist()

            if not buy_dates:
                continue

            # 랜덤하게 매수일 선택 (학습 다양성)
            if is_train:
                if len(buy_dates) > 5:
                    buy_dates = random.sample(buy_dates, 5)

            for buy_date in buy_dates:
                buy_idx = period_df.index.get_loc(buy_date)
                ret = calculate_3day_return(period_df, buy_idx)

                if ret is None:
                    continue

                # 리스크 관리 적용
                should_exit, reason = risk_mgr.should_exit(ret)
                if reason == "trading_halted":
                    continue

                is_win = judge_trade(ret)
                iter_wins += int(is_win)
                iter_trades += 1
                total_wins += int(is_win)
                total_trades += 1

                # 국면별 통계
                if buy_date in regime.index:
                    r = regime.loc[buy_date]
                else:
                    r = "sideways"
                regime_stats[r]["total"] += 1
                if is_win:
                    regime_stats[r]["win"] += 1

                # 리스크 업데이트
                risk_mgr.update_loss_count(not is_win)

                # 가중치 업데이트 (train만)
                if is_train:
                    # 어떤 전략이 기여했는지 판단
                    trade_results = []
                    for strat in strategies:
                        sig_val = signals[strat.name].get(buy_date, 0)
                        if sig_val > 0:
                            trade_results.append((strat.name, is_win))
                    if trade_results:
                        update_weights(strategies, trade_results)

            risk_mgr.reset()

        # 이터레이션 기록
        win_rate = total_wins / total_trades if total_trades > 0 else 0
        history["iteration"].append(iteration)
        history["win_rate"].append(win_rate)
        for s in strategies:
            history["weights"][s.name].append(s.weight)

        # 500회마다 중간 결과
        if is_train and iteration % 500 == 0:
            print(f"\n--- {iteration}회 중간 결과 ---")
            print(f"  누적 달성률: {win_rate*100:.2f}% ({total_wins}/{total_trades})")
            for s in strategies:
                print(f"  {s.name:12s}: {s.weight:.4f}")

        # 수렴 감지 (train만)
        if is_train and iteration > 100:
            new_weights = {s.name: s.weight for s in strategies}
            if check_convergence(old_weights, new_weights, threshold=0.01):
                if iteration > 500:  # 최소 500회 이상 돌아야 수렴 판정
                    print(f"\n수렴 감지! {iteration}회에서 종료")
                    converged = True
                    break

    final_win_rate = total_wins / total_trades if total_trades > 0 else 0

    return {
        "period": period,
        "iterations": iteration if is_train else 1,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": final_win_rate,
        "converged": converged,
        "regime_stats": regime_stats,
        "history": history,
        "strategies": {s.name: s.weight for s in strategies},
    }

# ========== 실행 ==========
print("\n" + "=" * 60)
print("Train 시뮬레이션 시작 (최대 10,000회)")
print("=" * 60)

strategies = get_all_strategies()
train_result = run_simulation(stock_data, regime, split_info, "train", max_iterations=10000, strategies=strategies)

print(f"\nTrain 결과: 달성률 {train_result['win_rate']*100:.2f}%")
print(f"  총 거래: {train_result['total_trades']}, 승: {train_result['total_wins']}")
print(f"  수렴: {'Yes' if train_result['converged'] else 'No'} ({train_result['iterations']}회)")

# 최적 가중치 저장
optimal_weights = train_result["strategies"]
print("\n최적 전략 가중치:")
for name, w in sorted(optimal_weights.items(), key=lambda x: -x[1]):
    print(f"  {name:12s}: {w:.4f}")

# Validation
print("\n" + "=" * 60)
print("Validation 검증 시작")
print("=" * 60)

val_strategies = get_all_strategies()
for s in val_strategies:
    s.weight = optimal_weights.get(s.name, 1/6)

val_result = run_simulation(stock_data, regime, split_info, "val", strategies=val_strategies)
print(f"\nValidation 결과: 달성률 {val_result['win_rate']*100:.2f}%")
print(f"  총 거래: {val_result['total_trades']}, 승: {val_result['total_wins']}")

# Test (Validation 55% 이상일 때)
test_result = None
if val_result["win_rate"] >= 0.55:
    print("\n" + "=" * 60)
    print("Validation 55% 이상! Test 최종 검증 시작")
    print("=" * 60)
    test_strategies = get_all_strategies()
    for s in test_strategies:
        s.weight = optimal_weights.get(s.name, 1/6)
    test_result = run_simulation(stock_data, regime, split_info, "test", strategies=test_strategies)
    print(f"\nTest 결과: 달성률 {test_result['win_rate']*100:.2f}%")
    print(f"  총 거래: {test_result['total_trades']}, 승: {test_result['total_wins']}")
else:
    print(f"\nValidation 달성률 {val_result['win_rate']*100:.2f}% < 55% → Test 생략")
    # 그래도 참고용으로 Test 실행
    print("참고용 Test 실행...")
    test_strategies = get_all_strategies()
    for s in test_strategies:
        s.weight = optimal_weights.get(s.name, 1/6)
    test_result = run_simulation(stock_data, regime, split_info, "test", strategies=test_strategies)
    print(f"  Test 참고 달성률: {test_result['win_rate']*100:.2f}%")

# ========== 최종 출력 ==========
print("\n" + "=" * 60)
print("최종 결과")
print("=" * 60)

# 1. 국면별 최적 전략 조합 + 가중치
print("\n[1] 국면별 최적 전략 조합:")
for r in ["bull", "bear", "sideways"]:
    print(f"\n  {r}:")
    stats = train_result["regime_stats"][r]
    wr = stats["win"] / stats["total"] * 100 if stats["total"] > 0 else 0
    print(f"    달성률: {wr:.1f}% ({stats['win']}/{stats['total']})")
    regime_weights = {}
    for name, base_w in optimal_weights.items():
        mult = REGIME_STRATEGY_MAP[r].get(name, 1.0)
        regime_weights[name] = base_w * mult
    total_rw = sum(regime_weights.values())
    for name, rw in sorted(regime_weights.items(), key=lambda x: -x[1]):
        print(f"    {name:12s}: {rw/total_rw:.4f}")

# 2. 전체 달성률
print(f"\n[2] 전체 달성률:")
print(f"  Train:      {train_result['win_rate']*100:.2f}%")
print(f"  Validation: {val_result['win_rate']*100:.2f}%")
if test_result:
    print(f"  Test:       {test_result['win_rate']*100:.2f}%")

# 3. JSON 저장
result_json = {
    "optimal_weights": optimal_weights,
    "train": {
        "win_rate": train_result["win_rate"],
        "total_trades": train_result["total_trades"],
        "total_wins": train_result["total_wins"],
        "iterations": train_result["iterations"],
        "converged": train_result["converged"],
        "regime_stats": train_result["regime_stats"],
    },
    "validation": {
        "win_rate": val_result["win_rate"],
        "total_trades": val_result["total_trades"],
        "total_wins": val_result["total_wins"],
        "regime_stats": val_result["regime_stats"],
    },
    "regime_strategy_map": REGIME_STRATEGY_MAP,
    "split_info": split_info,
}
if test_result:
    result_json["test"] = {
        "win_rate": test_result["win_rate"],
        "total_trades": test_result["total_trades"],
        "total_wins": test_result["total_wins"],
        "regime_stats": test_result["regime_stats"],
    }

with open("results/simulation_result.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)
print(f"\n[3] 결과 저장: results/simulation_result.json")

# 4. HTML 리포트 생성
print(f"\n[4] HTML 리포트 생성 중...")

history = train_result["history"]

html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Trading Simulation Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body { font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 40px; }
.card { background: white; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.metric { display: inline-block; text-align: center; padding: 15px 30px; margin: 10px; background: #ecf0f1; border-radius: 8px; }
.metric .value { font-size: 2em; font-weight: bold; color: #2c3e50; }
.metric .label { color: #7f8c8d; font-size: 0.9em; }
.chart { width: 100%%; height: 400px; }
table { width: 100%%; border-collapse: collapse; margin: 10px 0; }
th, td { padding: 10px; text-align: center; border: 1px solid #ddd; }
th { background: #3498db; color: white; }
tr:nth-child(even) { background: #f9f9f9; }
</style>
</head><body>
<h1>Trading Simulation Report</h1>
"""

# 요약 카드
html += '<div class="card"><h2>Performance Summary</h2>'
html += '<div class="metric"><div class="value">{:.1f}%</div><div class="label">Train</div></div>'.format(train_result["win_rate"]*100)
html += '<div class="metric"><div class="value">{:.1f}%</div><div class="label">Validation</div></div>'.format(val_result["win_rate"]*100)
if test_result:
    html += '<div class="metric"><div class="value">{:.1f}%</div><div class="label">Test</div></div>'.format(test_result["win_rate"]*100)
html += '<div class="metric"><div class="value">{}</div><div class="label">Iterations</div></div>'.format(train_result["iterations"])
html += '<div class="metric"><div class="value">{}</div><div class="label">Total Trades</div></div>'.format(train_result["total_trades"])
html += '</div>'

# 차트 1: 회차별 달성률 추이
iters = history["iteration"]
win_rates = [w * 100 for w in history["win_rate"]]
html += """
<div class="card">
<h2>Win Rate Over Iterations</h2>
<div id="chart1" class="chart"></div>
<script>
Plotly.newPlot('chart1', [{{
    x: {iters},
    y: {wr},
    type: 'scatter',
    mode: 'lines',
    name: 'Win Rate',
    line: {{color: '#3498db', width: 2}}
}}], {{
    xaxis: {{title: 'Iteration'}},
    yaxis: {{title: 'Win Rate (%%)'}},
    margin: {{t: 20}}
}});
</script></div>
""".format(iters=json.dumps(iters[::max(1,len(iters)//500)]),
           wr=json.dumps(win_rates[::max(1,len(win_rates)//500)]))

# 차트 2: 전략별 가중치 변화
html += '<div class="card"><h2>Strategy Weights Over Time</h2><div id="chart2" class="chart"></div><script>\nvar traces2 = ['
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
step = max(1, len(iters) // 500)
for i, (sname, weights) in enumerate(history["weights"].items()):
    sampled_w = [w * 100 for w in weights[::step]]
    sampled_i = iters[::step]
    html += '{{x:{x},y:{y},type:"scatter",mode:"lines",name:"{n}",line:{{color:"{c}",width:2}}}},'.format(
        x=json.dumps(sampled_i), y=json.dumps(sampled_w), n=sname, c=colors[i % len(colors)])
html += """];
Plotly.newPlot('chart2', traces2, {xaxis:{title:'Iteration'},yaxis:{title:'Weight (%)'},margin:{t:20}});
</script></div>
"""

# 차트 3: 국면별 성과 비교
html += '<div class="card"><h2>Performance by Market Regime</h2>'
html += '<table><tr><th>Regime</th><th>Total Trades</th><th>Wins</th><th>Win Rate</th></tr>'
for r in ["bull", "bear", "sideways"]:
    stats = train_result["regime_stats"][r]
    wr = stats["win"] / stats["total"] * 100 if stats["total"] > 0 else 0
    html += f'<tr><td>{r}</td><td>{stats["total"]}</td><td>{stats["win"]}</td><td>{wr:.1f}%</td></tr>'
html += '</table>'

# 바 차트
html += """
<div id="chart3" class="chart"></div>
<script>
var regimes = ['bull', 'bear', 'sideways'];
var winRates = [""" + ",".join([
    str(train_result["regime_stats"][r]["win"] / train_result["regime_stats"][r]["total"] * 100
        if train_result["regime_stats"][r]["total"] > 0 else 0)
    for r in ["bull", "bear", "sideways"]
]) + """];
Plotly.newPlot('chart3', [{
    x: regimes, y: winRates, type: 'bar',
    marker: {color: ['#2ecc71', '#e74c3c', '#f39c12']}
}], {yaxis:{title:'Win Rate (%)'},margin:{t:20}});
</script></div>
"""

# 최적 가중치 테이블
html += '<div class="card"><h2>Optimal Strategy Weights</h2>'
html += '<table><tr><th>Strategy</th><th>Weight</th><th>Bull</th><th>Bear</th><th>Sideways</th></tr>'
for name, w in sorted(optimal_weights.items(), key=lambda x: -x[1]):
    bull_w = w * REGIME_STRATEGY_MAP["bull"].get(name, 1)
    bear_w = w * REGIME_STRATEGY_MAP["bear"].get(name, 1)
    side_w = w * REGIME_STRATEGY_MAP["sideways"].get(name, 1)
    html += f'<tr><td>{name}</td><td>{w:.4f}</td><td>{bull_w:.4f}</td><td>{bear_w:.4f}</td><td>{side_w:.4f}</td></tr>'
html += '</table></div>'

html += '</body></html>'

with open("results/report.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"리포트 저장: results/report.html")
print("\n" + "=" * 60)
print("모든 작업 완료!")
print("=" * 60)
