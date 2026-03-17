"""기존 데이터 이전 1년 3개월치 추가 수집 후 병합"""
import os
import time
import ssl
import json
import urllib3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
from bs4 import BeautifulSoup

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

print("=" * 60)
print("추가 데이터 수집 시작 (2024-12-26 이전 ~1년 3개월)")
print("=" * 60)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def make_session():
    s = requests.Session()
    s.verify = False
    s.headers.update(HEADERS)
    return s

def get_naver_ohlcv_pages(code, start_page, end_page):
    """네이버 금융 특정 페이지 범위 크롤링"""
    sess = make_session()
    all_data = []
    for page in range(start_page, end_page + 1):
        url = f"https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
        try:
            resp = sess.get(url, timeout=10)
            resp.encoding = "euc-kr"
            tables = pd.read_html(StringIO(resp.text), header=0)
            if tables:
                df = tables[0].dropna(how="all")
                if len(df) == 0:
                    break
                all_data.append(df)
        except:
            break
        time.sleep(0.15)

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.dropna(subset=["날짜"])
    result["날짜"] = pd.to_datetime(result["날짜"])
    result = result.sort_values("날짜").reset_index(drop=True).set_index("날짜")
    keep_cols = [c for c in ["종가", "시가", "고가", "저가", "거래량"] if c in result.columns]
    result = result[keep_cols]
    for c in result.columns:
        result[c] = pd.to_numeric(result[c], errors="coerce")
    result = result.dropna()
    return result

# 티커 목록 로드
ticker_info = pd.read_csv("data/top100_tickers.csv", encoding="utf-8-sig")
tickers = ticker_info["ticker"].astype(str).str.zfill(6).tolist()
names = ticker_info["name"].tolist()

# 각 종목별 추가 수집 (페이지 41~80 = 이전 400거래일 추가)
print(f"\n[1/3] {len(tickers)}종목 추가 데이터 수집 (페이지 41~80)...")
success = 0
fail = 0
for idx, (ticker, name) in enumerate(zip(tickers, names)):
    print(f"  [{idx+1:3d}/{len(tickers)}] {ticker} ({name}) 추가수집...", end="", flush=True)

    existing_path = f"data/stocks/{ticker}.csv"
    if not os.path.exists(existing_path):
        print(" SKIP (기존 파일 없음)")
        fail += 1
        continue

    try:
        # 기존 데이터 로드
        existing_df = pd.read_csv(existing_path, index_col=0, parse_dates=True)
        old_count = len(existing_df)

        # 페이지 41~80 크롤링 (이전 데이터)
        new_df = get_naver_ohlcv_pages(ticker, 41, 80)

        if len(new_df) == 0:
            print(f" 추가 데이터 없음 (유지 {old_count}일)")
            continue

        # 병합 (중복 제거)
        merged = pd.concat([existing_df, new_df])
        merged = merged[~merged.index.duplicated(keep='first')]
        merged = merged.sort_index()

        # 저장
        merged.to_csv(existing_path, encoding="utf-8-sig")
        added = len(merged) - old_count
        print(f" OK ({old_count}→{len(merged)}일, +{added}일)")
        success += 1
    except Exception as e:
        print(f" FAIL ({e})")
        fail += 1
    time.sleep(0.3)

print(f"\n종목 수집 완료: 성공 {success}, 실패 {fail}")

# KOSPI 지수 추가 수집
print(f"\n[2/3] KOSPI 지수 추가 수집 (페이지 50~100)...")
SESSION = make_session()
kospi_extra = []
for page in range(50, 101):
    url = f"https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI&page={page}"
    try:
        resp = SESSION.get(url, timeout=10)
        resp.encoding = "euc-kr"
        tables = pd.read_html(StringIO(resp.text), header=0)
        if tables:
            df = tables[0].dropna(how="all")
            if len(df) == 0:
                break
            kospi_extra.append(df)
    except:
        break
    time.sleep(0.2)

if kospi_extra:
    new_kospi = pd.concat(kospi_extra, ignore_index=True)
    new_kospi = new_kospi.dropna(subset=["날짜"])
    new_kospi["날짜"] = pd.to_datetime(new_kospi["날짜"])
    new_kospi = new_kospi.sort_values("날짜").reset_index(drop=True).set_index("날짜")
    if "체결가" in new_kospi.columns:
        new_kospi = new_kospi.rename(columns={"체결가": "종가"})
    for c in new_kospi.columns:
        new_kospi[c] = pd.to_numeric(new_kospi[c], errors="coerce")

    # 기존 병합
    existing_kospi = pd.read_csv("data/kospi_index.csv", index_col=0, parse_dates=True)
    old_len = len(existing_kospi)
    merged_kospi = pd.concat([existing_kospi, new_kospi])
    merged_kospi = merged_kospi[~merged_kospi.index.duplicated(keep='first')]
    merged_kospi = merged_kospi.sort_index()
    merged_kospi.to_csv("data/kospi_index.csv", encoding="utf-8-sig")
    print(f"KOSPI 지수: {old_len}→{len(merged_kospi)}일 (+{len(merged_kospi)-old_len}일)")
else:
    print("KOSPI 추가 데이터 없음")

# 날짜 분리 재계산
print(f"\n[3/3] Train/Validation/Test 재분리...")
kospi = pd.read_csv("data/kospi_index.csv", index_col=0, parse_dates=True)
dates = sorted(kospi.index)
n = len(dates)
train_end_idx = int(n * 0.65)
val_end_idx = int(n * 0.85)

train_dates = dates[:train_end_idx]
val_dates = dates[train_end_idx:val_end_idx]
test_dates = dates[val_end_idx:]

split_info = {
    "train_start": str(train_dates[0].date()),
    "train_end": str(train_dates[-1].date()),
    "val_start": str(val_dates[0].date()),
    "val_end": str(val_dates[-1].date()),
    "test_start": str(test_dates[0].date()),
    "test_end": str(test_dates[-1].date()),
    "train_days": len(train_dates),
    "val_days": len(val_dates),
    "test_days": len(test_dates),
}

with open("data/split_info.json", "w", encoding="utf-8") as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)

print(f"  전체 기간: {dates[0].date()} ~ {dates[-1].date()} ({n}일)")
print(f"  Train:      {split_info['train_start']} ~ {split_info['train_end']} ({split_info['train_days']}일)")
print(f"  Validation: {split_info['val_start']} ~ {split_info['val_end']} ({split_info['val_days']}일)")
print(f"  Test:       {split_info['test_start']} ~ {split_info['test_end']} ({split_info['test_days']}일)")
print(f"\n추가 수집 및 병합 완료!")
