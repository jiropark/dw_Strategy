"""Step 1: 데이터 수집 - 네이버 금융 기반 (최적화)"""
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
from concurrent.futures import ThreadPoolExecutor, as_completed

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

print("=" * 60)
print("Step 1: 데이터 수집 시작 (네이버 금융)")
print("=" * 60)

os.makedirs("data/stocks", exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def make_session():
    s = requests.Session()
    s.verify = False
    s.headers.update(HEADERS)
    return s

SESSION = make_session()

def get_naver_ohlcv(code, pages=40):
    """네이버 금융에서 일별 시세 크롤링"""
    sess = make_session()
    all_data = []
    for page in range(1, pages + 1):
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
    cutoff = datetime.now() - timedelta(days=365*3)
    result = result[result.index >= cutoff]
    return result

def get_top100_tickers():
    """시가총액 상위 100종목 티커 추출"""
    tickers = []
    names = []
    for page in range(1, 5):
        url = f"https://finance.naver.com/sise/sise_market_sum.nhn?sosok=0&page={page}"
        try:
            resp = SESSION.get(url, timeout=10)
            resp.encoding = "euc-kr"
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"class": "type_2"})
            if table:
                for row in table.find_all("tr"):
                    cols = row.find_all("td")
                    if len(cols) >= 2:
                        link = cols[1].find("a")
                        if link and link.get("href") and "code=" in link["href"]:
                            code = link["href"].split("code=")[-1]
                            name = link.text.strip()
                            if code and name and len(code) == 6:
                                tickers.append(code)
                                names.append(name)
        except Exception as e:
            print(f"  page {page} 실패: {e}")
        time.sleep(0.5)
    return tickers[:100], names[:100]

# 기존 수집된 파일 체크
existing = set()
for f in os.listdir("data/stocks"):
    if f.endswith(".csv"):
        existing.add(f.replace(".csv", ""))

# 티커 목록 로드 또는 추출
if os.path.exists("data/top100_tickers.csv"):
    ticker_info = pd.read_csv("data/top100_tickers.csv", encoding="utf-8-sig")
    tickers = ticker_info["ticker"].astype(str).str.zfill(6).tolist()
    names = ticker_info["name"].tolist()
    print(f"기존 티커 목록 로드: {len(tickers)}종목")
else:
    print("\n[1/4] KOSPI 시가총액 상위 100종목 추출 중...")
    tickers, names = get_top100_tickers()
    print(f"  추출: {len(tickers)}종목")
    ticker_info = pd.DataFrame({"ticker": tickers, "name": names})
    ticker_info.to_csv("data/top100_tickers.csv", index=False, encoding="utf-8-sig")

# 미수집 종목만 필터
remaining = [(t, n) for t, n in zip(tickers, names) if t not in existing]
print(f"이미 수집: {len(existing)}, 남은 종목: {len(remaining)}")

# 수집
print(f"\n[2/4] 종목 OHLCV 수집 중...")
success = len(existing)
fail = 0
for idx, (ticker, name) in enumerate(remaining):
    total_idx = len(existing) + idx + 1
    print(f"  [{total_idx:3d}/{len(tickers)}] {ticker} ({name}) 수집중...", end="", flush=True)
    try:
        df = get_naver_ohlcv(ticker, pages=40)
        if len(df) > 0:
            df.to_csv(f"data/stocks/{ticker}.csv", encoding="utf-8-sig")
            print(f" OK ({len(df)}일)")
            success += 1
        else:
            print(" SKIP")
            fail += 1
    except Exception as e:
        print(f" FAIL ({e})")
        fail += 1
    time.sleep(0.3)

print(f"\n수집 완료: 성공 {success}, 실패 {fail}")

# 3. KOSPI 지수
if not os.path.exists("data/kospi_index.csv"):
    print(f"\n[3/4] KOSPI 지수 수집 중...")
    kospi_data = []
    for page in range(1, 50):
        url = f"https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI&page={page}"
        try:
            resp = SESSION.get(url, timeout=10)
            resp.encoding = "euc-kr"
            tables = pd.read_html(StringIO(resp.text), header=0)
            if tables:
                df = tables[0].dropna(how="all")
                if len(df) == 0:
                    break
                kospi_data.append(df)
        except:
            break
        time.sleep(0.2)

    if kospi_data:
        kospi = pd.concat(kospi_data, ignore_index=True)
        kospi = kospi.dropna(subset=["날짜"])
        kospi["날짜"] = pd.to_datetime(kospi["날짜"])
        kospi = kospi.sort_values("날짜").reset_index(drop=True).set_index("날짜")
        if "체결가" in kospi.columns:
            kospi = kospi.rename(columns={"체결가": "종가"})
        cutoff = datetime.now() - timedelta(days=365*3)
        kospi = kospi[kospi.index >= cutoff]
        for c in kospi.columns:
            kospi[c] = pd.to_numeric(kospi[c], errors="coerce")
        kospi.to_csv("data/kospi_index.csv", encoding="utf-8-sig")
        print(f"KOSPI 지수 수집 완료 ({len(kospi)}일)")
else:
    print(f"\n[3/4] KOSPI 지수 이미 수집됨")

# 4. 날짜 분리
print(f"\n[4/4] Train/Validation/Test 분리...")
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

print(f"  Train:      {split_info['train_start']} ~ {split_info['train_end']} ({split_info['train_days']}일)")
print(f"  Validation: {split_info['val_start']} ~ {split_info['val_end']} ({split_info['val_days']}일)")
print(f"  Test:       {split_info['test_start']} ~ {split_info['test_end']} ({split_info['test_days']}일)")
print(f"\nStep 1 완료!")
