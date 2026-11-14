#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_data.py —— 数据采集 (curl 后端, 稳定)
- FRED (St. Louis Fed): 宏观/利率/信用/金融条件/杠杆率 因子 (免 key CSV)
- Yahoo Finance: 黄金、美元指数、标普、原油 日度价格 (period1=0 取全历史)
输出: data/raw/*.csv
"""
import os, subprocess, json, datetime, time

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "data", "raw")
os.makedirs(RAW, exist_ok=True)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

FRED_SERIES = {
    "SP500": "标普500指数", "DGS10": "美国10Y国债收益率", "DGS2": "美国2Y国债收益率",
    "DFII10": "10Y TIPS实际利率", "T10YIE": "10Y盈亏平衡通胀", "T10Y2Y": "10Y-2Y期限利差",
    "DCOILWTICO": "WTI现货(周度)", "DCOILBRENTEU": "布伦特现货(周度)",
    "BAA10Y": "Baa-10Y信用利差", "BAMLC0A0CM": "投资级OAS", "BAMLH0A0HYM2": "高收益OAS",
    "BAMLC0A4CBBBEY": "投资级有效收益率", "NFCI": "金融条件指数",
    "NFCINONFINLEVERAGE": "NFCI非金融杠杆", "VIXCLS": "VIX恐慌指数",
    "DTWEXBGS": "广义美元指数", "DTWEXAFEGS": "发达经济体美元指数",
}
YAHOO_SYMBOLS = {
    "GC=F": "黄金期货", "GLD": "黄金ETF", "DX-Y.NYB": "美元指数DXY",
    "^GSPC": "标普500", "CL=F": "WTI原油期货", "^VIX": "VIX指数",
}

def curl(url, ua=False):
    cmd = ["curl", "--fail", "--silent", "--show-error", "--location",
           "--retry", "2", "--retry-delay", "3", "--retry-all-errors", "--max-time", "30"]
    if ua:
        cmd += ["-A", UA]
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

def fetch_fred(sid):
    # FRED 对浏览器 UA 返回空, 故不带 UA
    return curl(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")

def fetch_yahoo(sym):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1=0&period2=4102444800&interval=1d&events=div%2Csplit")
    data = json.loads(curl(url, ua=True))
    res = data["chart"]["result"][0]
    ts = res.get("timestamp", [])
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", [])
    rows = []
    for i, t in enumerate(ts):
        dt = datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")
        a = adj[i] if i < len(adj) else None
        rows.append((dt, q["open"][i], q["high"][i], q["low"][i], q["close"][i], a, q["volume"][i]))
    return rows

def main():
    print("== FRED ==", flush=True)
    for sid, name in FRED_SERIES.items():
        try:
            txt = fetch_fred(sid)
            if not txt.lstrip().startswith("observation_date"):
                print(f"  [FAIL] {sid}", flush=True); continue
            open(os.path.join(RAW, f"fred_{sid}.csv"), "w").write(txt)
            print(f"  [OK] {sid:22s} {name:16s} rows={txt.count(chr(10))-1}", flush=True)
        except Exception as e:
            print(f"  [ERR] {sid}: {e}", flush=True)
        time.sleep(0.2)
    print("== Yahoo ==", flush=True)
    for sym, name in YAHOO_SYMBOLS.items():
        try:
            rows = fetch_yahoo(sym)
            if not rows:
                print(f"  [FAIL] {sym}", flush=True); continue
            fp = os.path.join(RAW, f"yahoo_{sym.replace('^','')}.csv")
            with open(fp, "w") as f:
                f.write("Date,Open,High,Low,Close,AdjClose,Volume\n")
                for r in rows:
                    f.write(",".join("" if x is None else str(x) for x in r) + "\n")
            print(f"  [OK] {sym:12s} {name:14s} rows={len(rows)} {rows[0][0]}..{rows[-1][0]}", flush=True)
        except Exception as e:
            print(f"  [ERR] {sym}: {e}", flush=True)
        time.sleep(0.2)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
