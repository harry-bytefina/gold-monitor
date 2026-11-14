#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor.py —— 黄金择时动态监测站
最终模型: oil(0.65) + gap(0.35), 滞回带 ±0.05, 最小持仓8周, 危机±0.3, RSI 80/20, 二次项 γ=0.02
"""
import pandas as pd, numpy as np, os, json, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "output", "features_weekly.csv")
OUT_HTML = os.path.join(ROOT, "output", "monitor.html")
ANALYSIS_START = "2016-01-01"

def zc(s, w=52):
    m = s.rolling(w, min_periods=w//2).mean(); sd = s.rolling(w, min_periods=w//2).std()
    return (s - m) / sd

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1+up/dn)

def build_signal(df):
    oil = df["sig_oilgold"].to_numpy(); gap = df["sig_gap"].to_numpy()
    score = 0.65*oil + 0.35*gap
    crisis = (zc(df["credit"]) + zc(df["vix"]) + zc(np.log(df["dxy"]).diff(4))
              + zc(df["avgcorr"]) + zc(df["sb_corr"])) / 5
    cr = crisis.to_numpy(); chg4 = crisis.diff(4).to_numpy()
    R14 = rsi(df["gold"]).to_numpy()
    N = len(df); pos = np.zeros(N); state = 0; h = 0; flat = 999
    for t in range(N):
        s = score[t]
        if state == 1 and chg4[t] > 0.3: state = 0; h = 0; flat = 0
        if state == 0 and chg4[t] < -0.3 and cr[t] > 0.5: state = 1; h = 4
        if state == 1 and h == 0 and (not np.isfinite(s) or s < -0.05): state = 0; flat = 0
        theta = 0.05 + 0.02*((flat-17)/26)**2
        if state == 0 and h == 0 and np.isfinite(s) and s > theta: state = 1; h = 8
        if state == 0: flat += 1
        if h > 0: h -= 1
        pos[t] = state
    pos[R14 > 80] = 0.0; pos[(R14 < 20) & (score > -0.3)] = 1.0
    df = df.copy()
    df["score_final"] = score; df["crisis"] = crisis; df["rsi"] = R14
    df["oil"] = oil; df["gap"] = gap; df["position"] = pos
    return df

def fdate(t): return pd.Timestamp(t).strftime("%Y-%m-%d")

def latest_price_date(fallback):
    try:
        raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "yahoo_GLD.csv"))
        dates = pd.to_datetime(raw["Date"], errors="coerce")
        prices = pd.to_numeric(raw["AdjClose"], errors="coerce")
        prices = prices.where(prices.notna(), pd.to_numeric(raw["Close"], errors="coerce"))
        valid_dates = dates[dates.notna() & np.isfinite(prices) & (prices > 0)]
        return fdate(valid_dates.max()) if len(valid_dates) else fallback
    except (OSError, ValueError, KeyError):
        return fallback

def trade_history(df, equity, start="2016-01-01"):
    if df.empty:
        return []
    movement = df["position"].diff()
    events = df.index[movement.notna() & movement.ne(0)]
    events = events[events >= pd.Timestamp(start)]
    records = []
    for i, began in enumerate(events):
        ongoing = i == len(events) - 1
        ended = df.index[-1] if ongoing else events[i + 1]
        initial, final = float(equity.loc[began]), float(equity.loc[ended])
        gain = final / initial - 1 if (
            np.isfinite(initial) and np.isfinite(final) and initial > 0 and final > 0
        ) else None
        records.append({
            "start": began,
            "end": ended,
            "ongoing": ongoing,
            "position": int(df.loc[began, "position"]),
            "price": float(df.loc[began, "gold"]),
            "return": gain,
        })
    return records[::-1]


def history_html(records, visible=8):
    rows = []
    for i, record in enumerate(records):
        began = record["start"].strftime("%Y-%m-%d")
        ended = record["end"].strftime("%Y-%m-%d")
        cls, action = ("up", "买入") if record["position"] else ("dn", "卖出")
        extra = ' data-history-extra' if i >= visible else ''
        gain = record["return"]
        gain_cls = "up" if gain is not None and gain > 0 else "dn" if gain is not None and gain < 0 else "neutral"
        gain_text = f"{gain:+.2%}" if gain is not None else "—"
        progress = " · 进行中" if record["ongoing"] else ""
        interval = f"{began} 至 {ended}{progress}"
        rows.append(
            f'<tr class="history-row"{extra}>'
            f'<td class="mono">{began}</td>'
            f'<td><span class="trade-tag {cls}">{action}</span></td>'
            f'<td class="mono">{record["price"]:,.2f}</td>'
            f'<td class="history-return" data-history-return title="{interval}">'
            f'<span class="mono {gain_cls}">{gain_text}</span>'
            f'<small class="history-period">至 {ended}{progress}</small></td></tr>'
        )
    return "".join(rows)

def main():
    df = pd.read_csv(FEAT, index_col=0, parse_dates=True)
    df = build_signal(df)
    last = df.iloc[-1]; date = fdate(df.index[-1])
    gp = float(last["gold"]); pos = int(last["position"])
    score = float(last["score_final"]); rsi_v = float(last["rsi"])
    crisis_v = float(last["crisis"]); oil_v = float(last["oil"]); gap_v = float(last["gap"])
    previous_price = float(df["gold"].iloc[-2])
    price_change = gp - previous_price; price_pct = gp / previous_price - 1
    source_date = latest_price_date(date)

    pos_cn = "做多" if pos == 1 else "空仓"
    pos_color = "#34c759" if pos == 1 else "#ff3b30"
    if score > 0.05: action = "信号偏多 · 持有" if pos == 1 else "信号偏多 · 等待入场"
    elif score < -0.05: action = "信号偏空 · 持仓观察" if pos == 1 else "信号偏空 · 回避"
    else: action = "信号中性 · 持仓观察" if pos == 1 else "信号中性 · 观望"

    pos_s = df["position"]; diff = pos_s.diff().fillna(0)
    changes = diff[diff != 0]
    last_change = fdate(changes.index[-1]) if len(changes) else "—"
    since = None
    for t in df.index[::-1]:
        if pos_s.loc[t] != pos: break
        since = t
    hold_weeks = (df.index[-1] - since).days // 7 if since is not None else 0

    g = df["gold_ret"].fillna(0.0)
    r_strat = pos_s.shift(1).fillna(0)*g - 0.0010*pos_s.diff().abs().fillna(0)
    eq_s = (1+r_strat).cumprod(); eq_b = (1+g).cumprod()
    ytd = (1+r_strat.loc[df.index >= f"{df.index[-1].year}-01-01"]).prod()-1
    ytd_bh = (1+g.loc[df.index >= f"{df.index[-1].year}-01-01"]).prod()-1
    recent = df.loc[df.index >= ANALYSIS_START]
    window_equity = eq_s.loc[recent.index]
    window_returns = window_equity.pct_change().dropna()
    full_cagr = (window_equity.iloc[-1]/window_equity.iloc[0])**(52/(len(recent)-1))-1
    full_sharpe = window_returns.mean()/window_returns.std()*np.sqrt(52)
    full_mdd = (window_equity/window_equity.cummax()-1).min()
    def sj(s, nd=3):
        return json.dumps([[fdate(t), round(float(v), nd) if np.isfinite(v) else None]
                           for t, v in s.items()], allow_nan=False)
    GOLD = sj(recent["gold"], 2); SCORE = sj(recent["score_final"], 4); RSI = sj(recent["rsi"], 1)
    CRISIS = sj(recent["crisis"], 3); EQS = sj(window_equity, 8); EQB = sj(eq_b.loc[recent.index], 8)
    dd_s = (window_equity/window_equity.cummax()-1)*100; DD = sj(dd_s, 2)

    analysis_rows = []
    for t, row in recent.iterrows():
        values = {"date": fdate(t), "price": row["gold"], "equity": eq_s.loc[t],
                  "benchmark": eq_b.loc[t], "position": row["position"],
                  "score": row["score_final"], "rsi": row["rsi"],
                  "crisis": row["crisis"], "oil": row["oil"], "gap": row["gap"]}
        analysis_rows.append({k: v if k == "date" else
                              (round(float(v), 8) if np.isfinite(v) else None)
                              for k, v in values.items()})

    buy_pts = []; sell_pts = []
    px = recent["position"]; prev = None
    for t, v in px.items():
        if prev is not None and v != prev:
            (buy_pts if v == 1 else sell_pts).append({"x": fdate(t), "y": round(float(recent["gold"].loc[t]), 2)})
        prev = v

    history = trade_history(df, eq_s, ANALYSIS_START)
    hist_rows = history_html(history)

    trend = "上升" if score > df["score_final"].iloc[-5] else "下降"
    rsi_state = "超买" if rsi_v > 70 else ("超卖" if rsi_v < 30 else "中性")
    crisis_state = "风险偏高" if crisis_v > 0.8 else ("正常" if crisis_v > -0.8 else "极低")
    if pos == 1:
        commentary = f"模型当前<strong class='up'>做多</strong>，已持仓 {hold_weeks} 周。信号 S_t = {score:+.3f}（{trend}），RSI {rsi_v:.0f}（{rsi_state}），危机指数 {crisis_v:+.2f}（{crisis_state}）。"
    elif score > 0.05:
        commentary = f"模型当前<strong class='dn'>空仓</strong>，信号 S_t = {score:+.3f} 已上穿基础阈值 +0.05，入场仍需满足动态阈值与风险规则，继续<strong>等待</strong>。"
    elif score >= -0.05:
        commentary = f"模型当前<strong class='dn'>空仓</strong>，信号 S_t = {score:+.3f} 处于滞回带（±0.05）内，方向未明，保持<strong>观望</strong>。"
    else:
        commentary = f"模型当前<strong class='dn'>空仓</strong>，信号 S_t = {score:+.3f} 处于卖出区间，模型保持<strong>回避</strong>。"

    oil_pct = max(0, min(100, (oil_v+1)/2*100)); gap_pct = max(0, min(100, (gap_v+1)/2*100))

    tokens = {
        "DATE": date, "POS_CN": pos_cn, "POS_COLOR": pos_color, "ACTION": action,
        "GOLD_PRICE": f"{gp:,.2f}", "PRICE_CHANGE": f"{price_change:+.2f}",
        "PRICE_PCT": f"{price_pct:+.2%}", "PRICE_CLS": "up" if price_change >= 0 else "dn",
        "SOURCE_DATE": source_date, "WEEK_NOTE": "本周未完结" if date > source_date else "周度收盘",
        "HOLD_LABEL": "持仓时长" if pos == 1 else "空仓时长",
        "POSITION_EN": "LONG" if pos == 1 else "CASH", "STANCE": "持仓观察" if pos == 1 else "空仓观望",
        "GAUGE_STATE": "偏多区间" if score > 0.05 else ("偏空区间" if score < -0.05 else "中性区间"),
        "OIL_CONTRIB": f"{0.65*oil_v:+.3f}", "GAP_CONTRIB": f"{0.35*gap_v:+.3f}",
        "SAMPLE_START": fdate(recent.index[0]),
        "GOLD": f"{gp:.0f}", "SCORE_KPI": f"{score:+.3f}", "SCORE_CLS": "up" if score >= 0 else "dn",
        "RSI_V": f"{rsi_v:.0f}", "CRISIS_V": f"{crisis_v:+.2f}", "HOLD_W": str(hold_weeks),
        "LAST_CHG": last_change, "YTD": f"{ytd:+.1%}", "YTD_BH": f"{ytd_bh:+.1%}",
        "YTD_CLS": "up" if ytd >= 0 else "dn", "TREND": trend, "RSI_STATE": rsi_state,
        "CRISIS_STATE": crisis_state, "COMMENTARY": commentary,
        "OIL_CLS": "up" if oil_v >= 0 else "dn", "OIL_V": f"{oil_v:+.3f}", "GAP_V": f"{gap_v:+.3f}", "GAP_CLS": "up" if gap_v >= 0 else "dn",
        "OIL_PCT": f"{oil_pct:.0f}", "GAP_PCT": f"{gap_pct:.0f}", "HIST_ROWS": hist_rows, "HIST_COUNT": str(len(history)),
        "FULL_CAGR": f"{full_cagr:.1%}", "FULL_SHARPE": f"{full_sharpe:.2f}", "FULL_MDD": f"{full_mdd:.1%}",
        "GAUGE_VAL": f"{score:.4f}", "SCORE_PCT": f"{max(0, min(100, (score+1)/2*100)):.1f}",
        "LATEST_YEAR": str(df.index[-1].year),
        "GOLD_DATA": GOLD, "SCORE_DATA": SCORE, "RSI_DATA": RSI, "CRISIS_DATA": CRISIS,
        "EQS_DATA": EQS, "EQB_DATA": EQB, "DD_DATA": DD,
        "BUY_DATA": json.dumps(buy_pts), "SELL_DATA": json.dumps(sell_pts),
        "ANALYSIS_DATA": json.dumps(analysis_rows, allow_nan=False),
    }

    html = TEMPLATE
    for k, v in tokens.items():
        html = html.replace("{{" + k + "}}", v)

    assets = os.path.join(ROOT, "assets")
    if not os.path.isdir(assets):
        assets = os.path.join(ROOT, "deploy", "assets")
    shutil.copytree(assets,
                    os.path.join(os.path.dirname(OUT_HTML), "assets"), dirs_exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"监测站已生成: {OUT_HTML}")
    print(f"当前: {pos_cn} | 金价{gp:.0f} | score{score:+.3f} | RSI{rsi_v:.0f} | 危机{crisis_v:+.2f}")

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0c1013">
<meta name="description" content="AURUM 黄金全景研究台：价格、信号、收益与风险同屏分析，共享时间窗口与跨图准星，附完整因子构建方法。">
<title>AURUM · 黄金全景研究台</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%23111619'/%3E%3Ctext x='32' y='43' text-anchor='middle' font-family='Georgia' font-size='35' fill='%23dbc08a'%3EAu%3C/text%3E%3C/svg%3E">
<script src="assets/echarts-5.5.0/echarts.min.js" defer></script>
<style>
:root{color-scheme:dark;--bg:#0b0e10;--surface:#121719;--line:#d7dedc13;--text:#ebece7;--muted:#9ba6a5;--faint:#788583;--gold:#d8bb82;--green:#8ec9b1;--blue:#8dafc7;--red:#c88c81;--reading:#b6c0bc}
*{box-sizing:border-box;margin:0;padding:0}html{scroll-behavior:smooth;scroll-padding-top:78px}body{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','Helvetica Neue','PingFang SC','Microsoft YaHei',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;letter-spacing:.005em}body::before{content:'';position:absolute;inset:0 0 auto;height:1000px;z-index:-1;pointer-events:none;background:radial-gradient(ellipse at 75% -25%,#a9803930,transparent 58%),radial-gradient(ellipse at 4% 50%,#386e5f0e,transparent 48%)}a{color:inherit;text-decoration:none}button,input,select{font:inherit;color:inherit}button{cursor:pointer;border:0;background:none}button,a,summary{-webkit-tap-highlight-color:transparent}button:disabled,input:disabled,select:disabled{opacity:.4;cursor:not-allowed}[hidden]{display:none!important}button:focus-visible,a:focus-visible,summary:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--gold);outline-offset:4px}::selection{background:#6e5735;color:#fff}.shell{width:min(1536px,calc(100% - 64px));margin:0 auto}.mono{font-variant-numeric:tabular-nums;letter-spacing:-.02em}.sr-only,.chart-window{position:absolute!important;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}.up,.strategy-color{color:var(--green)}.dn{color:var(--red)}.signal-color{color:#a2b8ce}.skip-link{position:fixed;left:24px;top:-80px;z-index:200;background:var(--gold);color:var(--bg);padding:10px 16px;border-radius:8px}.skip-link:focus{top:12px}.topbar{height:65px;position:sticky;top:0;z-index:20;background:#0b0e1099;backdrop-filter:blur(24px) saturate(130%);-webkit-backdrop-filter:blur(24px) saturate(130%);border-bottom:1px solid #ffffff07;transition:background .3s}.topbar.is-scrolled{background:#101517dc}.topbar-inner{position:relative;height:64px;display:flex;align-items:center;justify-content:space-between;gap:30px}.brand{display:flex;align-items:center;gap:12px}.brand-icon{width:33px;height:36px;font:24px Georgia,serif;color:var(--gold);line-height:36px;position:relative;letter-spacing:-.08em}.brand-icon>span{position:absolute;font:7px -apple-system,sans-serif;letter-spacing:0;top:-2px;right:-1px;color:#8c7a56}.wordmark{font-size:13px;letter-spacing:.28em;font-weight:600}.wordmark>span{display:block;color:#8c9794;font-size:8px;letter-spacing:.18em;font-weight:400;margin-top:3px}.primary-nav{position:relative;display:flex;align-items:center;gap:6px;isolation:isolate;padding:4px;border-radius:999px;border:1px solid #ffffff09;background:#151b1b6b}.primary-nav::before{content:'';position:absolute;top:4px;left:0;height:28px;width:var(--nav-w,84px);transform:translateX(var(--nav-x,4px));border-radius:999px;background:linear-gradient(150deg,#d8bb821a,#d8bb8209);box-shadow:inset 0 1px 0 #f1e4cb12,inset 0 0 0 1px #d8bb8218;z-index:-1;transition:transform .45s cubic-bezier(.22,1,.36,1),width .45s cubic-bezier(.22,1,.36,1)}.primary-nav a{padding:6px 19px;font-size:12px;color:#98a29f;line-height:16px;transition:color .2s}.primary-nav a[aria-current]{color:#e3d2ae}.primary-nav a:hover{color:#eee6d8}.snapshot{display:flex;align-items:center;gap:8px;font-size:9px;color:#80908b}.snapshot b{font-size:10px;font-weight:400;color:#aab4ad}.status-dot{width:4px;height:4px;background:var(--green);border-radius:50%;box-shadow:0 0 9px #8ec9b150}.reading-progress{position:absolute;bottom:0;left:0;right:0;height:1px;transform:scaleX(var(--reading-progress,0));transform-origin:left;background:var(--gold);opacity:.65}.intro{height:99px;display:flex;align-items:center;gap:42px}.eyebrow,.overline{font-size:9px;letter-spacing:.17em;color:#a79269;font-weight:400}.intro .eyebrow{font-size:8px;letter-spacing:.2em;margin-bottom:7px}.intro h1{font-size:32px;font-weight:550;letter-spacing:-.04em;line-height:1.2}.intro h1 span{color:#c4b088}.intro-note{font-size:11px;line-height:1.9;color:#bdc3bc;padding-left:34px;border-left:1px solid #d8bb8222}.intro-note span{font-size:10px;color:#82918b}.intro-link{font-size:10px;color:#bda880;margin-left:auto;display:flex;gap:25px;align-items:center}.intro-link span{font-size:20px;font-weight:200;transition:transform .3s}.intro-link:hover span{transform:translate(2px,-2px)}
.analysis-workspace{min-width:0;grid-template-columns:minmax(0,1fr);height:clamp(584px,calc(100svh - 184px),890px);display:grid;grid-template-rows:49px 67px minmax(0,1fr) 88px;border:1px solid #ded7c622;border-radius:18px;background:linear-gradient(125deg,#ffffff02,transparent 50%),#111719;box-shadow:inset 0 1px 0 #f4e7ce07,0 28px 90px #0003;position:relative;isolation:isolate;outline:none}.workspace-toolbar{display:flex;align-items:center;justify-content:space-between;padding:0 20px;border-bottom:1px solid var(--line);gap:12px;background:linear-gradient(100deg,#d8bb8208,transparent 60%);border-radius:18px 18px 0 0}.workspace-title{display:flex;align-items:center;gap:10px}.workspace-title h2{font-size:13px;font-weight:500}.live-mark{width:5px;height:5px;background:var(--gold);border-radius:50%;box-shadow:0 0 12px #d8bb8230}.workspace-caption{font-size:8px;letter-spacing:.11em;color:#71817c;border-left:1px solid #ffffff12;padding-left:11px;margin-left:2px}.workspace-tools{display:flex;align-items:center;gap:17px}.sync-status{font-size:9px;color:#8fa498;display:flex;align-items:center;gap:5px}.sync-status i{width:11px;height:6px;border:1px solid #8fa49870;border-radius:6px;position:relative}.sync-status i::before{content:'';position:absolute;width:5px;height:4px;border:1px solid #8fa49870;border-radius:4px;left:2px;top:-3px}.icon-button{display:inline-flex;align-items:center;gap:6px;font-size:10px;color:#acb4ae;min-height:30px;padding:4px 5px;border-radius:5px;transition:background .2s,color .2s}.icon-button:hover{color:#e7d4ad;background:#d8bb820b}.icon-button svg{width:15px;height:15px}.icon-button kbd{font-size:9px;border:1px solid var(--line);padding:0 4px;color:var(--faint);border-radius:3px}.window-stats{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));align-items:center;border-bottom:1px solid var(--line);padding:0 20px;background:#ffffff01}.window-stats>div{padding-left:22px;border-left:1px solid #ffffff09}.window-stats>div:first-child{padding-left:0;border-left:0}.window-stats dt{color:#93a49b;font-size:10px;line-height:1.7}.window-stats dt span{font-size:8px;color:#83958a;margin-left:3px}.window-stats dd{font-size:23px;line-height:1.2;font-weight:450;margin-top:5px;letter-spacing:-.04em}.workspace-body{display:grid;grid-template-columns:188px minmax(0,1fr);min-height:0}.market-rail{border-right:1px solid var(--line);padding:19px 20px 12px;display:flex;flex-direction:column;min-height:0;background:linear-gradient(150deg,#d8bb8204,transparent 70%)}.rail-asset{display:flex;justify-content:space-between;align-items:baseline;gap:8px}.asset-symbol{font-size:12px;letter-spacing:.02em}.asset-symbol>span{font-size:9px;color:#86968d;margin-left:6px}.rail-unit{font-size:8px;color:#84958c}.rail-price{font-size:33px;font-weight:500;line-height:1.2;letter-spacing:-.05em;margin-top:7px}.rail-change{font-size:11px;margin-top:6px}.rail-change>span{font-size:8px;color:#7c8d83;margin-left:6px}.rail-position{margin-top:21px;border-top:1px solid var(--line);padding-top:15px}.rail-label{display:flex;align-items:center;justify-content:space-between;font-size:10px;color:#93a499}.rail-label>span{font-size:8px;color:#a08c68;letter-spacing:.05em}.rail-position p{font-size:18px;line-height:1.5;font-weight:450;margin-top:5px;display:flex;align-items:center;gap:10px}.position-dot{display:inline-block;width:4px;height:4px;border-radius:50%;background:var(--gold)}.rail-action{font-size:10px;color:#87978b;line-height:1.6;margin-top:3px;display:block}.rail-signal{margin-top:18px}.rail-label b{font-weight:400;font-size:11px}.signal-track{height:3px;background:linear-gradient(90deg,#c88c8133,#68786d33,#8ec9b133);margin-top:11px;position:relative;border-radius:2px}.signal-track::before{content:'';position:absolute;height:7px;width:1px;left:50%;top:-2px;background:#82908660}.signal-track>span{position:absolute;width:3px;height:9px;top:-3px;transform:translateX(-50%);border-radius:2px;background:#dbc395;box-shadow:0 0 7px #d8bb8230}.signal-extents{font-size:8px;display:flex;justify-content:space-between;color:#657a6d;margin-top:6px}.signal-extents span:nth-child(2){color:#93a18e;font-size:8px}.rail-facts{margin-top:17px;display:grid;gap:7px}.rail-facts>div{display:flex;align-items:baseline;justify-content:space-between;gap:5px}.rail-facts dt{font-size:10px;color:#7f9185}.rail-facts dd{font-size:11px;font-variant-numeric:tabular-nums;color:#c6cdbf}.rail-facts small{font-size:8px;color:#8d9d92;font-weight:400;margin-left:3px}.rail-asof{font-size:9px;line-height:1.8;color:#7d8f86;margin-top:auto;padding-top:13px}.comparison-area{display:grid;grid-template-rows:27px minmax(0,1fr);min-height:0;min-width:0}.probe-strip{display:flex;align-items:center;justify-content:space-between;padding:0 18px;border-bottom:1px solid #ffffff07;background:#080d0e25;gap:12px}.probe-strip>p{display:flex;gap:8px;align-items:center;color:#899b8d;font-size:8px}.probe-dot{width:3px;height:3px;border-radius:50%;background:#8da792}.probe-strip #probe-date{color:#c7cfbf;font-size:9px}.probe-help{font-size:9px;letter-spacing:.02em;color:#7f9287}.mobile-chart-tabs{display:none}.charts-matrix{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(2,minmax(0,1fr));min-height:0;min-width:0}.chart-cell{display:grid;grid-template-columns:minmax(0,1fr);grid-template-rows:35px minmax(0,1fr);min-height:0;min-width:0;position:relative;transition:background .2s}.chart-cell:nth-child(odd){border-right:1px solid var(--line)}.chart-cell:nth-child(-n+2){border-bottom:1px solid var(--line)}.chart-cell:hover{background:#d8bb8202}.chart-head{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:5px 18px 0;min-width:0}.chart-head h3{font-size:12px;font-weight:450;display:flex;align-items:baseline;gap:7px;white-space:nowrap}.chart-index{font-size:7px;font-variant-numeric:tabular-nums;color:#81978b}.chart-head small{font-size:9px;color:#84968d;font-weight:400;margin-left:1px}.chart-value{font-size:15px;font-weight:450;line-height:1;letter-spacing:-.025em}.chart-head-right{display:flex;align-items:center;gap:12px;min-width:0}.micro-legend{display:flex;align-items:center;gap:4px;font-size:9px;color:#889c8a;white-space:nowrap}.micro-legend i{width:6px;height:2px;margin-left:4px;display:inline-block}.strategy-dot{background:var(--green)}.benchmark-dot{background:var(--blue)}#equity-scale{font-size:8px;color:#91a28e;background:#0c1212;border:1px solid #b9cebb16;border-radius:4px;padding:3px;outline-offset:2px}.plot{min-width:0;min-height:0;width:100%;height:100%;touch-action:pan-y}.timeline-dock{position:relative;border-top:1px solid #ded7c61a;padding:9px 20px 7px;border-radius:0 0 18px 18px;background:linear-gradient(100deg,#d8bb8205,transparent),#101618}.timeline-controls{display:flex;align-items:center;justify-content:space-between;gap:14px;height:25px}.window-presets{display:flex;gap:3px}.window-presets button{font-size:11px;line-height:18px;color:#82978b;padding:1px 10px;border-radius:5px;transition:background .2s,color .2s;min-height:22px}.window-presets button[aria-pressed=true]{color:#dac59f;background:#d8bb8213;box-shadow:inset 0 0 0 1px #d8bb8214}.window-presets button:hover{color:#e0d5bd}.timeline-controls #analysis-period{font-size:10px;color:#95a38f;letter-spacing:.015em}.date-controls{display:flex;align-items:center;gap:16px}.date-popover{position:relative}.date-popover summary{list-style:none;font-size:9px;color:#a9b39f;cursor:pointer;display:flex;gap:12px;align-items:center;padding:4px}.date-popover summary::-webkit-details-marker{display:none}.date-popover[open] summary{color:var(--gold)}.date-form{position:absolute;right:0;bottom:32px;z-index:30;display:grid;grid-template-columns:1fr 1fr;gap:13px;background:#1a2221f5;backdrop-filter:blur(20px);border:1px solid #e7d4b131;border-radius:12px;padding:17px;box-shadow:0 15px 60px #0007;width:330px}.date-form label{font-size:10px;color:#9daf9d;display:grid;gap:8px;min-width:0}.date-form input{width:100%;min-width:0;background:#0b1211b0;border:1px solid #b4c5b322;border-radius:6px;padding:8px 7px;font-size:11px;color:#c8d1bd}.utility-button{font-size:11px;padding:8px 13px;border-radius:7px;border:1px solid #d8bb8233;color:#dcc9a4;background:#d8bb8214}.date-form button{grid-column:1/-1;justify-self:end}.reset-button{font-size:15px;color:#819480;display:flex;gap:5px;align-items:center}.reset-button span{font-size:9px}.reset-button:hover{color:#d5c29c}#timeline{min-width:0;height:34px;width:100%;touch-action:none;cursor:ew-resize;margin-top:4px}.timeline-ends{display:flex;justify-content:space-between;padding:0 9px;font-size:7px;color:#7a8d83;letter-spacing:.03em;line-height:10px}.analysis-error{position:static;grid-column:1/-1;background:#291e1d;border:1px solid #c88c8160;border-radius:8px;padding:10px 14px;font-size:11px;color:#e0b0a7;max-width:100%;}.workspace-footnote{display:flex;align-items:center;justify-content:space-between;gap:20px;font-size:8px;color:#809187;padding:10px 2px}.workspace-footnote>p:first-child{display:flex;align-items:center;gap:6px}.key-dot{width:3px;height:3px;background:#9c8a63;border-radius:50%;display:inline-block}.workspace-focused{overflow:hidden}.workspace-focused .analysis-workspace{position:fixed;inset:16px;width:auto;height:calc(100dvh - 32px);max-height:none;z-index:100;border-radius:16px;box-shadow:0 0 0 40px #0b0e10}.workspace-focused #workspace-focus{display:none}.workspace-focused .topbar{visibility:hidden}.workspace-focused .timeline-dock{border-radius:0 0 16px 16px}.workspace-focused .market-rail{padding-top:26px}.workspace-focused .rail-price{font-size:40px}.workspace-focused .rail-position{margin-top:30px}.workspace-focused .rail-signal{margin-top:28px}.workspace-focused .rail-facts{margin-top:30px;gap:16px}
.model-section{margin-top:100px;padding-top:35px;border-top:1px solid #d8bb8223}.section-heading{display:flex;justify-content:space-between;align-items:flex-end;gap:30px;margin-bottom:36px}.section-heading h2{font-size:32px;font-weight:500;letter-spacing:-.035em;margin-top:15px;line-height:1.4}.section-heading h2 span{color:#aa9a7b}.section-heading>p{color:#84998a;font-size:12px;line-height:1.9}.model-layout{display:grid;grid-template-columns:1fr 1fr;border:1px solid #d6dcca16;border-radius:16px;background:linear-gradient(115deg,#d8bb8204,transparent 60%),#111819;overflow:hidden}.factor-panel{padding:32px 36px;border-right:1px solid var(--line)}.panel-head{display:flex;justify-content:space-between;align-items:center;gap:16px}.panel-head h3{font-size:15px;font-weight:450}.factor{margin-top:30px}.factor-top{display:flex;justify-content:space-between;font-size:12px;align-items:center}.factor-top small{font-size:9px;color:#83998b;margin-left:8px}.factor-top strong{font-weight:400;font-size:16px}.factor-track{height:3px;background:#d8bb820b;margin:14px 0 9px;overflow:hidden}.factor-track>span{display:block;height:100%;background:linear-gradient(90deg,#7c6d4e,var(--gold))}.gap-track{background:#8dafc70c}.gap-track>span{background:linear-gradient(90deg,#4e606c,var(--blue))}.factor-detail{display:flex;justify-content:space-between;font-size:9px;color:#8d9e95}.score-equation{display:flex;justify-content:space-between;align-items:center;margin-top:30px;padding-top:20px;border-top:1px solid var(--line);gap:20px;font-size:10px;color:#97a89a}.score-equation strong{font-size:27px;font-weight:450;color:#ddc493}.text-link{display:inline-flex;align-items:center;gap:15px;font-size:11px;color:#c5ad7e;margin-top:24px}.text-link span{transition:transform .3s}.text-link:hover span{transform:translate(2px,-2px)}.observation-panel{padding:32px 36px;display:flex;flex-direction:column}.observation-panel h3{font-size:23px;font-weight:450;letter-spacing:-.02em;margin-top:19px}.observation-copy{font-size:14px;line-height:2;color:#bac4b5;max-width:570px;margin-top:17px}.observation-facts{display:flex;gap:48px;margin-top:29px}.observation-facts dt{font-size:10px;color:#7b9280}.observation-facts dd{font-size:22px;font-weight:400;margin-top:8px}.observation-facts small{font-size:9px;color:#859b8f;display:block;margin-top:7px}.observation-note{font-size:10px;color:#899c91;line-height:1.9;border-top:1px solid var(--line);padding-top:19px;margin-top:auto}.history-heading{display:flex;justify-content:space-between;align-items:baseline;margin:45px 0 20px}.history-heading h3{font-size:16px;font-weight:450}.history-heading h3 span{font-size:8px;letter-spacing:.14em;color:#879c90;margin-left:15px}.history-heading p{font-size:10px;color:#879d90}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;text-align:left}th,td{padding:15px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{font-size:10px;color:#708d75;font-weight:400}td{font-size:12px;color:#b4c1af;font-variant-numeric:tabular-nums}tr td:first-child,tr th:first-child{padding-left:0}tr td:last-child,tr th:last-child{text-align:right;padding-right:0}tbody tr:hover{background:#d8bb8203}.trade-tag{font-size:9px;display:inline-flex;align-items:center;gap:6px;padding:4px 9px;border-radius:4px}.trade-tag.buy{color:var(--green);background:#8ec9b109}.trade-tag.sell{color:var(--red);background:#c88c8109}.rules{margin-top:22px;border-bottom:1px solid var(--line)}.rules summary{cursor:pointer;display:flex;justify-content:space-between;list-style:none;padding:16px 0;font-size:11px;color:#9bad94}.rules summary::-webkit-details-marker{display:none}.rules summary>span{font-size:16px;color:#8c7e5f}.rules-body{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:28px;padding:12px 0 25px}.rules-body p{font-size:11px;line-height:1.95;color:#92a498}.research-notes{position:relative;margin-top:100px;padding-top:60px;border-top:1px solid #d8bb8226}.research-heading{max-width:850px;margin-bottom:54px}.research-heading h2{font-size:48px;line-height:1.3;font-weight:500;letter-spacing:-.04em;margin:20px 0 22px}.research-heading h2 span:first-child{color:#d4b981}.research-heading h2 span:last-child{color:#e6e8df}.research-heading>p:last-child{font-size:15px;color:#9fac9f;line-height:1.95;max-width:740px}.research-layout{display:grid;grid-template-columns:200px minmax(0,1fr);gap:70px}.research-sidebar{position:sticky;top:90px;align-self:start;padding-top:5px}.research-sidebar-label{font-size:9px;letter-spacing:.1em;color:#85978e;margin:0 0 16px 12px}.research-sidebar nav{display:flex;flex-direction:column;gap:4px}.research-sidebar nav a{min-height:40px;display:flex;align-items:center;gap:8px;font-size:11px;color:#94a79b;padding:9px 12px;border-radius:7px;transition:color .2s,background .2s}.research-sidebar nav a span{font-size:8px;color:#8a7854;min-width:20px}.research-sidebar nav a[aria-current]{background:linear-gradient(100deg,#d8bb8213,transparent);color:#d3bd94;box-shadow:inset 0 1px 0 #ffffff05}.research-sidebar nav a:hover{color:#c9caba;background:#ffffff03}.research-reading{border-top:1px solid var(--line);margin:21px 12px 0;padding-top:15px;color:#84978b;font-size:9px;line-height:2.2}.research-reading>span{display:block}.research-reading strong{font-weight:400;color:#ab966e;margin-left:7px}.research-content{max-width:810px;min-width:0}.method-article{border-bottom:1px solid var(--line);padding-bottom:42px;margin-bottom:42px}.method-article:last-child{margin-bottom:0}.method-article .method-index{font-size:9px;color:#ad9569;letter-spacing:.13em;font-family:'SFMono-Regular',Consolas,monospace;margin-bottom:16px;line-height:1.6}.method-article h3{font-size:26px;font-weight:500;letter-spacing:-.025em;margin-bottom:20px;line-height:1.5}.method-article p,.method-article li,.method-article dd{font-size:15px;line-height:1.95;color:var(--reading);overflow-wrap:anywhere}.method-article p+p{margin-top:16px}.method-article .method-lead{font-size:17px;color:#c8d1c0;margin-bottom:22px;line-height:1.9}.method-article strong{font-weight:500;color:#d9dfcf}.method-article a{color:#cbb385;border-bottom:1px solid #d8bb822c;transition:color .2s}.method-article a:hover{color:#f0dbaf;border-color:#d8bb8280}.method-article ol,.method-article ul{padding-left:23px;margin:20px 0}.method-article li+li{margin-top:13px}.method-steps li::marker{color:#ad9365}.method-formula{margin:24px 0;padding:26px 28px;max-width:100%;overflow-x:auto;overscroll-behavior-x:contain;border:1px solid #d8bb8213;border-radius:12px;background:linear-gradient(130deg,#d8bb8205,transparent),#101719;color:#e3cba1;font-family:'STIX Two Math','Cambria Math','Times New Roman',serif;font-size:20px;line-height:1.9;scrollbar-width:thin;scrollbar-color:#d8bb8240 transparent}.method-formula math{font-family:'STIX Two Math','Cambria Math','Times New Roman',serif;font-size:20px;overflow:visible}.method-formula>*+*{margin-top:15px}.method-formula .formula-caption{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;font-size:11px;color:#7f9980;line-height:1.9;margin-top:15px}.method-definitions{display:grid;grid-template-columns:100px minmax(0,1fr);gap:11px 18px;margin:20px 0 26px}.method-definitions>div{display:contents}.method-definitions dt{color:#c5ad80;font-family:Georgia,'PingFang SC',serif;font-size:14px;line-height:1.9}.method-definitions dd{font-size:13px;color:#879e85;margin:0}.method-note{border-left:2px solid #d8bb823c;padding:3px 0 3px 17px;margin:23px 0}.method-article p.method-note{font-size:13px;line-height:1.95;color:#8da186}.method-table-wrap{max-width:100%;overflow-x:auto}.method-table{table-layout:fixed;margin:20px 0 27px;min-width:0}.method-table caption{font-size:11px;color:#8ca286;text-align:left;padding-bottom:12px}.method-table th,.method-table td{padding:12px 10px!important;text-align:left!important;font-size:12px;line-height:1.9;white-space:normal;overflow-wrap:anywhere;vertical-align:top}.method-table th{font-size:11px;color:#b8a783}.method-table th:first-child{width:24%}.method-table td{color:#9caf93}.reference-list li span{display:block;color:#8fa293;font-size:12px;line-height:1.9;margin-top:4px}.method-article .method-implementation{font-size:12px;color:#8a9d93;border-top:1px solid var(--line);padding-top:22px;margin-top:30px}.research-bottom-link{display:inline-flex;align-items:center;gap:25px;font-size:12px;color:#bca372;margin-top:25px}.research-bottom-link span{transition:transform .3s}.research-bottom-link:hover span{transform:translateY(-3px)}footer{margin-top:90px!important;border-top:1px solid var(--line);padding:29px 0 34px;display:flex;justify-content:space-between;gap:30px;align-items:start}.footer-brand{font-size:15px;letter-spacing:.2em;color:#c2aa7b}.footer-brand span{font-size:9px;display:block;color:#879b8f;letter-spacing:.08em;margin-top:7px}.footer-note{font-size:9px;color:#84998d;line-height:2;text-align:right}.footer-note a{display:block;color:#ac976f;margin-top:8px}.chart-error{display:flex;align-items:center;justify-content:center;width:100%;height:100%;font-size:10px;color:#7c937b;text-align:center;padding:15px}
@media(min-width:1600px){.intro{height:112px}.intro h1{font-size:34px}.analysis-workspace{height:clamp(620px,calc(100svh - 198px),920px)}.workspace-body{grid-template-columns:210px minmax(0,1fr)}.market-rail{padding:24px}.rail-price{font-size:38px}.rail-position{margin-top:27px}.rail-signal{margin-top:25px}.rail-facts{margin-top:26px;gap:12px}.chart-head h3{font-size:12px}.chart-head small{font-size:9px}.chart-value{font-size:16px}}
@media(max-width:1200px){.shell{width:calc(100% - 40px)}.intro-note{padding-left:24px}.workspace-body{grid-template-columns:164px minmax(0,1fr)}.market-rail{padding:18px 15px 12px}.rail-price{font-size:30px}.rail-label{font-size:8px}.rail-facts dt{font-size:8px}.rail-position p{font-size:17px}.chart-head{padding-left:12px;padding-right:12px;gap:5px}.chart-head h3{gap:5px;font-size:10px}.chart-head small{font-size:7px}.micro-legend{display:none}.window-stats dd{font-size:22px}.window-stats>div{padding-left:16px}.research-layout{gap:45px;grid-template-columns:175px minmax(0,1fr)}.model-layout{grid-template-columns:1fr 1fr}.factor-panel,.observation-panel{padding:28px}.observation-copy{font-size:13px}.section-heading h2{font-size:29px}.timeline-controls #analysis-period{font-size:8px}}
@media(max-height:800px) and (min-width:901px){.market-rail{padding-top:14px}.rail-price{font-size:29px}.rail-position{margin-top:14px;padding-top:11px}.rail-signal{margin-top:13px}.rail-facts{margin-top:12px;gap:5px}.rail-asof{padding-top:8px;font-size:7px}.rail-position p{font-size:16px}.rail-label{font-size:8px}.rail-change{font-size:10px}.rail-facts small{font-size:7px}.rail-action{font-size:8px}.signal-extents{font-size:7px}}
@media(max-width:900px){.snapshot>span{display:none}.intro{gap:25px}.intro h1{font-size:27px}.intro-note{font-size:10px;padding-left:20px}.intro-note span{font-size:9px}.intro-link{display:none}.workspace-body{grid-template-columns:minmax(0,1fr)}.market-rail{display:none}.analysis-workspace{height:calc(100svh - 180px);min-height:600px}.window-stats dd{font-size:21px}.window-stats dt{font-size:8px}.window-stats>div{padding-left:15px}.probe-help{font-size:7px}.timeline-controls #analysis-period{font-size:8px}.date-controls{gap:9px}.reset-button span{display:none}.window-presets button{padding:1px 8px}.research-layout{grid-template-columns:minmax(0,1fr);gap:32px}.research-sidebar{position:relative;top:0;min-width:0}.research-sidebar-label{margin-left:0;margin-bottom:12px}.research-sidebar nav{flex-direction:row;overflow-x:auto;gap:7px;padding-bottom:10px;scrollbar-width:thin;scrollbar-color:#d8bb822b transparent}.research-sidebar nav a{flex-shrink:0;border:1px solid #ffffff0c;border-radius:999px;min-height:36px;font-size:10px;padding:8px 12px;background:#111819}.research-sidebar nav a span{min-width:auto;font-size:8px}.research-reading{display:flex;gap:16px;margin:6px 0 0;padding-top:8px}.research-content{max-width:100%}.research-heading h2{font-size:41px}.model-layout{grid-template-columns:1fr 1fr}.observation-facts{gap:25px}.observation-facts dd{font-size:18px}.observation-note{font-size:9px}.section-heading>p{font-size:10px}.method-article h3{font-size:25px}.rules-body{grid-template-columns:1fr;gap:15px}}
@media(max-width:620px){html{scroll-padding-top:66px}.shell{width:calc(100% - 28px)}.topbar{height:57px}.topbar-inner{height:56px;gap:15px}.brand{gap:8px}.brand-icon{font-size:23px;width:30px}.wordmark{font-size:11px;letter-spacing:.2em}.wordmark>span{font-size:7px;letter-spacing:.09em}.primary-nav{gap:1px;padding:3px}.primary-nav::before{top:3px;height:25px}.primary-nav a{font-size:9px;line-height:15px;padding:5px 9px;white-space:nowrap}.snapshot{display:none}.intro{height:99px;gap:12px;align-items:center;justify-content:space-between}.intro .eyebrow{font-size:6px;letter-spacing:.12em;margin-bottom:9px}.intro h1{font-size:26px;line-height:1.2;letter-spacing:-.05em}.intro-note{display:none}.analysis-workspace{height:646px;min-height:0;grid-template-rows:43px 93px minmax(0,1fr) 94px;border-radius:12px}.workspace-toolbar{padding:0 12px;border-radius:12px 12px 0 0;gap:8px}.workspace-title{gap:7px}.workspace-title h2{font-size:11px}.workspace-caption{display:none}.live-mark{width:4px;height:4px}.workspace-tools{gap:8px}.sync-status{display:none}.icon-button{font-size:9px;gap:4px;padding:4px;min-height:30px}.icon-button svg{width:13px;height:13px}.window-stats{grid-template-columns:repeat(3,minmax(0,1fr));padding:10px 12px;gap:10px 0}.window-stats>div{padding-left:12px}.window-stats>div:nth-child(4){padding-left:0;border-left:0}.window-stats dt{font-size:8px;line-height:1.5;white-space:nowrap}.window-stats dt span{font-size:7px;margin-left:2px}.window-stats dd{font-size:19px;line-height:1.2;margin-top:3px}.comparison-area{grid-template-rows:50px minmax(0,1fr)}.probe-strip{flex-wrap:wrap;gap:0;padding:5px 10px;align-content:space-between}.probe-strip>p{font-size:7px;gap:6px}.probe-strip #probe-date{font-size:8px}.probe-help{display:none}.mobile-chart-tabs{display:flex;gap:3px;width:100%;justify-content:space-between}.mobile-chart-tabs button{font-size:8px;color:#708e74;border-radius:4px;padding:3px 9px;line-height:1.4;min-height:20px;flex:1}.mobile-chart-tabs button[aria-pressed=true]{background:#d8bb8211;color:#d8c193}.chart-head{padding:5px 9px 0;align-items:flex-start;gap:4px}.chart-cell{grid-template-rows:31px minmax(0,1fr)}.chart-head h3{font-size:9px;gap:4px;line-height:1.4;flex-wrap:wrap}.chart-head small{font-size:6px;display:block;flex-basis:100%;margin:0 0 0 13px;line-height:1.2}.chart-index{font-size:6px}.chart-value{font-size:11px;margin-top:2px}.chart-head-right{gap:4px}#equity-scale{font-size:7px;padding:2px;max-width:36px}.timeline-dock{padding:9px 10px 6px;border-radius:0 0 12px 12px}.timeline-controls{height:37px;gap:4px;flex-wrap:wrap;align-content:space-between}.window-presets{gap:2px;order:1}.window-presets button{font-size:8px;padding:1px 7px;line-height:17px;min-height:20px}.timeline-controls #analysis-period{font-size:7px;order:3;width:100%;letter-spacing:.015em;margin-top:1px}.date-controls{order:2;gap:8px}.date-popover summary{font-size:8px;gap:5px;padding:1px}.reset-button{font-size:13px}.date-form{width:270px;padding:14px;gap:10px;right:-20px;bottom:27px}.date-form label{font-size:9px}.date-form input{font-size:10px;min-height:36px}.utility-button{font-size:10px;padding:6px 10px}#timeline{height:29px;margin-top:4px}.timeline-ends{font-size:6px;line-height:9px}.analysis-error{font-size:10px;bottom:100px;right:10px;padding:9px 12px}.workspace-footnote{display:block;font-size:7px;line-height:1.8;padding:9px 1px;color:#5c785f}.workspace-footnote>p:nth-child(2){margin-top:4px}.workspace-focused .analysis-workspace{inset:7px;height:calc(100dvh - 14px);border-radius:12px;min-height:0}.workspace-focused .chart-cell{grid-template-rows:34px minmax(0,1fr)}.charts-matrix[data-view]:not([data-view=all]){grid-template-columns:minmax(0,1fr);grid-template-rows:minmax(0,1fr)}.charts-matrix[data-view=c1] [data-cell]:not([data-cell=c1]),.charts-matrix[data-view=c2] [data-cell]:not([data-cell=c2]),.charts-matrix[data-view=c3] [data-cell]:not([data-cell=c3]),.charts-matrix[data-view=c4] [data-cell]:not([data-cell=c4]){display:none}.charts-matrix[data-view]:not([data-view=all]) .chart-cell{border:0}.charts-matrix[data-view]:not([data-view=all]) .chart-head{padding:9px 14px 0}.charts-matrix[data-view]:not([data-view=all]) .chart-head h3{font-size:11px}.charts-matrix[data-view]:not([data-view=all]) .chart-head small{display:inline;flex-basis:auto;font-size:8px;margin-left:5px}.charts-matrix[data-view]:not([data-view=all]) .chart-value{font-size:16px}.model-section{margin-top:60px;padding-top:28px}.section-heading{display:block;margin-bottom:26px}.section-heading .eyebrow{font-size:8px}.section-heading h2{font-size:27px;margin-top:13px}.section-heading>p{font-size:11px;margin-top:17px;line-height:1.9}.model-layout{grid-template-columns:minmax(0,1fr);border-radius:12px}.factor-panel{border-right:0;border-bottom:1px solid var(--line);padding:25px}.observation-panel{padding:25px}.panel-head h3{font-size:14px}.overline{font-size:8px}.factor-top{font-size:11px}.factor-top small{font-size:8px;margin-left:6px}.factor-top strong{font-size:15px}.factor-detail{font-size:9px}.score-equation{font-size:9px;gap:10px}.score-equation strong{font-size:25px}.observation-copy{font-size:14px;line-height:2}.observation-facts dd{font-size:21px}.observation-facts{gap:32px}.observation-note{margin-top:25px;font-size:10px}.history-heading{margin-top:35px;display:block}.history-heading h3{font-size:15px}.history-heading h3 span{font-size:7px;margin-left:12px}.history-heading p{font-size:9px;margin-top:8px}th{font-size:8px}td{font-size:10px}th,td{padding:14px 7px}.trade-tag{font-size:8px;padding:3px 5px}.rules summary{font-size:10px}.rules-body p{font-size:11px}.research-notes{margin-top:60px;padding-top:36px}.research-heading{margin-bottom:29px}.research-heading .eyebrow{font-size:8px}.research-heading h2{font-size:33px;margin:17px 0 19px;line-height:1.4}.research-heading>p:last-child{font-size:14px;line-height:1.95}.research-layout{gap:29px}.research-sidebar-label{font-size:8px}.research-sidebar nav a{font-size:10px}.research-reading{font-size:8px}.method-article{padding-bottom:34px;margin-bottom:34px}.method-article .method-index{font-size:8px;margin-bottom:14px}.method-article h3{font-size:23px;line-height:1.5;margin-bottom:18px}.method-article p,.method-article li{font-size:14px;line-height:1.95}.method-article .method-lead{font-size:15px;line-height:1.95}.method-formula{padding:21px 17px;font-size:18px;margin:23px 0}.method-formula math{font-size:18px}.method-formula .formula-caption{font-size:10px}.method-definitions{grid-template-columns:69px minmax(0,1fr);gap:12px}.method-definitions dt{font-size:13px}.method-definitions dd{font-size:12px}.method-article p.method-note{font-size:12px}.method-note{padding-left:14px}.method-table th,.method-table td{font-size:11px;padding:10px 7px!important}.method-table th{font-size:10px}.reference-list li span{font-size:11px}.method-article .method-implementation{font-size:11px}footer{margin-top:60px!important;display:block;padding:25px 0}.footer-brand{font-size:13px}.footer-note{text-align:left;font-size:9px;margin-top:22px}}
@media(max-width:360px){.shell{width:calc(100% - 20px)}.brand{gap:5px}.wordmark{font-size:10px}.wordmark>span{font-size:6px}.primary-nav a{font-size:8px;padding:5px 7px}.primary-nav{gap:0}.topbar-inner{gap:8px}.intro h1{font-size:25px}.intro .eyebrow{font-size:5px}.workspace-tools{gap:5px}.icon-button{font-size:8px}.icon-button svg{width:12px}.workspace-title h2{font-size:10px}.window-stats dd{font-size:18px}.window-stats>div{padding-left:9px}.window-stats dt{font-size:7px}.window-stats dt span{font-size:6px}.chart-head{padding-left:7px;padding-right:7px}.chart-value{font-size:10px}.chart-head h3{font-size:8px}.chart-head small{font-size:6px}.chart-head-right{gap:3px}#equity-scale{max-width:32px;font-size:6px}.window-presets button{padding:1px 6px}.date-controls{gap:5px}.date-popover summary{font-size:7px}.timeline-controls #analysis-period{font-size:6px}.model-layout .factor-panel,.observation-panel{padding:22px}.section-heading h2{font-size:25px}.research-heading h2{font-size:29px}.method-article h3{font-size:21px}.research-reading{gap:10px}.history-heading h3 span{font-size:6px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*::before,*::after{animation:none!important;transition:none!important}.intro-link:hover span,.text-link:hover span,.research-bottom-link:hover span{transform:none}}@media(prefers-reduced-transparency:reduce){.topbar,.topbar.is-scrolled{background:#101517;backdrop-filter:none;-webkit-backdrop-filter:none}.date-form{background:#1a2221;backdrop-filter:none;-webkit-backdrop-filter:none}}

.analysis-workspace::before{content:'';position:absolute;top:-1px;left:18px;right:18px;height:1px;background:linear-gradient(90deg,#e6d3a438,#e6d3a40b 65%,transparent);pointer-events:none}.market-rail{background:radial-gradient(ellipse at 0 0,#d8bb820a,transparent 67%)}
@media(max-height:820px) and (min-width:901px){.intro{height:63px}.intro .eyebrow{font-size:7px;margin-bottom:5px}.intro h1{font-size:29px}.analysis-workspace{height:calc(100svh - 148px);min-height:584px;grid-template-rows:44px 60px minmax(0,1fr) 80px}.timeline-dock{padding-top:6px;padding-bottom:3px}.timeline-controls{height:23px}.rail-asof{font-size:8px}.rail-label{font-size:9px}.rail-facts dt{font-size:9px}}

.mobile-marketline,.mobile-step-controls{display:none}
@media(max-width:620px){.intro{position:relative;align-items:flex-start;padding-top:17px}.intro h1{font-size:25px}.intro .eyebrow{font-size:6px;margin-bottom:6px}.mobile-marketline{display:flex;position:absolute;bottom:11px;left:1px;align-items:baseline;gap:9px;font-size:10px;color:#8e9e93}.mobile-marketline b{font-size:12px;font-weight:450;color:#ced3c5}.mobile-marketline>span:last-child{font-size:9px;color:#9b9d88}.analysis-workspace{height:calc(100svh - 164px);min-height:460px;grid-template-rows:43px 93px minmax(0,1fr) 112px}.comparison-area{grid-template-rows:69px minmax(0,1fr)}.probe-strip{padding:6px 10px 3px}.probe-strip>p{font-size:8px}.probe-strip #probe-date{font-size:10px}.mobile-chart-tabs{gap:3px}.mobile-chart-tabs button{font-size:10px;min-height:36px;padding:8px 6px;flex:1;white-space:nowrap;border-radius:6px}.mobile-chart-tabs button:first-child{flex:1.6;box-shadow:inset 0 0 0 1px #d8bb821c;color:#c4b08b}.mobile-chart-tabs button[aria-pressed=true]{color:#e2cea6;background:#d8bb821a}.timeline-controls{height:43px}.window-presets button{font-size:10px;padding:3px 7px;min-height:28px;line-height:20px}.timeline-controls #analysis-period{font-size:8px;line-height:12px}.date-controls{gap:5px}.date-popover summary{font-size:9px;min-height:30px}.reset-button{min-height:30px;min-width:21px;justify-content:center}.mobile-step-controls{display:flex;align-items:center;border:1px solid #d8bb8214;border-radius:5px;overflow:hidden}.mobile-step-controls button{font-family:Georgia,serif;font-size:21px;line-height:28px;min-height:30px;width:25px;color:#b7aa8b}.mobile-step-controls button+button{border-left:1px solid #d8bb8214}.mobile-step-controls button:disabled{opacity:.2}.date-popover summary{gap:4px}#timeline{height:44px;margin-top:3px}.timeline-ends{font-size:7px}.date-form{bottom:37px;right:-20px}.chart-cell{grid-template-rows:35px minmax(0,1fr)}.charts-matrix[data-view]:not([data-view=all]) .chart-cell{grid-template-rows:43px minmax(0,1fr)}.charts-matrix[data-view]:not([data-view=all]) .chart-head h3{font-size:12px}.charts-matrix[data-view]:not([data-view=all]) .chart-head small{font-size:9px}.charts-matrix[data-view]:not([data-view=all]) .chart-value{font-size:19px}.charts-matrix[data-view]:not([data-view=all]) #equity-scale{font-size:9px;max-width:48px;padding:3px}.workspace-focused .analysis-workspace{height:calc(100dvh - 14px);min-height:0}}
@media(max-width:360px){.window-presets{gap:0}.window-presets button{font-size:9px;padding:3px 6px}.date-popover summary{font-size:8px}.mobile-step-controls button{width:21px}.timeline-controls #analysis-period{font-size:7px}.mobile-chart-tabs button{font-size:9px;padding:8px 4px}.mobile-marketline{font-size:9px}.mobile-marketline b{font-size:11px}}

.exit-phone,.nav-mobile-key{display:none}.benchmark-color{color:var(--blue)}
@media(max-width:620px){.exit-phone{display:inline}.exit-desktop,#workspace-exit kbd{display:none}.charts-matrix[data-view=all] #equity-scale{display:none}.charts-matrix[data-view=all] .nav-unit{display:none}.charts-matrix[data-view=all] .nav-mobile-key{display:inline;font-size:7px}.charts-matrix[data-view]:not([data-view=all]) .micro-legend{display:flex;font-size:8px}.charts-matrix[data-view]:not([data-view=all]) #equity-scale{max-width:54px;min-width:48px;font-size:10px}}

.analysis-workspace{grid-template-rows:49px 67px minmax(0,1fr) 110px}.timeline-instruction{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:10px;color:#bca77e;margin:6px 5px 3px;line-height:16px}.timeline-label{display:flex;align-items:center;gap:7px;color:#ddc69c;font-weight:500}.timeline-label svg{width:24px;height:14px;flex-shrink:0}.timeline-instruction>span:last-child{font-size:9px;color:#9eac9e}#timeline{border-radius:6px;box-shadow:inset 0 0 0 1px #d8bb8212;background:linear-gradient(180deg,#d8bb8209,#d8bb8202);cursor:ew-resize}.timeline-ends{color:#91a08d}
@media(max-height:820px) and (min-width:901px){.analysis-workspace{grid-template-rows:44px 60px minmax(0,1fr) 102px}}
@media(max-width:620px){.analysis-workspace{grid-template-rows:43px 93px minmax(0,1fr) 134px}.timeline-instruction{margin-top:6px;margin-left:3px;margin-right:3px;gap:7px;font-size:10px;line-height:15px}.timeline-label{gap:5px;font-size:10px}.timeline-label svg{width:20px;height:13px}.timeline-instruction>span:last-child{font-size:9px;white-space:nowrap}#timeline{background:linear-gradient(180deg,#d8bb8212,#d8bb8206)}}
@media(max-width:360px){.timeline-instruction{font-size:9px;gap:5px}.timeline-label{font-size:9px;gap:4px}.timeline-label svg{width:17px}.timeline-instruction>span:last-child{font-size:9px}}

.method-cards .research-layout{grid-template-columns:minmax(0,1fr);gap:26px}.method-cards .research-sidebar{position:relative;top:0;padding:0;min-width:0}.method-cards .research-sidebar-label{margin:0 0 16px;color:#a8b3a3;font-size:10px}.method-cards .research-sidebar nav{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:9px;overflow:visible;padding:0;scrollbar-width:thin;isolation:isolate}.method-cards .research-sidebar nav a{border:1px solid #c6ceba1b;border-radius:11px;min-height:58px;display:flex;align-items:flex-start;justify-content:center;flex-direction:column;gap:5px;padding:10px 13px;font-size:11px;line-height:1.6;color:#96a790;background:linear-gradient(135deg,#ffffff02,transparent),#101718;transition:background .25s,border-color .25s,box-shadow .25s,color .25s}.method-cards .research-sidebar nav a span{color:#7a8c76;min-width:0;font-size:8px;letter-spacing:.08em}.method-cards .research-sidebar nav a[aria-selected=true]{border-color:#d8bb825c;background:radial-gradient(ellipse at 0 0,#d8bb8217,transparent 85%),#171e1d;color:#e5cea1;box-shadow:inset 0 1px 0 #f6dba521,0 7px 20px #0002}.method-cards .research-sidebar nav a[aria-selected=true] span{color:#c3a979}.method-cards .research-sidebar nav a:hover{border-color:#d8bb8247;color:#d0c9b3}.method-cards .research-reading{margin:16px 0 0;padding:0;border:0;display:flex;justify-content:space-between;gap:20px;font-size:9px;color:#7d927d}.method-cards .research-reading strong{color:#c7ae7c;font-variant-numeric:tabular-nums}.method-cards .research-content{position:relative;max-width:none;border:1px solid #d8bb8226;border-radius:20px;background:radial-gradient(ellipse at 100% 0,#d8bb8206,transparent 60%),#101719;box-shadow:inset 0 1px 0 #ffffff06;padding:40px 48px;min-height:550px;min-width:0}.method-cards .method-article{max-width:860px;margin:0 auto;padding:0;border:0}.method-cards .method-article h3{font-size:29px;margin-bottom:22px}.method-cards .method-formula{background:linear-gradient(130deg,#d8bb8209,transparent),#0c1315;border-color:#d8bb821b}.method-cards .method-article:focus-visible{outline:1px solid #d8bb8270;outline-offset:12px}.method-cards .research-bottom-link{margin-top:26px}
@media(max-width:1100px){.method-cards .research-sidebar nav{grid-template-columns:repeat(4,minmax(0,1fr))}.method-cards .research-content{padding:35px}}
@media(max-width:620px){.method-cards .research-layout{gap:19px}.method-cards .research-sidebar-label{font-size:9px;margin-bottom:13px}.method-cards .research-sidebar nav{display:flex;flex-direction:row;gap:8px;overflow-x:auto;padding:0 1px 9px;scroll-snap-type:x proximity}.method-cards .research-sidebar nav a{width:144px;min-width:144px;min-height:67px;flex-shrink:0;white-space:normal;font-size:12px;padding:11px 14px;border-radius:10px;scroll-snap-align:center}.method-cards .research-sidebar nav a span{font-size:9px}.method-cards .research-reading{font-size:8px;margin-top:7px;gap:10px}.method-cards .research-content{padding:26px 20px;min-height:470px;border-radius:14px}.method-cards .method-article h3{font-size:23px;margin-bottom:20px}.method-cards .method-article .method-index{font-size:8px}.method-cards .method-article .method-lead{font-size:15px}.method-cards .method-formula{padding:21px 15px;margin-left:-3px;margin-right:-3px;max-width:calc(100% + 6px)}.method-cards .method-article p,.method-cards .method-article li{font-size:14px}.method-cards .method-definitions{grid-template-columns:62px minmax(0,1fr);gap:12px}.method-cards .method-definitions dd{font-size:12px}}

.history-scroll-hint{display:none}.history-table{border-top:1px solid #d8bb8223}.history-table th{padding-top:18px;padding-bottom:18px;color:#9bad97}.history-return>span{font-size:15px;letter-spacing:-.025em}.history-period{display:block;font-size:9px;line-height:1.8;color:#7f9585;margin-top:5px}.history-return .neutral{color:#b9c4b1}.history-actions{display:flex;justify-content:space-between;align-items:flex-start;gap:35px;padding-top:21px}.history-note{font-size:10px;line-height:1.95;color:#92a58f;max-width:710px}.history-note span{display:block;color:#728b79;margin-top:5px}.history-toggle{display:flex;align-items:center;gap:15px;min-height:46px;flex-shrink:0;border:1px solid #d8bb8240;border-radius:9px;padding:11px 16px;color:#d6bd8c;font:inherit;font-size:11px;background:linear-gradient(135deg,#d8bb8211,transparent),#141c1b;box-shadow:inset 0 1px 0 #ffffff06;cursor:pointer;transition:border-color .2s,background .2s}.history-toggle:hover{border-color:#d8bb8277;background-color:#1c2420}.history-toggle small{font-size:9px;color:#84977f}.history-toggle svg{width:14px;height:14px;transition:transform .2s}.history-toggle[aria-expanded=true] svg{transform:rotate(180deg)}.history-toggle:focus-visible{outline:2px solid #d8bb82;outline-offset:4px}
@media(max-width:620px){.history-scroll-hint{display:flex;justify-content:space-between;align-items:center;font-size:10px;line-height:1.9;color:#b49f77;margin:0 0 13px}.history-scroll-hint span{font-size:17px}.history-heading{align-items:flex-start;gap:12px}.history-heading h3 span{display:block;margin:8px 0 0;font-size:7px}.history-heading p{font-size:9px;text-align:right;line-height:1.9}.history-table table{min-width:430px}.history-table th,.history-table td{padding:14px 10px}.history-table th{font-size:9px}.history-table td{font-size:11px}.history-table tr>*:first-child{padding-left:0;position:sticky;left:0;z-index:1;background:var(--bg);box-shadow:8px 0 12px -10px #000}.history-table tr>*:last-child{padding-right:0}.history-return>span{font-size:14px}.history-period{font-size:8px}.history-actions{flex-direction:column;gap:19px;padding-top:18px}.history-note{font-size:10px}.history-toggle{width:100%;justify-content:center;min-height:48px;font-size:12px}.history-toggle small{font-size:10px}.history-table{background:linear-gradient(90deg,transparent 93%,#0b12117a)}}
</style>
</head>
<body>
<a class="skip-link" href="#overview">跳至主要内容</a>
<header class="topbar">
  <div class="shell topbar-inner">
    <a class="brand" href="#overview" aria-label="AURUM 黄金择时首页"><span class="brand-icon" aria-hidden="true">Au<span>79</span></span><span class="wordmark">AURUM<span>黄金策略研究</span></span></a>
    <nav class="primary-nav" aria-label="主导航"><a href="#overview" aria-current="location">全景分析</a><a href="#methodology">模型透视</a><a href="#research-notes">研究方法</a></nav>
    <div class="snapshot"><i class="status-dot" aria-hidden="true"></i><span>价格交易日</span><b class="mono">{{SOURCE_DATE}}</b></div>
    <div class="reading-progress" aria-hidden="true"></div>
  </div>
</header>
<main class="shell" id="overview">
  <section class="intro" aria-labelledby="page-title"><div><p class="eyebrow">GOLD / A MULTIDIMENSIONAL PERSPECTIVE</p><h1 id="page-title">黄金，<span>置于全局。</span></h1></div><p class="intro-note">价格的变化，只是故事的一面。<br><span>把信号、收益与风险，放在同一个时间里观察。</span></p><p class="mobile-marketline">GLD <b class="mono">{{GOLD_PRICE}}</b><span class="{{PRICE_CLS}}">{{PRICE_PCT}}</span><span>· {{STANCE}}</span></p><a class="intro-link" href="#research-notes">探索研究方法 <span aria-hidden="true">↗</span></a></section>

  <section class="analysis-workspace" id="analysis-workspace" aria-labelledby="analysis-title" tabindex="-1">
    <div class="workspace-toolbar"><div class="workspace-title"><span class="live-mark" aria-hidden="true"></span><h2 id="analysis-title">全景工作台</h2><span class="workspace-caption">GLD / WEEKLY</span></div><div class="workspace-tools"><span class="sync-status"><i aria-hidden="true"></i>四图同窗</span><button class="icon-button" id="export-csv" type="button" aria-label="导出区间 CSV"><svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M10 3v9m-3-3 3 3 3-3M4 13v4h12v-4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg><span>导出</span></button><button class="icon-button" id="workspace-focus" type="button" aria-pressed="false"><svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M7 3H3v4m10-4h4v4M3 13v4h4m10-4v4h-4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg><span>沉浸分析</span></button><button class="icon-button" id="workspace-exit" type="button" hidden><span aria-hidden="true">×</span><span class="exit-desktop">退出沉浸</span><span class="exit-phone">返回精读</span><kbd>esc</kbd></button></div></div>
    <dl class="window-stats"><div><dt>策略收益</dt><dd class="mono" data-stat="strategyReturn">—</dd></div><div><dt>基准收益</dt><dd class="mono" data-stat="benchmarkReturn">—</dd></div><div><dt>超额收益 <span>百分点</span></dt><dd class="mono" data-stat="excessReturn">—</dd></div><div><dt>最大回撤</dt><dd class="mono dn" data-stat="maxDrawdown">—</dd></div><div><dt>年化波动</dt><dd class="mono" data-stat="volatility">—</dd></div><div><dt>夏普比率</dt><dd class="mono" data-stat="sharpe">—</dd></div></dl>
    <div class="workspace-body">
      <aside class="market-rail" aria-label="最新市场快照"><div class="rail-asset"><span class="asset-symbol">GLD<span>黄金 ETF</span></span><span class="rail-unit">USD / 份</span></div><div class="rail-price mono">{{GOLD_PRICE}}</div><div class="rail-change mono {{PRICE_CLS}}">{{PRICE_PCT}}<span>较上周</span></div><div class="rail-position"><div class="rail-label">最新模型仓位 <span>{{POSITION_EN}}</span></div><p>{{STANCE}}<span class="position-dot" aria-hidden="true"></span></p><span class="rail-action">{{ACTION}}</span></div><div class="rail-signal"><div class="rail-label">复合信号 <b class="mono {{SCORE_CLS}}">{{SCORE_KPI}}</b></div><div class="signal-track"><span style="left:{{SCORE_PCT}}%"></span></div><div class="signal-extents"><span>−1</span><span>{{GAUGE_STATE}}</span><span>+1</span></div></div><dl class="rail-facts"><div><dt>RSI · 14 周</dt><dd>{{RSI_V}} <small>{{RSI_STATE}}</small></dd></div><div><dt>危机指数</dt><dd>{{CRISIS_V}} <small>{{CRISIS_STATE}}</small></dd></div><div><dt>{{HOLD_LABEL}}</dt><dd>{{HOLD_W}} <small>周</small></dd></div></dl><p class="rail-asof">最新价格 {{SOURCE_DATE}}<br>周度标签 {{DATE}} · {{WEEK_NOTE}}</p></aside>
      <div class="comparison-area">
        <div class="probe-strip"><p><span class="probe-dot" aria-hidden="true"></span><span id="probe-date" class="mono">{{DATE}}</span><span id="probe-position">最新周 · {{POS_CN}}</span></p><span class="probe-help">移动准星，四图同日对照</span><div class="mobile-chart-tabs" role="group" aria-label="图表阅读视图"><button type="button" data-chart-view="all" aria-pressed="true">四图对照</button><button type="button" data-chart-view="c1" aria-pressed="false">价格</button><button type="button" data-chart-view="c3" aria-pressed="false">净值</button><button type="button" data-chart-view="c2" aria-pressed="false">信号</button><button type="button" data-chart-view="c4" aria-pressed="false">回撤</button></div></div>
        <div class="charts-matrix" data-view="all">
          <article class="chart-cell" data-cell="c1"><div class="chart-head"><h3><span class="chart-index">01</span>黄金价格 <small>USD / 份</small></h3><strong class="chart-value mono" data-probe="price">{{GOLD_PRICE}}</strong></div><div class="plot" id="c1" role="img" aria-label="GLD 周度价格与历史买卖信号，窗口由下方统一时间条控制"></div><p class="chart-window" id="window-c1"></p></article>
          <article class="chart-cell" data-cell="c3"><div class="chart-head"><h3><span class="chart-index">02</span>策略与基准 <small><span class="nav-unit">起点 100</span><span class="nav-mobile-key"><span class="strategy-color">策略</span> / <span class="benchmark-color">基准</span></span></small></h3><div class="chart-head-right"><span class="micro-legend"><i class="strategy-dot"></i>策略<i class="benchmark-dot"></i>基准</span><select id="equity-scale" aria-label="净值坐标"><option value="value">线性</option><option value="log">对数</option></select><strong class="chart-value mono strategy-color" data-probe="equity">—</strong></div></div><div class="plot" id="c3" role="img" aria-label="策略与买入持有净值，窗口起点归一为100"></div><p class="chart-window" id="window-c3"></p></article>
          <article class="chart-cell" data-cell="c2"><div class="chart-head"><h3><span class="chart-index">03</span>复合信号 <small>阈值 ±0.05</small></h3><strong class="chart-value mono signal-color" data-probe="score">{{SCORE_KPI}}</strong><span id="window-signal-value" class="sr-only">{{SCORE_KPI}}</span></div><div class="plot" id="c2" role="img" aria-label="复合信号及正负0.05基础阈值"></div><p class="chart-window" id="window-c2"></p></article>
          <article class="chart-cell" data-cell="c4"><div class="chart-head"><h3><span class="chart-index">04</span>策略回撤 <small>区间内高点</small></h3><strong class="chart-value mono dn" data-probe="drawdown">—</strong><span id="drawdown-window-value" class="sr-only"></span></div><div class="plot" id="c4" role="img" aria-label="从统一窗口起点重新计算的策略回撤"></div><p class="chart-window" id="window-c4"></p></article>
        </div>
      </div>
    </div>
    <div class="timeline-dock"><div class="timeline-controls"><div class="window-presets" role="group" aria-label="统一分析窗口"><button type="button" data-window="6m" aria-pressed="false">半年</button><button type="button" data-window="1y" aria-pressed="false">1年</button><button type="button" data-window="2y" aria-pressed="false">2年</button><button type="button" data-window="5y" aria-pressed="false">5年</button><button type="button" data-window="all" aria-pressed="true">2016起</button></div><p id="analysis-period" class="mono">正在读取周度数据…</p><div class="date-controls"><div class="mobile-step-controls"><button type="button" data-shift-window="-1" aria-label="向前平移一个窗口">‹</button><button type="button" data-shift-window="1" aria-label="向后平移一个窗口">›</button></div><details class="date-popover"><summary>精确日期 <span aria-hidden="true">⌄</span></summary><form id="analysis-form" class="date-form"><label for="analysis-start">开始<input type="date" id="analysis-start" required aria-describedby="analysis-error"></label><label for="analysis-end">结束<input type="date" id="analysis-end" required aria-describedby="analysis-error"></label><p id="analysis-error" class="analysis-error" role="alert" hidden></p><button class="utility-button primary" type="submit">应用区间</button></form></details><button class="reset-button" type="button" id="reset-window" aria-label="重置区间">↺ <span>重置</span></button></div></div><div class="timeline-instruction" id="timeline-instruction"><span class="timeline-label"><svg viewBox="0 0 24 14" fill="none" aria-hidden="true"><rect x="1" y="2" width="4" height="10" rx="1" stroke="currentColor"/><rect x="19" y="2" width="4" height="10" rx="1" stroke="currentColor"/><path d="M7 7h10m-8-2L7 7l2 2m6-4 2 2-2 2" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/></svg>拖动时间窗口</span><span id="timeline-hint">拖动两端握柄，缩小观察窗口</span></div><div id="timeline" role="img" aria-describedby="timeline-instruction" aria-label="四张图共用的时间选择条，拖动两端改变区间，拖动中间平移"></div><div class="timeline-ends mono"><span>2016</span><span>2018</span><span>2020</span><span>2022</span><span>2024</span><span>{{LATEST_YEAR}}</span></div></div>
  </section>
  <div class="workspace-footnote"><p><span class="key-dot" aria-hidden="true"></span><span>拖动底部两端选窗 · 拖动中段平移 · Ctrl + 滚轮缩放</span></p><p id="analysis-context">统一窗口统计</p><span class="sr-only" id="window-cagr"></span><span class="sr-only" id="window-exposure"></span></div>

  <section class="model-section" id="methodology" aria-labelledby="method-title"><div class="section-heading"><div><p class="eyebrow">02 / THE INTELLIGENCE LAYER</p><h2 id="method-title">让每一次决策，<span>都有迹可循。</span></h2></div><p>两项方向因子，四层风险约束。<br>从信号到持仓，保留完整的推理路径。</p></div>
    <div class="model-layout"><article class="factor-panel"><div class="panel-head"><h3>信号的构成</h3><span class="overline">LATEST FACTORS</span></div><div class="factor"><div class="factor-top"><span>金油相对动量 <small>相对趋势</small></span><strong class="mono {{OIL_CLS}}">{{OIL_V}}</strong></div><div class="factor-track"><span style="width:{{OIL_PCT}}%"></span></div><div class="factor-detail"><span>权重 65%</span><span>加权贡献 {{OIL_CONTRIB}}</span></div></div><div class="factor"><div class="factor-top"><span>宏观残差缺口 <small>历史偏离</small></span><strong class="mono {{GAP_CLS}}">{{GAP_V}}</strong></div><div class="factor-track gap-track"><span style="width:{{GAP_PCT}}%"></span></div><div class="factor-detail"><span>权重 35%</span><span>加权贡献 {{GAP_CONTRIB}}</span></div></div><div class="score-equation"><span>0.65 × 相对动量 + 0.35 × 残差缺口</span><strong class="mono">{{SCORE_KPI}}</strong></div><a class="text-link" href="#factor-relative">展开完整构建方法 <span aria-hidden="true">↗</span></a></article><article class="observation-panel"><span class="overline">THE WEEKLY PERSPECTIVE</span><h3>本期观察</h3><p class="observation-copy">{{COMMENTARY}}</p><dl class="observation-facts"><div><dt>本年策略收益</dt><dd class="mono {{YTD_CLS}}">{{YTD}}</dd><small>同期基准 {{YTD_BH}}</small></div><div><dt>上次仓位切换</dt><dd class="mono">{{LAST_CHG}}</dd><small>周度标签 · 当前{{POS_CN}}</small></div></dl><p class="observation-note">仓位由复合信号、保护计数与风控共同决定。最新得分描述方向倾向，不等于下一周收益预测。</p></article></div>
    <div class="history-heading"><h3>决策足迹 <span>THE SIGNAL JOURNAL</span></h3><p id="history-summary">2016 起 · 共 {{HIST_COUNT}} 次仓位切换</p></div>
    <p class="history-scroll-hint">左右滑动，查看完整记录与区间收益 <span aria-hidden="true">↔</span></p>
    <div class="table-scroll history-table"><table><caption class="sr-only">模型仓位切换与操作区间收益记录</caption><thead><tr><th scope="col">周度日期</th><th scope="col">模型动作</th><th scope="col">GLD 价格 / USD</th><th scope="col">本期操作区间收益</th></tr></thead><tbody id="history-rows">{{HIST_ROWS}}</tbody></table></div>
    <div class="history-actions"><p class="history-note">区间收益为本次切换周至下次切换周的策略净值变化；进行中区间截至最新周。<span>沿用一周滞后执行及每次切换 0.10% 的成本；起点当周损益不重复计入，终点换仓成本计入本段，因此空仓段也可能显示 −0.10%。</span></p><button type="button" class="history-toggle" id="history-toggle" aria-expanded="false" aria-controls="history-rows" hidden><span>展开完整历史</span><small>{{HIST_COUNT}} 次切换</small><svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m4 6 4 4 4-4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg></button></div>
    <details class="rules"><summary>绩效统计与数据口径 <span aria-hidden="true">＋</span></summary><div class="rules-body"><p>GLD 为黄金 ETF 复权价格，单位为 USD / 份。周五为归组标签；最新价格交易日为 {{SOURCE_DATE}}，周度标签为 {{DATE}}（{{WEEK_NOTE}}）。</p><p>展示区间自 2016 年开始，更早数据用于滚动热身。区间收益采用净值端点比值，超额为策略与基准收益之差（百分点）。波动率使用周收益样本标准差按 √52 年化；夏普假设无风险利率为零。少于 13 周收益观测时不显示年化收益。区间回撤从所选起点重新计算；持仓占比为观测点的仓位均值，调仓仅计区间内部切换。</p><p>仓位滞后一周，每单位仓位切换扣除 0.10% 成本。现有研究将周度对数收益按 (1+r) 连乘，可能与 ETF 价格端点收益不同。历史回测不代表未来收益，完整假设与局限见下方研究方法。</p></div></details>
  </section>
<section id="research-notes" class="research-notes" aria-labelledby="research-title">
  <header class="research-heading">
    <p class="eyebrow">RESEARCH METHODOLOGY</p>
    <h2 id="research-title"><span>信号背后，</span><span>方法透明。</span></h2>
    <p class="research-intro">选择一个主题，展开对应的构建卡片。从原始数据、数学公式到执行边界，一次读透一层逻辑。</p>
  </header>
  <div class="research-layout">
    <aside class="research-sidebar">
      <p class="research-sidebar-label">选择研究主题 / EXPLORE THE MODEL</p>
      <nav aria-label="研究方法目录">
        <a href="#method-data"><span>01</span> 数据与标准化</a>
        <a href="#factor-relative"><span>02</span> 黄金相对原油动量</a>
        <a href="#factor-gap"><span>03</span> 宏观残差缺口</a>
        <a href="#factor-crisis"><span>04</span> 危机指数</a>
        <a href="#factor-rsi"><span>05</span> 周度 RSI</a>
        <a href="#model-execution"><span>06</span> 组合与执行口径</a>
        <a href="#method-references"><span>07</span> 来源与研究边界</a>
      </nav>
      <p class="research-reading"><span>7 个研究主题 · 点击切换</span><span>当前主题 <strong id="reading-percent">01 / 07</strong></span></p>
    </aside>
    <div class="research-content">
      <article id="method-data" class="method-article" aria-labelledby="method-data-title">
        <p class="method-index">01 / DATA FOUNDATION</p>
        <h3 id="method-data-title">先统一时间，再比较信号。</h3>
        <p class="method-lead">所有因子按周计算。黄金价格采用 GLD 的美元每股价格，不能读作美元每盎司现货金价；GLD 的目标是反映金条表现并扣除基金费用。<a class="method-cite" href="#source-gld">[1]</a></p>
        <p class="method-copy">行情取 Yahoo 的复权收盘价，缺失时回退收盘价；宏观数据取 FRED。日频面板先向前填充，再取每个周五标签下的最后一笔观测。周五是归组标签，节假日或本周尚未结束时，最新价格可能早于标签日。展示从 <strong>2016 年</strong>开始，更早数据仍参与滚动窗口和持仓状态的热身。</p>
        <div class="method-formula">
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="滚动标准分等于当期值减去窗口均值，再除以窗口样本标准差">
            <mrow><msub><mi>z</mi><mrow><mi>w</mi><mo>,</mo><mi>t</mi></mrow></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><msub><mi>x</mi><mi>t</mi></msub><mo>−</mo><msub><mi>μ</mi><mrow><mi>w</mi><mo>,</mo><mi>t</mi></mrow></msub></mrow><msub><mi>s</mi><mrow><mi>w</mi><mo>,</mo><mi>t</mi></mrow></msub></mfrac></mrow>
          </math>
          <p class="formula-caption">窗口含当期及之前 w−1 周；有效观测至少 w/2 个。</p>
        </div>
        <dl class="method-definitions">
          <div><dt>μ 与 s</dt><dd>同一滚动窗口内的均值、样本标准差；标准差分母为 n−1。零方差或观测不足时，该值不可用。</dd></div>
          <div><dt>52 周 / 104 周</dt><dd>分别至少需 26 / 52 个有效观测。标准化用当期数据，但不引用未来周。</dd></div>
          <div><dt>Δ 与 r</dt><dd>Δx 为本周减上周；价格收益 r＝Δ ln P。利率和利差保留原始百分数单位，其差值以百分点计。</dd></div>
        </dl>
        <p class="method-note">滚动标准化只看当前及过去，不等于数据在历史当时已经可得。当前下载的宏观数据可能包含后续修订，且观察日期与实际发布日期不同，详见研究边界。</p>
      </article>

      <article id="factor-relative" class="method-article" aria-labelledby="factor-relative-title">
        <p class="method-index">02 / RELATIVE MOMENTUM · 65%</p>
        <h3 id="factor-relative-title">黄金相对原油的半年动量。</h3>
        <p class="method-lead">先比较黄金与原油过去 26 周的表现，再判断这份相对强弱是否高于它自身最近一年的常态。比值方向是<strong>黄金 / 原油</strong>。</p>
        <div class="method-formula">
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="相对动量等于当期黄金原油比的对数，减去二十六周前比值的对数">
            <mrow><msub><mi>m</mi><mi>t</mi></msub><mo>=</mo><mi mathvariant="normal">ln</mi><mfrac><msub><mi>G</mi><mi>t</mi></msub><msub><mi>O</mi><mi>t</mi></msub></mfrac><mo>−</mo><mi mathvariant="normal">ln</mi><mfrac><msub><mi>G</mi><mrow><mi>t</mi><mo>−</mo><mn>26</mn></mrow></msub><msub><mi>O</mi><mrow><mi>t</mi><mo>−</mo><mn>26</mn></mrow></msub></mfrac></mrow>
          </math>
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="相对动量因子等于五十二周滚动标准分除以二的双曲正切">
            <mrow><msub><mi>f</mi><mrow><mi mathvariant="normal">rel</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><mi mathvariant="normal">tanh</mi><mo>(</mo><mfrac><mrow><msub><mi>z</mi><mrow><mn>52</mn><mo>,</mo><mi>t</mi></mrow></msub><mo>(</mo><mi>m</mi><mo>)</mo></mrow><mn>2</mn></mfrac><mo>)</mo></mrow>
          </math>
        </div>
        <dl class="method-definitions">
          <div><dt>G 与 O</dt><dd>GLD 与 WTI 原油期货连续行情 CL=F 的周度价格；两者不是同单位的实物价格。</dd></div>
          <div><dt>m 的含义</dt><dd>黄金 26 周累计对数收益减原油同期累计对数收益；计价单位的固定倍数会在对数差分中消去。</dd></div>
        </dl>
        <ol class="method-steps">
          <li>用 26 周变化提取相对趋势，再以 52 周滚动均值和标准差消除量纲。</li>
          <li>用 tanh 将极端标准分平滑压缩到 −1 与 1 之间，减弱异常值对组合的影响。</li>
        </ol>
        <p class="method-copy">例如，两者都下跌但原油跌得更深，原始相对动量也可能为正；反之，黄金上涨而原油涨得更快，相对动量可能为负。标准化还会再次比较历史常态，因此原始相对收益为正时，最终因子仍可能低于零。</p>
        <p class="method-note">正分表示这份相对动量高于自身滚动均值，不能直接等同于“黄金正在上涨”。因子值不是胜率；26、52 与除数 2 是模型参数。</p>
      </article>

      <article id="factor-gap" class="method-article" aria-labelledby="factor-gap-title">
        <p class="method-index">03 / MACRO RESIDUAL GAP · 35%</p>
        <h3 id="factor-gap-title">把宏观解释后的剩余走势留下来。</h3>
        <p class="method-lead">每周重新估计黄金对七项宏观变化的敏感度，再用上一周的系数解释本周涨跌。未被解释的部分逐周累积，形成一个去趋势后的残差缺口。</p>
        <div class="method-table-wrap">
          <table class="method-table">
            <caption>回归向量 X 的七项输入</caption>
            <thead><tr><th scope="col">变量</th><th scope="col">每周变换与数据含义</th></tr></thead>
            <tbody>
              <tr><th scope="row">美元</th><td>Δ ln DXY；Yahoo DX-Y.NYB 美元指数。</td></tr>
              <tr><th scope="row">美股</th><td>Δ ln SPX；Yahoo ^GSPC 标普 500 指数。</td></tr>
              <tr><th scope="row">实际利率</th><td>Δ DFII10；10 年期通胀保值国债收益率，百分点。<a class="method-cite" href="#source-real">[2]</a></td></tr>
              <tr><th scope="row">通胀补偿</th><td>Δ T10YIE；10 年期名义与实际国债收益率之差，百分点。<a class="method-cite" href="#source-breakeven">[3]</a></td></tr>
              <tr><th scope="row">原油</th><td>Δ ln WTI；Yahoo CL=F 连续期货行情。</td></tr>
              <tr><th scope="row">信用利差</th><td>Δ BAA10Y；Baa 公司债收益率减 10 年期国债收益率，百分点。<a class="method-cite" href="#source-credit">[4]</a></td></tr>
              <tr><th scope="row">金融条件</th><td>Δ NFCI；芝加哥联储金融条件指数的点数变化。<a class="method-cite" href="#source-nfci">[5]</a></td></tr>
            </tbody>
          </table>
        </div>
        <div class="method-formula">
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="系数通过二十六周指数加权最小二乘估计，半衰期十三周，无截距">
            <mrow><msub><mover><mi>β</mi><mo>^</mo></mover><mi>t</mi></msub><mo>=</mo><munder><mi mathvariant="normal">arg min</mi><mi>β</mi></munder><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>0</mn></mrow><mn>25</mn></munderover><msup><mn>2</mn><mrow><mo>−</mo><mi>j</mi><mo>/</mo><mn>13</mn></mrow></msup><msup><mrow><mo>(</mo><msub><mi>r</mi><mrow><mi>G</mi><mo>,</mo><mi>t</mi><mo>−</mo><mi>j</mi></mrow></msub><mo>−</mo><msup><mi>β</mi><mi mathvariant="normal">T</mi></msup><msub><mi>X</mi><mrow><mi>t</mi><mo>−</mo><mi>j</mi></mrow></msub><mo>)</mo></mrow><mn>2</mn></msup></mrow>
          </math>
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="本周残差等于黄金本周收益减上一周系数乘以本周宏观变化">
            <mrow><msub><mi>e</mi><mi>t</mi></msub><mo>=</mo><msub><mi>r</mi><mrow><mi>G</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>−</mo><msubsup><mover><mi>β</mi><mo>^</mo></mover><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow><mi mathvariant="normal">T</mi></msubsup><msub><mi>X</mi><mi>t</mi></msub></mrow>
          </math>
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="累计残差减去自身五十二周均值得到缺口">
            <mrow><msub><mi>C</mi><mi>t</mi></msub><mo>=</mo><munder><mo>∑</mo><mrow><mi>u</mi><mo>≤</mo><mi>t</mi></mrow></munder><msub><mi>e</mi><mi>u</mi></msub><mo>,</mo><mspace width="1em"/><msub><mi>g</mi><mi>t</mi></msub><mo>=</mo><msub><mi>C</mi><mi>t</mi></msub><mo>−</mo><msub><mrow><msub><mi mathvariant="normal">MA</mi><mn>52</mn></msub><mo>(</mo><mi>C</mi><mo>)</mo></mrow><mi>t</mi></msub></mrow>
          </math>
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="缺口先以零填充缺失，再做一百零四周标准化，除以二并取双曲正切">
            <mrow><msub><mi>f</mi><mrow><mi mathvariant="normal">gap</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><mi mathvariant="normal">tanh</mi><mo>(</mo><mfrac><mrow><msub><mi>z</mi><mrow><mn>104</mn><mo>,</mo><mi>t</mi></mrow></msub><mo>(</mo><mi mathvariant="normal">fill₀</mi><mo>(</mo><mi>g</mi><mo>)</mo><mo>)</mo></mrow><mn>2</mn></mfrac><mo>)</mo></mrow>
          </math>
        </div>
        <ol class="method-steps">
          <li><strong>估计敏感度。</strong>使用原始变换后的七列 X，不加截距、不预先做 z 标准化。最近一周权重为 1，向前 13 周降至 1/2；每次使用 26 周观测。</li>
          <li><strong>取滞后系数。</strong>β 只用到上一周的数据，但 e 仍需本周 X 与本周黄金收益。它是一步滞后系数的解释残差，不能视为提前一周预测收益。</li>
          <li><strong>累积并去均值。</strong>累计残差减去其 52 周均值，均值至少需要 26 个有效观测；缺口缺失先填 0，再做 104 周 z 标准化和 tanh 压缩。</li>
        </ol>
        <p class="method-note">“缺口”是相对自身历史水平的累计残差偏离，不是以美元计价的公允价值减现价。正值本身不证明低估或未来均值回归。热身期会将回归 X 的缺失值置零；当本周黄金收益或任一 X 非有限时，先把用于回归的该周黄金收益置零。这会影响最早期估计。</p>
      </article>

      <article id="factor-crisis" class="method-article" aria-labelledby="factor-crisis-title">
        <p class="method-index">04 / CRISIS OVERLAY</p>
        <h3 id="factor-crisis-title">把压力水平与变化速度分开看。</h3>
        <p class="method-lead">危机指数 K 汇总五种市场状态。它既观察信用与波动压力，也观察美元走强、跨资产同向和股债分散失效；用于持仓规则，不进入 65% / 35% 的方向得分。</p>
        <div class="method-formula">
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="危机指数为五个分量各自五十二周标准分的等权平均">
            <mrow><msub><mi>K</mi><mi>t</mi></msub><mo>=</mo><mfrac><mn>1</mn><mn>5</mn></mfrac><munderover><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mn>5</mn></munderover><msub><mi>z</mi><mrow><mn>52</mn><mo>,</mo><mi>t</mi></mrow></msub><mo>(</mo><msub><mi>x</mi><mi>k</mi></msub><mo>)</mo><mo>,</mo><mspace width="1em"/><msub><mi>Δ</mi><mn>4</mn></msub><msub><mi>K</mi><mi>t</mi></msub><mo>=</mo><msub><mi>K</mi><mi>t</mi></msub><mo>−</mo><msub><mi>K</mi><mrow><mi>t</mi><mo>−</mo><mn>4</mn></mrow></msub></mrow>
          </math>
        </div>
        <dl class="method-definitions">
          <div><dt>信用压力</dt><dd>BAA10Y 的利差水平；越高表示 Baa 公司债相对国债的收益率补偿越大。<a class="method-cite" href="#source-credit">[4]</a></dd></div>
          <div><dt>权益波动</dt><dd>VIXCLS 指数水平，反映美股期权隐含的近期预期波动，非黄金波动率。<a class="method-cite" href="#source-vix">[6]</a></dd></div>
          <div><dt>美元冲击</dt><dd>ln DXYₜ − ln DXYₜ₋₄，即美元指数 4 周累计对数变化。</dd></div>
          <div><dt>跨资产相关</dt><dd>黄金、美元、美股、原油、债券方向代理与信用方向代理，最近 26 周的 15 对两两相关系数取算术平均；保留正负号，不取绝对值。</dd></div>
          <div><dt>股债相关</dt><dd>美股收益与债券方向代理的 26 周滚动相关系数。</dd></div>
        </dl>
        <p class="method-copy">债券代理为 −8ΔDGS10，信用代理为 −4ΔBAA10Y；系数仅用于构造方向序列，利率仍以百分点计，不能当作可交易债券组合收益。跨资产相关模块先按全样本列均值与标准差变换，再把缺失置零。完整无缺失窗口内，相关系数不受这种正尺度变换影响；含缺失值的窗口则可能受填值影响。<a class="method-cite" href="#source-treasury">[7]</a></p>
        <p class="method-note">五项分别标准化后等权相加，任一分量缺失时 K 不可用。K 是相对历史的压力刻度，并非危机概率。它与研究流水线另存的 CCI、吸收率及信用因子不同；这些参考项未直接进入当前方向得分。</p>
      </article>

      <article id="factor-rsi" class="method-article" aria-labelledby="factor-rsi-title">
        <p class="method-index">05 / WEEKLY RSI · 14</p>
        <h3 id="factor-rsi-title">衡量涨跌力度，再覆盖极端仓位。</h3>
        <p class="method-lead">RSI 以每周 GLD 价格涨跌额衡量上行动力与下行动力。这里的 14 指 <strong>14 周</strong>，不是日线常见的 14 个交易日。指标源于 Wilder 的技术分析体系。<a class="method-cite" href="#source-wilder">[8]</a></p>
        <div class="method-formula">
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="上行动力为本周价格上涨额，下行动力为价格下跌额的正值">
            <mrow><msub><mi>U</mi><mi>t</mi></msub><mo>=</mo><mi mathvariant="normal">max</mi><mo>(</mo><mi>Δ</mi><msub><mi>G</mi><mi>t</mi></msub><mo>,</mo><mn>0</mn><mo>)</mo><mo>,</mo><mspace width="1em"/><msub><mi>D</mi><mi>t</mi></msub><mo>=</mo><mi mathvariant="normal">max</mi><mo>(</mo><mo>−</mo><mi>Δ</mi><msub><mi>G</mi><mi>t</mi></msub><mo>,</mo><mn>0</mn><mo>)</mo></mrow>
          </math>
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="平滑上涨额等于十三倍上周平滑值加本周上涨额，再除以十四，下跌额同理">
            <mrow><msub><mover><mi>U</mi><mo>¯</mo></mover><mi>t</mi></msub><mo>=</mo><mfrac><mrow><mn>13</mn><msub><mover><mi>U</mi><mo>¯</mo></mover><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>+</mo><msub><mi>U</mi><mi>t</mi></msub></mrow><mn>14</mn></mfrac><mo>,</mo><mspace width="1em"/><msub><mi mathvariant="normal">RSI</mi><mi>t</mi></msub><mo>=</mo><mn>100</mn><mo>−</mo><mfrac><mn>100</mn><mrow><mn>1</mn><mo>+</mo><msub><mover><mi>U</mi><mo>¯</mo></mover><mi>t</mi></msub><mo>/</mo><msub><mover><mi>D</mi><mo>¯</mo></mover><mi>t</mi></msub></mrow></mfrac></mrow>
          </math>
        </div>
        <p class="method-copy">下跌额均值 D̄ 按同样递推平滑。实现采用 α＝1/14、adjust=False，以首个有效周涨跌额初始化，不先计算前 14 周简单均值。因此它与采用另一种初始化的 RSI 在早期可能不同；若分母与分子均为零，结果可能缺失。<a class="method-cite" href="#source-ewm">[9]</a></p>
        <dl class="method-definitions">
          <div><dt>计算尺度</dt><dd>输入是价格差额而非对数收益，历史变动的影响以 13/14 的系数逐周衰减；14 周是平滑参数，计算并非只保留最近 14 周。</dd></div>
          <div><dt>指标解释</dt><dd>上涨与下跌的平滑力度相同，对应 RSI 为 50；接近 100 表示上涨力度占优，接近 0 则相反。极端读数反映当前强弱，不独立证明下一周反转。</dd></div>
        </dl>
        <p class="method-note">模型实际规则为 RSI &gt; 80 时覆盖为空仓；RSI &lt; 20 且综合得分 &gt; −0.3 时覆盖为做多。界面用于描述状态的 70 / 30 刻度，不是这两条执行阈值。</p>
      </article>

      <article id="model-execution" class="method-article" aria-labelledby="model-execution-title">
        <p class="method-index">06 / SIGNAL TO POSITION</p>
        <h3 id="model-execution-title">分数表达倾向，规则决定仓位。</h3>
        <p class="method-lead">综合得分 S 是两因子的固定加权平均。仓位只取 1（做多）或 0（空仓）；得分低于零不产生做空仓位。权重与交易阈值是模型设计，不代表学界验证的最优组合。</p>
        <div class="method-formula">
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="综合得分等于百分之六十五相对动量因子加百分之三十五残差缺口因子">
            <mrow><msub><mi>S</mi><mi>t</mi></msub><mo>=</mo><mn>0.65</mn><msub><mi>f</mi><mrow><mi mathvariant="normal">rel</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>+</mo><mn>0.35</mn><msub><mi>f</mi><mrow><mi mathvariant="normal">gap</mi><mo>,</mo><mi>t</mi></mrow></msub></mrow>
          </math>
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="动态入场阈值为零点零五加零点零二乘以空仓计数减十七再除二十六的平方">
            <mrow><mi>θ</mi><mo>(</mo><mi>F</mi><mo>)</mo><mo>=</mo><mn>0.05</mn><mo>+</mo><mn>0.02</mn><msup><mrow><mo>(</mo><mfrac><mrow><mi>F</mi><mo>−</mo><mn>17</mn></mrow><mn>26</mn></mfrac><mo>)</mo></mrow><mn>2</mn></msup></mrow>
          </math>
          <p class="formula-caption">F 为内部空仓计数。阈值在 F＝17 时最低，之后再次上升。</p>
        </div>
        <ol class="method-steps">
          <li><strong>危机离场。</strong>内部持仓时，若 Δ₄K &gt; 0.3，先清仓并重置计数。</li>
          <li><strong>危机回补。</strong>内部空仓、Δ₄K &lt; −0.3 且 K &gt; 0.5 时入场，设 4 周保护计数。</li>
          <li><strong>常规退出。</strong>内部持仓且保护计数归零时，若 S &lt; −0.05 或 S 不可用则离场。</li>
          <li><strong>常规入场。</strong>内部空仓、无保护计数且 S &gt; θ(F) 时入场，设 8 周保护计数。</li>
          <li><strong>最后覆盖。</strong>完成内部状态序列后，再按 RSI 80 / 20 规则覆盖输出仓位。</li>
        </ol>
        <p class="method-copy">内部状态从空仓、F＝999、保护计数为 0 开始；离场时 F 归零，入场不归零，每个内部空仓周末 F 加 1。上述条件按顺序独立判断，同周早先的动作可能被后续动作覆盖，例如危机离场后仍可能满足常规重入。8 / 4 周是保护计数参数，入场当周即递减；危机条件及最终 RSI 覆盖可打断仓位。RSI 覆盖不会回写内部持仓状态或重置计数，因此不能把它理解为一次完整的新状态转换。</p>
        <div class="method-formula">
          <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="当前回测周收益等于上周输出仓位乘以本周黄金对数收益，减千分之一换仓成本">
            <mrow><msub><mi>r</mi><mrow><mi mathvariant="normal">model</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><msub><mi>p</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><msub><mi>r</mi><mrow><mi>G</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>−</mo><mn>0.001</mn><mo>|</mo><msub><mi>p</mi><mi>t</mi></msub><mo>−</mo><msub><mi>p</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>|</mo></mrow>
          </math>
          <p class="formula-caption">现有净值按 ∏(1＋r<sub>model,t</sub>) 累乘；基准同样按 ∏(1＋r<sub>G,t</sub>) 计算。</p>
        </div>
        <p class="method-note">信号仓位滞后一周计收益，每单位仓位变动扣 10 个基点，买卖分别计费。现有实现将<strong>对数收益代入简单收益的连乘式</strong>；它与用 exp(r)−1 计算的可交易简单收益存在差异，基准也不严格等于价格比。页面绩效因此沿用研究口径，尚未完整建模滑点、实际成交时点及其他摩擦。</p>
      </article>

      <article id="method-references" class="method-article method-references" aria-labelledby="method-references-title">
        <p class="method-index">07 / SOURCES &amp; SCOPE</p>
        <h3 id="method-references-title">让解释与证据保持同一尺度。</h3>
        <p class="method-lead">以上公式对应当前运行逻辑。数据文档定义变量，方法文献解释计算来源；它们不为本项目的参数选择或未来收益背书。</p>
        <p class="method-copy">宏观面板按观察日期对齐，未逐项重建历史发布日期和数据版本。FRED 默认提供当前所知的历史信息，而历史当时可得的信息需要结合实时区间或 ALFRED 版本查询。故本研究尚非严格的 point-in-time 回测，可能受到发布滞后与修订影响。2016 年是展示起点，不能据此宣称从该年开始独立样本外验证。<a class="method-cite" href="#source-vintages">[10]</a></p>
        <p class="method-copy">图表窗口切换只改变观测与统计区间，不重新训练模型。净值可在窗口左端归一为 100；回撤相对该窗口内历史高点计算，因此会漏掉左端以前的高点。夏普按周收益均值除以样本标准差再乘 √52、无风险利率取零；年化收益使用区间首尾净值与周数。</p>
        <ol class="reference-list">
          <li id="source-gld"><a href="https://www.ssga.com/us/en/individual/etfs/spdr-gold-shares-gld" target="_blank" rel="noopener noreferrer">SPDR Gold Shares · GLD 基金目标与产品资料</a><span>价格跟踪目标、基金费用与份额口径。</span></li>
          <li id="source-real"><a href="https://fred.stlouisfed.org/series/DFII10" target="_blank" rel="noopener noreferrer">FRED · DFII10</a><span>10 年期通胀保值国债固定期限收益率。</span></li>
          <li id="source-breakeven"><a href="https://fred.stlouisfed.org/series/T10YIE" target="_blank" rel="noopener noreferrer">FRED · T10YIE</a><span>10 年期盈亏平衡通胀率，由名义与实际国债收益率差构造。</span></li>
          <li id="source-credit"><a href="https://fred.stlouisfed.org/series/BAA10Y" target="_blank" rel="noopener noreferrer">FRED · BAA10Y</a><span>Moody’s Baa 公司债相对 10 年期国债收益率利差。</span></li>
          <li id="source-nfci"><a href="https://fred.stlouisfed.org/series/NFCI" target="_blank" rel="noopener noreferrer">FRED / Chicago Fed · NFCI</a><span>金融条件指数；正值表示相对历史均值偏紧，负值表示偏松。</span></li>
          <li id="source-vix"><a href="https://fred.stlouisfed.org/series/VIXCLS" target="_blank" rel="noopener noreferrer">FRED / Cboe · VIXCLS</a><span>由股指期权价格反映的近期预期市场波动。</span></li>
          <li id="source-treasury"><a href="https://fred.stlouisfed.org/series/DGS10" target="_blank" rel="noopener noreferrer">FRED · DGS10</a><span>10 年期美国国债固定期限名义收益率，百分数。</span></li>
          <li id="source-wilder"><a href="https://windsorpublishing.com/product/new-concepts-in-technical-trading-systems/" target="_blank" rel="noopener noreferrer">Wilder, J. W. (1978) · New Concepts in Technical Trading Systems</a><span>RSI 的原始著作；此处链接为出版社书目资料。</span></li>
          <li id="source-ewm"><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html" target="_blank" rel="noopener noreferrer">pandas · 指数加权窗口文档</a><span>α 与 adjust=False 的递推定义；页面 RSI 初始化以实际实现为准。</span></li>
          <li id="source-vintages"><a href="https://fred.stlouisfed.org/docs/api/fred/realtime_period.html" target="_blank" rel="noopener noreferrer">FRED / ALFRED · Real-Time Periods</a><span>观察时点、历史信息集与数据版本的区分。</span></li>
        </ol>
        <p class="method-implementation">研究中另存的信用、耦合、股债信号、CCI、Kalman 系数与挑战者组合用于研究参考，未作为额外因子加入当前 65% / 35% 得分。</p>
      </article>
    </div>
  </div>
  <a class="research-bottom-link" href="#overview">返回研究台 <span aria-hidden="true">↑</span></a>
</section>

</main>
<footer class="shell"><div class="footer-brand">AURUM<span>理性观察 · 长期视角</span></div><div class="footer-note">Yahoo Finance / FRED · 周度因子模型 · 价格交易日 {{SOURCE_DATE}}<br>仅供研究参考，不构成投资建议。<a href="#overview">返回顶部 ↑</a></div></footer>
<noscript><p class="shell" style="padding-bottom:24px;color:#939d9f">启用 JavaScript 可使用图表与区间统计；市场快照、信号记录与完整研究方法仍可直接阅读。</p></noscript>
<script>
(() => {
  const positive = value => Number.isFinite(value) && value > 0;
  const ratio = (last, first) => positive(last) && positive(first) ? last / first - 1 : null;

  function normalize(data, start) {
    const base = data[start]?.[1];
    return data.map(([date, value]) => [date,
      positive(base) && positive(value) ? value / base * 100 : null]);
  }

  function drawdowns(data, start) {
    let peak = null;
    return data.map(([date, value], index) => {
      if (index < start || !positive(value)) return [date, null];
      peak = peak === null ? value : Math.max(peak, value);
      return [date, (value - peak) / peak * 100];
    });
  }

  function summarize(rows, start, end) {
    const selected = rows.slice(start, end + 1);
    const first = selected[0], last = selected[selected.length - 1];
    const intervals = Math.max(0, selected.length - 1);
    const strategyReturn = ratio(last?.equity, first?.equity);
    const benchmarkReturn = ratio(last?.benchmark, first?.benchmark);
    const returns = [];
    let peak = null, maxDrawdown = null, held = 0, positions = 0, tradeCount = 0;
    selected.forEach((row, index) => {
      if (positive(row.equity)) {
        peak = peak === null ? row.equity : Math.max(peak, row.equity);
        maxDrawdown = Math.min(maxDrawdown ?? 0, (row.equity - peak) / peak);
      }
      if (Number.isFinite(row.position)) {
        held += row.position;
        positions++;
        if (index && Number.isFinite(selected[index - 1].position) &&
            row.position !== selected[index - 1].position) tradeCount++;
      }
      if (index) {
        const weeklyReturn = ratio(row.equity, selected[index - 1].equity);
        if (weeklyReturn !== null) returns.push(weeklyReturn);
      }
    });
    let volatility = null, sharpe = null;
    if (returns.length >= 2) {
      const mean = returns.reduce((total, value) => total + value, 0) / returns.length;
      const variance = returns.reduce((total, value) => total + (value - mean) ** 2, 0) /
        (returns.length - 1);
      const deviation = Math.sqrt(variance);
      volatility = deviation * Math.sqrt(52);
      if (deviation > 0) sharpe = mean / deviation * Math.sqrt(52);
    }
    return {
      start, end, observations: selected.length, intervals,
      priceReturn: ratio(last?.price, first?.price),
      strategyReturn, benchmarkReturn,
      excessReturn: strategyReturn !== null && benchmarkReturn !== null ?
        strategyReturn - benchmarkReturn : null,
      cagr: intervals >= 13 && strategyReturn !== null ?
        (1 + strategyReturn) ** (52 / intervals) - 1 : null,
      volatility, sharpe, maxDrawdown,
      exposure: positions ? held / positions : null,
      tradeCount,
      latestScore: Number.isFinite(last?.score) ? last.score : null
    };
  }

  window.GoldAnalysis = {summarize, normalize, drawdowns};
})();

window.addEventListener('DOMContentLoaded', () => {
  const GOLD={{GOLD_DATA}}, SCORE={{SCORE_DATA}}, EQS={{EQS_DATA}}, EQB={{EQB_DATA}};
  const BUY={{BUY_DATA}}, SELL={{SELL_DATA}}, ROWS={{ANALYSIS_DATA}};
  const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const workspace=document.getElementById('analysis-workspace');
  const datePopover=document.querySelector('.date-popover');
  const mainNav=document.querySelector('nav.primary-nav');
  const navLinks=[...mainNav.querySelectorAll('a')];
  const navSections=navLinks.map(link=>document.querySelector(link.getAttribute('href')));
  const topbar=document.querySelector('.topbar');
  const research=document.getElementById('research-notes');
  const readingLinks=[...research.querySelectorAll('nav a')];
  const articles=readingLinks.map(link=>document.querySelector(link.getAttribute('href')));
  let scrollQueued=false;
  const setCurrent=(links,active)=>links.forEach(link=>{
    if(link.getAttribute('href')==='#'+active.id)link.setAttribute('aria-current','location');
    else link.removeAttribute('aria-current');
  });
  const updateNavigation=()=>{
    const threshold=topbar.getBoundingClientRect().bottom+28;
    let active=navSections[0];
    navSections.forEach(section=>{if(section.getBoundingClientRect().top<=threshold)active=section;});
    setCurrent(navLinks,active);
    const current=mainNav.querySelector('[aria-current]');
    mainNav.style.setProperty('--nav-x',current.offsetLeft+'px');
    mainNav.style.setProperty('--nav-w',current.offsetWidth+'px');
    topbar.classList.toggle('is-scrolled',window.scrollY>20);
    const rect=research.getBoundingClientRect();
    const progress=Math.max(0,Math.min(1,(threshold-rect.top)/Math.max(1,rect.height-window.innerHeight+threshold)));
    topbar.style.setProperty('--reading-progress',progress);

    scrollQueued=false;
  };
  const queueNavigation=()=>{if(!scrollQueued){scrollQueued=true;requestAnimationFrame(updateNavigation);}};
  window.addEventListener('scroll',queueNavigation,{passive:true});
  window.addEventListener('resize',queueNavigation);
  document.fonts.ready.then(queueNavigation);
  updateNavigation();
  const historyToggle=document.getElementById('history-toggle');
  const historyExtra=[...document.querySelectorAll('[data-history-extra]')];
  const historyCount=document.querySelectorAll('#history-rows tr').length;
  if(historyExtra.length){
    historyExtra.forEach(row=>{row.hidden=true;});
    historyToggle.hidden=false;
    document.getElementById('history-summary').textContent='最近 8 次 · 共 '+historyCount+' 次切换';
    historyToggle.addEventListener('click',()=>{
      const expanded=historyToggle.getAttribute('aria-expanded')!=='true';
      historyExtra.forEach(row=>{row.hidden=!expanded;});
      historyToggle.setAttribute('aria-expanded',String(expanded));
      historyToggle.querySelector('span').textContent=expanded?'收起较早记录':'展开完整历史';
      document.getElementById('history-summary').textContent=expanded?'2016 起 · 共 '+historyCount+' 次切换':'最近 8 次 · 共 '+historyCount+' 次切换';
      if(!expanded){
        historyToggle.focus({preventScroll:true});
        const rect=historyToggle.getBoundingClientRect();
        if(rect.top<topbar.getBoundingClientRect().bottom||rect.bottom>innerHeight)historyToggle.scrollIntoView({block:'center',behavior:'instant'});
      }
      queueNavigation();
    });
  }
  const methodDirectory=research.querySelector('[aria-label="研究方法目录"]');
  const methodLayout=research.querySelector('.research-layout');
  let activeMethod=articles[0];
  research.classList.add('method-cards');
  methodDirectory.setAttribute('role','tablist');
  readingLinks.forEach((link,index)=>{
    link.id='tab-'+articles[index].id;
    link.setAttribute('role','tab');link.setAttribute('aria-controls',articles[index].id);
    articles[index].setAttribute('role','tabpanel');
    articles[index].setAttribute('aria-labelledby',link.id);
  });
  const selectMethod=(article,hash,align=false)=>{
    activeMethod=article;
    const index=articles.indexOf(article);
    articles.forEach(item=>{item.hidden=item!==article;});
    readingLinks.forEach((link,position)=>{
      const selected=position===index;
      link.setAttribute('aria-selected',String(selected));link.tabIndex=selected?0:-1;
      if(selected)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');
    });
    document.getElementById('reading-percent').textContent=String(index+1).padStart(2,'0')+' / 07';
    if(hash&&location.hash!==hash)history.pushState(null,'',hash);
    if(align){
      methodLayout.scrollIntoView({block:'start',behavior:'instant'});
      const tab=readingLinks[index],rect=tab.getBoundingClientRect(),bounds=methodDirectory.getBoundingClientRect();
      if(rect.left<bounds.left||rect.right>bounds.right)methodDirectory.scrollLeft+=rect.left-bounds.left-(bounds.width-rect.width)/2;
    }
    queueNavigation();
  };
  const openMethodHash=(hash,align)=>{
    const target=document.getElementById(hash.slice(1));
    const article=target?.closest('.method-article');
    if(!article)return false;
    selectMethod(article,hash,align);
    if(target!==article)requestAnimationFrame(()=>target.scrollIntoView({block:'start',behavior:'instant'}));
    return true;
  };
  readingLinks.forEach((link,index)=>link.addEventListener('keydown',event=>{
    let next=index;
    if(event.key==='ArrowRight'||event.key==='ArrowDown')next=(index+1)%articles.length;
    else if(event.key==='ArrowLeft'||event.key==='ArrowUp')next=(index+articles.length-1)%articles.length;
    else if(event.key==='Home')next=0;
    else if(event.key==='End')next=articles.length-1;
    else return;
    event.preventDefault();selectMethod(articles[next],'#'+articles[next].id,true);readingLinks[next].focus({preventScroll:true});
  }));
  document.addEventListener('click',event=>{
    const link=event.target.closest('a[href^="#"]');
    if(!link||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
    const hash=link.getAttribute('href');
    if(!document.getElementById(hash.slice(1))?.closest('.method-article'))return;
    event.preventDefault();openMethodHash(hash,true);
  });
  window.addEventListener('hashchange',()=>openMethodHash(location.hash,true));
  window.addEventListener('popstate',()=>openMethodHash(location.hash,true));
  selectMethod(articles[0],null,false);
  if(location.hash)requestAnimationFrame(()=>openMethodHash(location.hash,true));

  if(!reducedMotion && 'IntersectionObserver' in window){
    const reveal=new IntersectionObserver(entries=>entries.forEach(entry=>{
      if(!entry.isIntersecting)return;
      entry.target.animate([{opacity:.5,transform:'translateY(16px)'},{opacity:1,transform:'translateY(0)'}],
        {duration:700,easing:'cubic-bezier(.22,1,.36,1)'});
      reveal.unobserve(entry.target);
    }),{threshold:.08});
    document.querySelectorAll('.section-heading,.research-heading').forEach(node=>reveal.observe(node));
  }

  const focusButton=document.getElementById('workspace-focus');
  const exitButton=document.getElementById('workspace-exit');
  const background=[topbar,...[...document.querySelector('main').children].filter(node=>node!==workspace),document.querySelector('footer')];
  let focused=false,savedScroll=0,mobileDetail='c1';
  const focusMode=enabled=>{
    if(enabled===focused)return;
    focused=enabled;
    if(enabled)savedScroll=window.scrollY;
    document.body.classList.toggle('workspace-focused',enabled);
    focusButton.setAttribute('aria-pressed',String(enabled));
    exitButton.hidden=!enabled;
    background.forEach(node=>{node.inert=enabled;});
    if(enabled){
      workspace.setAttribute('role','dialog');workspace.setAttribute('aria-modal','true');
      exitButton.focus({preventScroll:true});
    }else{
      if(window.innerWidth<=620&&document.querySelector('.charts-matrix').dataset.view==='all')applyChartView(mobileDetail);
      workspace.removeAttribute('role');workspace.removeAttribute('aria-modal');
      window.scrollTo({top:savedScroll,behavior:'instant'});
      focusButton.focus({preventScroll:true});
    }
  };
  focusButton.addEventListener('click',()=>focusMode(true));
  exitButton.addEventListener('click',()=>focusMode(false));
  document.addEventListener('pointerdown',event=>{if(datePopover.open&&!datePopover.contains(event.target))datePopover.open=false;});
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape'&&datePopover.open){event.preventDefault();datePopover.open=false;return;}
    if(!focused)return;
    if(event.key==='Escape'){event.preventDefault();focusMode(false);}
    if(event.key==='Tab'){
      const controls=[...workspace.querySelectorAll('button:not([disabled]),a[href],input:not([disabled]),select,summary')].filter(node=>node.getClientRects().length);
      const first=controls[0],last=controls[controls.length-1];
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
    }
  });
  const chartIds=['c1','c2','c3','c4','timeline'];
  if(!window.echarts){
    chartIds.forEach(id=>{const note=document.createElement('p');note.className='chart-error';note.textContent='图表资源未加载，请检查网络后刷新。';document.getElementById(id).replaceChildren(note);});
    workspace.querySelectorAll('button,input,select').forEach(control=>{control.disabled=true;});
    document.getElementById('analysis-period').textContent='图表资源未加载，区间分析暂不可用';
    return;
  }
  const colors={gold:'#d8bb82',green:'#8ec9b1',red:'#c88c81',blue:'#8dafc7',signal:'#a2b8ce',muted:'#84988d',line:'rgba(183,210,191,.065)'};
  const plots=Object.fromEntries(chartIds.slice(0,4).map(id=>[id,echarts.init(document.getElementById(id),null,{renderer:'svg'})]));
  const {c1,c2,c3,c4}=plots;
  const timeline=echarts.init(document.getElementById('timeline'),null,{renderer:'svg'});
  const dates=GOLD.map(row=>row[0]),lastIndex=dates.length-1;
  const analysis=window.GoldAnalysis;
  const startInput=document.getElementById('analysis-start'),endInput=document.getElementById('analysis-end');
  const error=document.getElementById('analysis-error');
  const presetButtons=[...document.querySelectorAll('[data-window]')];
  const mobile=()=>window.innerWidth<=620;
  const axisLabel={color:colors.muted,fontSize:10,fontFamily:'-apple-system,BlinkMacSystemFont,sans-serif',hideOverlap:true};
  const chartGrid=()=>({left:mobile()?34:47,right:mobile()?10:20,top:7,bottom:mobile()?21:25,containLabel:false});
  const axisForCount=(count,width)=>({axisLabel:{interval:Math.max(0,Math.ceil(count/(width<260?3:width<500?4:6))-1),formatter:value=>count>156?value.slice(0,4):value.slice(0,7)}});
  const xAxis={type:'category',data:dates,boundaryGap:false,axisLine:{show:false},axisTick:{show:false},axisPointer:{show:true,snap:true,lineStyle:{color:'#d8bb8270',type:'dashed',width:1},label:{show:false}},axisLabel:{...axisLabel,margin:8,showMinLabel:true,showMaxLabel:true}};
  const yAxis=(extra={})=>({type:'value',scale:true,splitNumber:3,axisLine:{show:false},axisTick:{show:false},axisPointer:{show:false},axisLabel:{...axisLabel,margin:8},splitLine:{lineStyle:{color:colors.line,type:'dashed'}},...extra});
  const timelineHandle=()=>mobile()?'path://M-9 -16H9Q12 -16 12 -13V13Q12 16 9 16H-9Q-12 16 -12 13V-13Q-12 -16 -9 -16Z M-1 -5V5H1V-5Z':'path://M-3 -16H3Q5 -16 5 -14V14Q5 16 3 16H-3Q-5 16 -5 14V-14Q-5 -16 -3 -16Z';
  const inside=id=>[{id:id+'-inside',type:'inside',xAxisIndex:0,startValue:0,endValue:lastIndex,minValueSpan:1,zoomOnMouseWheel:'ctrl',moveOnMouseWheel:false,moveOnMouseMove:true,preventDefaultMouseMove:false,throttle:60}];
  const base=id=>({animation:false,textStyle:{fontFamily:'-apple-system,BlinkMacSystemFont,sans-serif'},grid:chartGrid(),xAxis,yAxis:yAxis(),tooltip:{trigger:'axis',showContent:false,axisPointer:{type:'line',snap:true}},dataZoom:inside(id)});
  const gradient=(top,bottom)=>new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:top},{offset:1,color:bottom}]);
  const line=(name,data,color,extra={})=>({name,type:'line',data:data.map(row=>row[1]),symbol:'none',connectNulls:false,lineStyle:{color,width:1.5},itemStyle:{color},...extra});
  c1.setOption({...base('c1'),yAxis:yAxis({axisLabel:{...axisLabel,formatter:value=>'$'+value}}),series:[line('GLD',GOLD,colors.gold,{areaStyle:{color:gradient('#d8bb8215','#d8bb8200')}}),{name:'买入',type:'scatter',data:BUY.map(point=>[point.x,point.y]),symbol:'triangle',symbolSize:5,itemStyle:{color:colors.green},z:5},{name:'卖出',type:'scatter',data:SELL.map(point=>[point.x,point.y]),symbol:'triangle',symbolRotate:180,symbolSize:5,itemStyle:{color:colors.red},z:5}]});
  c2.setOption({...base('c2'),yAxis:yAxis({min:-1,max:1,splitNumber:2}),series:[line('复合信号',SCORE,colors.signal,{lineStyle:{color:colors.signal,width:1.3},areaStyle:{color:gradient('#8dafc70a','#8dafc700')},markArea:{silent:true,itemStyle:{color:'#d8bb8207'},data:[[{yAxis:-.05},{yAxis:.05}]]},markLine:{silent:true,symbol:'none',label:{show:false},lineStyle:{type:'dashed',width:.7},data:[{yAxis:.05,lineStyle:{color:'#8ec9b15c'}},{yAxis:-.05,lineStyle:{color:'#c88c815c'}}]}})]});
  c3.setOption({...base('c3'),yAxis:yAxis({axisLabel:{...axisLabel,formatter:value=>value.toLocaleString('en-US',{maximumFractionDigits:0})},logBase:2}),series:[line('择时策略',analysis.normalize(EQS,0),colors.green,{lineStyle:{color:colors.green,width:1.7}}),line('买入持有',analysis.normalize(EQB,0),colors.blue,{lineStyle:{color:colors.blue,width:1.2,type:'dashed'}})]});
  c4.setOption({...base('c4'),yAxis:yAxis({max:0,min:extent=>extent.min<0?Math.floor(extent.min/5)*5:-1,axisLabel:{...axisLabel,formatter:value=>value+'%'}}),series:[line('策略区间回撤',analysis.drawdowns(EQS,0),colors.red,{lineStyle:{color:colors.red,width:1.1},areaStyle:{color:gradient('#c88c8100','#c88c8120')}})]});
  timeline.setOption({animation:false,grid:{left:8,right:8,top:0,bottom:0},xAxis:{type:'category',data:dates,show:false,boundaryGap:false},yAxis:{type:'value',show:false},tooltip:{show:false},series:[line('GLD 历史',GOLD,colors.gold,{silent:true,lineStyle:{opacity:0},areaStyle:{opacity:0}})],dataZoom:[{id:'shared-timeline',type:'slider',xAxisIndex:0,filterMode:'none',startValue:0,endValue:lastIndex,minValueSpan:1,left:mobile()?18:8,right:mobile()?18:8,top:mobile()?5:3,height:mobile()?32:24,realtime:true,throttle:60,brushSelect:false,showDetail:false,moveHandleSize:0,borderColor:'#cab08366',borderRadius:5,backgroundColor:'#0a100f',fillerColor:'#c6a97145',handleIcon:timelineHandle(),handleSize:mobile()?'125%':'105%',handleStyle:{color:'#e0c28d',borderColor:'#ffdfab',borderWidth:1.2},emphasis:{handleStyle:{color:'#f0d5a2',borderColor:'#f0d5a2'}},dataBackground:{lineStyle:{color:'#638472',width:.8,opacity:.6},areaStyle:{color:'#6c8c7522'}},selectedDataBackground:{lineStyle:{color:'#e1c699',width:1.1},areaStyle:{color:'#c4ac7f28'}}}]});
  let currentRange=[0,lastIndex],normalizedStrategy=[],normalizedBenchmark=[],drawdown=[];
  let lastProbe=-1,hoverSync=false,queuedRange=null,rangeFrame=0;
  const percent=(value,signed=false)=>Number.isFinite(value)?(signed&&value>0?'+':'')+(value*100).toFixed(2)+'%':'—';
  const number=value=>Number.isFinite(value)?value.toFixed(2):'—';
  const cutoffRange=months=>{
    if(!months)return [0,lastIndex];
    const cutoff=new Date(dates[lastIndex]+'T00:00:00Z');cutoff.setUTCMonth(cutoff.getUTCMonth()-months);
    return [Math.max(0,dates.findIndex(value=>value>=cutoff.toISOString().slice(0,10))),lastIndex];
  };
  const showProbe=index=>{
    if(index<currentRange[0]||index>currentRange[1]||index===lastProbe)return;
    lastProbe=index;
    const row=ROWS[index];
    document.getElementById('probe-date').textContent=row.date;
    document.getElementById('probe-position').textContent=(index===lastIndex?'最新周 · ':'模型仓位 · ')+(row.position===1?'做多':'空仓');
    const values={price:Number.isFinite(row.price)?row.price.toFixed(2):'—',score:Number.isFinite(row.score)?(row.score>0?'+':'')+row.score.toFixed(3):'—',equity:number(normalizedStrategy[index]?.[1]),drawdown:Number.isFinite(drawdown[index]?.[1])?drawdown[index][1].toFixed(2)+'%':'—'};
    Object.entries(values).forEach(([key,value])=>{document.querySelector(`[data-probe="${key}"]`).textContent=value;});
  };
  const clearPointers=()=>{
    Object.values(plots).forEach(chart=>{chart.dispatchAction({type:'updateAxisPointer',currTrigger:'leave'},{silent:true});chart.dispatchAction({type:'hideTip'},{silent:true});});
  };
  const renderStatistics=stats=>{
    const [start,end]=currentRange;
    document.getElementById('analysis-period').textContent=`${dates[start]} — ${dates[end]} · ${stats.observations} 周度观测`;
    startInput.value=dates[start];endInput.value=dates[end];
    document.getElementById('timeline-hint').textContent=start===0&&end===lastIndex?'拖动两端握柄，缩小观察窗口':'两端缩放 · 按住中间选区左右平移';
    for(const key of ['strategyReturn','benchmarkReturn','excessReturn','maxDrawdown','volatility','sharpe']){
      const node=document.querySelector(`[data-stat="${key}"]`),value=stats[key];
      node.textContent=key==='sharpe'?number(value):key==='excessReturn'?(Number.isFinite(value)?(value>0?'+':'')+(value*100).toFixed(2):'—'):percent(value,['strategyReturn','benchmarkReturn'].includes(key));
      node.title=Number.isFinite(value)?'':'样本不足或区间净值无波动，无法计算';
      if(['strategyReturn','benchmarkReturn','excessReturn'].includes(key)){node.classList.toggle('up',value>0);node.classList.toggle('dn',value<0);}
    }
    document.getElementById('window-cagr').textContent=percent(stats.cagr,true);
    document.getElementById('window-exposure').textContent=percent(stats.exposure);
    document.getElementById('drawdown-window-value').textContent=percent(stats.maxDrawdown);
    document.getElementById('window-signal-value').textContent=Number.isFinite(ROWS[end].score)?ROWS[end].score.toFixed(3):'—';
    document.getElementById('analysis-context').innerHTML=`${stats.intervals} 周 · 调仓 <span data-stat="tradeCount">${stats.tradeCount}</span> 次 · 价格涨跌 <span data-stat="priceReturn">${percent(stats.priceReturn,true)}</span> · 年化 ${percent(stats.cagr,true)}`;
    document.querySelectorAll('[data-shift-window]').forEach(button=>{button.disabled=Number(button.dataset.shiftWindow)<0?start===0:end===lastIndex;});
    presetButtons.forEach(button=>{const candidate=cutoffRange({'6m':6,'1y':12,'2y':24,'5y':60,'all':0}[button.dataset.window]);button.setAttribute('aria-pressed',String(start===candidate[0]&&end===candidate[1]));});
  };
  const renderRange=(start,end,source='external',force=false)=>{
    if(source==='external'&&rangeFrame){cancelAnimationFrame(rangeFrame);rangeFrame=0;queuedRange=null;}
    start=Math.max(0,Math.min(lastIndex-1,Math.round(start)));end=Math.max(start+1,Math.min(lastIndex,Math.round(end)));
    if(!force&&start===currentRange[0]&&end===currentRange[1])return;
    currentRange=[start,end];error.hidden=true;
    const stats=analysis.summarize(ROWS,start,end);
    normalizedStrategy=analysis.normalize(EQS,start);normalizedBenchmark=analysis.normalize(EQB,start);drawdown=analysis.drawdowns(EQS,start);
    Object.entries(plots).forEach(([id,chart])=>{
      const option={dataZoom:[{id:id+'-inside',startValue:start,endValue:end}],xAxis:axisForCount(end-start+1,chart.getWidth())};
      if(id==='c3')option.series=[{data:normalizedStrategy.map(row=>row[1])},{data:normalizedBenchmark.map(row=>row[1])}];
      if(id==='c4')option.series=[{data:drawdown.map(row=>row[1])}];
      chart.setOption(option,{silent:true});
      document.getElementById('window-'+id).textContent=`窗口 ${dates[start]} — ${dates[end]} · ${stats.observations} 个观测`;
    });
    if(source!=='timeline')timeline.dispatchAction({type:'dataZoom',startValue:start,endValue:end},{silent:true});
    renderStatistics(stats);clearPointers();lastProbe=-1;showProbe(end);
  };
  const selectedRange=chart=>{
    const zoom=chart.getOption().dataZoom[0];
    const index=(value,fallback)=>typeof value==='number'?Math.round(value):typeof value==='string'?dates.indexOf(value):Math.round(fallback/100*lastIndex);
    return [index(zoom.startValue,zoom.start||0),index(zoom.endValue,zoom.end??100)];
  };
  const queueRange=(chart,source)=>{
    queuedRange={values:selectedRange(chart),source};
    if(!rangeFrame)rangeFrame=requestAnimationFrame(()=>{const latest=queuedRange;rangeFrame=0;queuedRange=null;renderRange(...latest.values,latest.source);});
  };
  timeline.on('datazoom',()=>queueRange(timeline,'timeline'));
  Object.entries(plots).forEach(([id,chart])=>{
    chart.on('datazoom',()=>queueRange(chart,id));
    chart.on('updateAxisPointer',event=>{
      if(hoverSync)return;
      const axis=event.axesInfo?.find(item=>item.axisDim==='x');if(!axis)return;
      const index=typeof axis.value==='number'?Math.round(axis.value):dates.indexOf(axis.value);
      if(index<currentRange[0]||index>currentRange[1])return;
      hoverSync=true;showProbe(index);
      Object.entries(plots).forEach(([other,target])=>{if(other!==id)target.dispatchAction({type:'showTip',seriesIndex:0,dataIndex:index},{silent:true});});
      hoverSync=false;
    });
  });
  document.querySelector('.charts-matrix').addEventListener('mouseleave',()=>{clearPointers();lastProbe=-1;showProbe(currentRange[1]);});
  for(const input of [startInput,endInput]){input.min=dates[0];input.max=dates[lastIndex];}
  document.getElementById('analysis-form').noValidate=true;
  document.getElementById('analysis-form').addEventListener('submit',event=>{
    event.preventDefault();
    const startDate=startInput.value,endDate=endInput.value,start=dates.findIndex(date=>date>=startDate);
    let end=lastIndex;while(end>=0&&dates[end]>endDate)end--;
    let message='';
    if(!startDate||!endDate)message='请填写开始和结束日期。';
    else if(startDate>endDate)message='开始日期不能晚于结束日期。';
    else if(startDate<dates[0]||endDate>dates[lastIndex])message=`请选择 ${dates[0]} 至 ${dates[lastIndex]} 以内的日期。`;
    else if(start<0||end-start<1)message='所选区间至少需要两个周度观测点。';
    if(message){error.textContent=message;error.hidden=false;return;}
    renderRange(start,end);datePopover.open=false;
  });
  presetButtons.forEach(button=>button.addEventListener('click',()=>{renderRange(...cutoffRange({'6m':6,'1y':12,'2y':24,'5y':60,'all':0}[button.dataset.window]));datePopover.open=false;}));
  document.getElementById('reset-window').addEventListener('click',()=>{renderRange(0,lastIndex);datePopover.open=false;});
  document.getElementById('equity-scale').addEventListener('change',event=>c3.setOption({yAxis:{type:event.target.value}}));
  const matrix=document.querySelector('.charts-matrix');
  function applyChartView(view){
    matrix.dataset.view=view;
    document.querySelectorAll('[data-chart-view]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.chartView===view)));
  }
  document.querySelectorAll('[data-chart-view]').forEach(button=>button.addEventListener('click',()=>{
    const view=button.dataset.chartView;
    if(view!=='all')mobileDetail=view;
    applyChartView(view);
    if(view==='all'&&mobile())focusMode(true);
  }));
  const phoneMedia=window.matchMedia('(max-width:620px)');
  phoneMedia.addEventListener('change',()=>applyChartView(phoneMedia.matches&&!focused?mobileDetail:'all'));
  document.querySelectorAll('[data-shift-window]').forEach(button=>button.addEventListener('click',()=>{
    const span=currentRange[1]-currentRange[0];
    const start=Math.max(0,Math.min(lastIndex-span,currentRange[0]+Number(button.dataset.shiftWindow)*(span+1)));
    renderRange(start,start+span);
  }));
  document.getElementById('export-csv').addEventListener('click',()=>{
    renderRange(...selectedRange(timeline));
    const [start,end]=currentRange;
    const cells=[['日期（周度标签）','GLD复权价格（USD/份）','模型仓位（1做多/0空仓）','复合信号','金油相对动量信号','宏观残差缺口信号','RSI（14周）','危机指数','策略区间净值（起点100）','基准区间净值（起点100）','策略区间回撤（%）']];
    ROWS.slice(start,end+1).forEach((row,offset)=>{const index=start+offset;cells.push([row.date,row.price,row.position,row.score,row.oil,row.gap,row.rsi,row.crisis,normalizedStrategy[index][1],normalizedBenchmark[index][1],drawdown[index][1]]);});
    const csv='\ufeff'+cells.map(row=>row.map(value=>'"'+(value??'').toString().replaceAll('"','""')+'"').join(',')).join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download=`aurum-window-${dates[start]}-${dates[end]}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  renderRange(0,lastIndex,'external',true);
  const observer=new ResizeObserver(entries=>{
    entries.forEach(entry=>{
      const id=entry.target.id,chart=id==='timeline'?timeline:plots[id];
      if(!entry.contentRect.width||!entry.contentRect.height)return;
      chart.resize();
      if(id==='timeline')timeline.setOption({dataZoom:[{id:'shared-timeline',left:mobile()?18:8,right:mobile()?18:8,top:mobile()?5:3,height:mobile()?32:24,handleIcon:timelineHandle(),handleSize:mobile()?'125%':'105%'}]},{silent:true});
      if(id!=='timeline')chart.setOption({grid:chartGrid(),yAxis:{axisLabel:{fontSize:mobile()?8:10}},xAxis:{...axisForCount(currentRange[1]-currentRange[0]+1,chart.getWidth()),axisLabel:{...axisForCount(currentRange[1]-currentRange[0]+1,chart.getWidth()).axisLabel,fontSize:mobile()?7:10}}},{silent:true});
    });
  });
  chartIds.forEach(id=>observer.observe(document.getElementById(id)));
  applyChartView(mobile()?'c1':'all');
});

</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
