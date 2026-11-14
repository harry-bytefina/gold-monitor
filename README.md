# 黄金择时动态监测站

基于 oil(0.65)+gap(0.35) 结构化因子模型的黄金择时监测站，由 GitHub Actions 每日 09:00（新加坡 / 北京时间，UTC+8）自动刷新。GitHub 定时任务可能排队延迟，不承诺准点。

- 信号仪表盘（做多/空仓）
- 因子贡献分解
- 金价买卖信号、净值、回撤图表

数据来源：Yahoo Finance / FRED（免费公开数据）。仅供研究参考，不构成投资建议。

图表使用随站点发布的 `assets/echarts-5.5.0/echarts.min.js`，浏览时无需访问外部图表 CDN。部署时保留完整 assets 目录，其中包含原版图表库及上游许可文件。

## 自动更新

[查看运行记录或手动运行](https://github.com/harry-bytefina/gold-monitor/actions/workflows/daily-update.yml)。工作流同时在 `main` 收到推送时运行。

每次从公开 Yahoo / FRED 数据源重新抓取数据，校验 14 项必需数据的格式、历史长度和最新有效观测日期，再重算周度因子、生成网页、记录更新并直接部署 GitHub Pages。无需本地电脑开机，也无需额外数据 API 密钥。站点地址保持不变。

校验或计算失败时，部署不会执行，网站保留上一版；错误详情和抓取日志保留 14 天。每日尝试刷新不代表每日一定有新行情：周末、休市及周度宏观数据可能保持原值。`refresh-status.json` 随站点发布，列出本次检查时间及各来源最新有效日期。

GitHub 对长期无仓库活动的公开仓库可能停用定时工作流（当前规则为 60 天）；可在 Actions 页重新启用。正常行情更新会提交 `index.html`，运行结果可在上述链接核对。

## 本地复现

需要 Python 3.12、curl：

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/fetch_data.py
python src/validate_data.py
python src/pipeline_weekly.py
python src/monitor.py
```

生成文件为 `output/monitor.html`。部署仅打包页面、`assets/` 与刷新状态，不打包源码或原始数据。
