# 城市物流无人机航网规划与空地协同调度系统

**Drone Delivery Network and Air–Ground Dispatch**

[打开 Streamlit 在线演示](https://city-drone-dispatch.streamlit.app/)

## 在同一页完成建网和调度

1. 使用内置模拟数据，或从「导入运行数据」上传 CSV 或 CSV.GZ。设置规划订单数（默认 4,000，上限 100,000），点击「分析并构建航网」。
2. 在 OpenStreetMap 上先后选择取货点和目的地，设置下单时间，计算配送建议。侧栏可按时段限制备用电池、无人机或起降位。也可载入电池扰动样例，查看同一订单在无扰和受扰条件下的两种 ETA。
3. 人工确认配送方式后，在「手工订单」填写实际送达日期和时间。同一来源且位于研究范围内的已完成新单可纳入下一次建网。
4. 在地图下方设置运营订单数（默认 201，网页上限 500），运行四策略对照。对照使用同一组订单及初始条件，显示两项准时率、分母和较纯地面的变化。侧栏的运营订单量扰动作用于这组实验。
5. 在页面底部下载工作区 ZIP，换会话时可恢复航网和手工记录。自备文件恢复后，建网或批量仿真需要重新上传同一份源 CSV。

## 数据和记录

`data/demo_orders.csv.gz` 有 592,667 条独立生成的模拟记录，生成命令为 `python -m src.demo_data`。程序从 CSV 的时空联合分布抽取独立的规划和运营样本，生成备妥时间、重量和交付时限。地图显示真实地理底图，骑手里程按模拟网格估算。

CSV 最少需要 `customer_pay_time` 及 `pickup_latitude`、`pickup_longitude`、`dropoff_latitude`、`dropoff_longitude` 四列 WGS84 经纬度。可在界面设置字段映射或切换编码。上传提交给当前 Streamlit 会话处理。工作区 ZIP 保存航网、来源指纹和手工记录，源 CSV 不随 ZIP 打包。

## 本机开发和多日对照

支持 Python 3.12 至 3.14。在线底图需要网络连接。

```powershell
python -m pip install -r requirements.txt
python -m src.cli serve
python -m src.cli prepare-data --input data/demo_orders.csv.gz --out data/generated
python -m src.cli generate --profile data/generated/profile.csv.gz --purpose planning --count 4000 --seed 101 --out data/generated/planning
python -m src.cli build-network --orders data/generated/planning/orders.json --profile data/generated/profile.csv.gz --out data/generated/network
python -m src.cli batch --profile data/generated/profile.csv.gz --network data/generated/network/network.json --seeds 201:230 --out runs/batch
python -m pytest -q
```

多日命令默认保留常态 201 单及其余场景 302 单的口径，可通过 `--orders-per-day` 指定统一数量，或通过 `--scenarios normal heavy` 选择复现范围。将输出的 `batch.json` 或 `batch.csv` 上传到网页的「导入多日对照结果」。工作区 ZIP 与对照结果分别保存。

Streamlit Community Cloud 使用本仓库的 `main` 分支和 `streamlit_app.py` 入口。GitHub Actions 验证 Python 3.12 和 3.14。
