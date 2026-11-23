# 城市物流无人机航网规划与空地协同调度系统

**Drone Delivery Network and Air–Ground Dispatch**

[打开 Streamlit 在线演示](https://city-drone-dispatch.streamlit.app/)

## 在同一页完成建网和调度

1. 从「导入运行数据」上传 CSV 或 CSV.GZ。按日组划分，或填写互不重叠的规划、运营来源数量。规划默认从来源抽取最多 4,000 条，可调整数量或使用全部规划来源。展开「建网参数」即可修改阈值，然后点击「分析并构建航网」。
2. 在 OpenStreetMap 上先后选择取货点和目的地，设置下单时间，计算配送建议。侧栏可按时段限制备用电池、无人机或起降位。应用资源扰动后，可查看同一订单在无扰和受扰条件下的两种 ETA。
3. 人工确认配送方式后，在「手工订单」填写实际送达日期和时间。同一来源且位于研究范围内的已完成新单可纳入下一次建网。
4. 在地图下方设置运营订单数（默认 201，网页上限 500），运行四策略对照。对照使用同一组订单及初始条件，显示两项准时率、分母和较纯地面的变化。侧栏的运营订单量扰动作用于这组实验。

默认建网使用 1–4.04 km 取送直线距离、3 km 两端直线接驳半径和 2 kg 载重门槛，目标为 15 个 Hub 和 10 个接驳站。Hub 使用业务价值加权的贪心覆盖选址。有参考配送时长时纳入高绕行加权。站点候选来自 CSV 的取送点，地图显示可调度的 Hub–接驳站航段，以及供规划查看的 Hub 间走廊。

## 数据和记录

`data/demo_orders.csv.gz` 有 592,667 条独立生成的模拟记录，可用于命令行或手工上传，生成命令为 `python -m src.demo_data`。正常日组模式使用前四个有效日组规划、接下来的三个用于运营。有效日组不足七个时共享来源，需要严格拆分可指定来源数量。「运营订单数／模拟日」指定每轮从运营来源生成多少单。地图显示真实地理底图，骑手里程按模拟网格估算。

CSV 最少需要 `customer_pay_time` 及 `pickup_latitude`、`pickup_longitude`、`dropoff_latitude`、`dropoff_longitude` 四列 WGS84 经纬度。支持 UTF-8 和 GB18030。提供 `weight_kg` 时使用该列，否则按业务类型生成仿真重量。上传由当前 Streamlit 会话处理。线上手工记录随当前会话保存。

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

多日命令默认保留常态 201 单及其余场景 302 单的口径，可通过 `--orders-per-day` 指定统一数量，或通过 `--scenarios normal heavy` 选择复现范围。对照结果写入指定输出目录。

Streamlit Community Cloud 使用本仓库的 `main` 分支和 `streamlit_app.py` 入口。GitHub Actions 验证 Python 3.12 和 3.14。
