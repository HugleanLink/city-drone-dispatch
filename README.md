# 城市物流无人机航网规划与空地协同调度系统

**Drone Delivery Network and Air–Ground Dispatch**

[打开 Streamlit 在线演示](https://city-drone-dispatch.streamlit.app/)

## 在同一页完成建网和调度

1. 从「导入运行数据」上传 CSV 或 CSV.GZ。分别选择规划和运营的日组区间，或填写互不重叠的规划、运营来源数量。规划默认从来源抽取最多 4,000 条，可调整数量或使用全部规划来源。展开侧栏的「建网参数」即可修改阈值，然后点击「分析并构建航网」。
2. 在 OpenStreetMap 上先后选择取货点和目的地，设置下单时间，计算配送建议。侧栏可按时段限制备用电池、无人机或起降位。应用资源扰动后，可查看同一订单在无扰和受扰条件下的两种 ETA。
3. 人工确认配送方式后，在「手工订单」填写实际送达日期和时间。同一来源且位于研究范围内的已完成新单可纳入下一次建网。
4. 在地图下方设置运营订单数（默认 201，网页上限 500），运行「配送效率对照」。用同一批订单对比相对纯地面的单均节省、耗时降幅、仅地面等待和净节省骑手分钟。侧栏可调整分时骑手、各 Hub 的无人机以及电池和站点设施，需求扰动作用于本轮批次。

默认建网使用 1–4.04 km 取送直线距离、3 km 两端直线接驳半径和 2 kg 载重门槛，Hub 和接驳站的数量上限默认分别为 15 和 10，设为 0 可使用建网算法筛出的全部候选站点。Hub 使用业务价值加权的贪心覆盖选址。有参考配送时长时纳入高绕行加权。站点候选来自 CSV 的取送点，地图显示可调度的 Hub–接驳站航段。可勾选查看 Hub 间规划走廊。

## 数据和记录

`data/demo_orders.csv.gz` 有 592,667 条独立生成的模拟记录，可用于命令行或手工上传，生成命令为 `python -m src.demo_data`。网页按顺序为有效日组编号。默认用前四个有效日组规划，剩余日组用于运营。改变规划日期时自动建议运营日期，也可以手动调整。仅有一个有效日组时，按日组模式共用该日，或改用手动数量拆分。全天接单并允许次日送达。地面耗时按时段、取送区域、距离档和业务类型校准。规划来源用于估时校准，运营来源用于抽样对照。

CSV 最少需要 `customer_pay_time`、`fetch_time`、`arrived_time` 及 `pickup_latitude`、`pickup_longitude`、`dropoff_latitude`、`dropoff_longitude` 四列 WGS84 经纬度。支持 UTF-8 和 GB18030。提供 `weight_kg` 时使用该列，否则按业务类型生成仿真重量。上传由当前 Streamlit 会话处理。线上手工记录随当前会话保存。

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

多日命令默认保留常态 201 单及其余场景 302 单的口径，可通过 `--orders` 指定统一数量，或通过 `--scenarios normal heavy` 选择复现范围。对照结果写入指定输出目录。

Streamlit Community Cloud 使用本仓库的 `main` 分支和 `streamlit_app.py` 入口。GitHub Actions 验证 Python 3.12 和 3.14。
