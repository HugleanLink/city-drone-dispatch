# 城市物流无人机航网规划与空地协同调度系统

**Drone Delivery Network and Air–Ground Dispatch**

本系统根据订单需求生成固定航网，并在给定骑手、无人机、电池和起降位的条件下，为新订单提供地面直送、空地接驳或等待的建议。支持比较资源扰动前后同一订单的结果，以及纯地面、固定分流、逐单动态和多订单试排四种策略。

## 启动

需要 Python 3.12 或 3.13。首次安装后可双击 Windows 的 `启动系统.cmd`。

```powershell
python -m pip install -r requirements.txt
python -m src.cli serve
```

浏览器打开 `http://127.0.0.1:8501`。导航中的 **新单决策** 可从空队列开始。**运行控制** 提供常态、高需求、三单试排和备用电池扰动样例。打开 **回放与对照** 可查看实际交付、运行四策略及下载运行包。

## 数据

`data/demo_orders.csv.gz` 包含 **592,667 条独立生成的模拟记录**。模拟点位用于验证规划和调度方法，站点与航段均为研究设定。生成脚本固定随机种子，可运行 `python -m src.demo_data` 复现。默认从完整数据的联合分布随机抽取 4,000 条规划订单，运行时抽取 201 或 302 条运营订单。

本地模式支持读取自备 CSV 或上传。字段和窗口由 **数据与场景** 页校验。保留取送点、下单时间、业务和价格的联合分布，按业务参数生成备妥时间、重量和时限。CSV 坐标按输入经纬度处理，离线坐标示意展示模拟地面网格和固定航段。

本地运行写入 `runs/` 下的 SQLite。云端为独立会话提供记录，下载 ZIP 后可恢复；Community Cloud 本地文件不保证跨重启保存。

## 批量复现

```powershell
python -m src.cli prepare-data --input data/demo_orders.csv.gz --out data/generated
python -m src.cli generate --profile data/generated/profile.csv.gz --purpose planning --count 4000 --seed 101 --out data/generated/planning
python -m src.cli build-network --orders data/generated/planning/orders.json --profile data/generated/profile.csv.gz --out data/generated/network
python -m src.cli batch --profile data/generated/profile.csv.gz --network data/generated/network/network.json --seeds 201:230 --out runs/batch
python -m pytest -q
```

批量命令运行八类场景。可用 `--scenarios normal heavy` 缩小场景。每组四策略共享输入哈希，准时率和分位区间来自运行结果。结果可在页面导出。

## 部署

在 [Streamlit Community Cloud](https://share.streamlit.io/) 选择此仓库、`main` 分支、`streamlit_app.py` 入口，Python 版本选择 3.12。依赖由仓库根目录的 `requirements.txt` 安装。默认场景随仓库部署。

[Streamlit 部署说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
