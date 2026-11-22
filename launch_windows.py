"""Open the local dashboard from a Windows executable."""
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Import the application so PyInstaller discovers its dependencies. The entry
# script itself is bundled as data and executed by the Streamlit runner.
import src.ui  # noqa: F401
from streamlit.web import cli


def choose_port():
    for port in range(8501, 8521):
        with socket.socket() as server:
            try:
                server.bind(("127.0.0.1",port))
                return port
            except OSError:
                continue
    raise OSError("8501 至 8520 端口均不可用")


def open_when_healthy(url):
    for _ in range(60):
        try:
            with urllib.request.urlopen(url+"/_stcore/health",timeout=1) as response:
                if response.read().strip()==b"ok":
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(.5)
    print("浏览器未自动打开，请访问",url,flush=True)


def main():
    frozen=getattr(sys,"frozen",False)
    assets=Path(sys._MEIPASS) if frozen else Path(__file__).resolve().parent
    location=Path(sys.executable).parent if frozen else assets
    os.chdir(location)
    os.environ["DRONE_DISPATCH_MODE"]="local"
    os.environ["DRONE_DISPATCH_RUNS"]=str(location/"runs")
    # ui may already be imported for dependency collection.
    src.ui.CLOUD=False
    port=choose_port()
    url=f"http://127.0.0.1:{port}"
    print("城市物流无人机航网规划与空地协同调度系统")
    print(f"打开 {url}。关闭此窗口结束服务。",flush=True)
    if os.environ.get("DRONE_DISPATCH_NO_BROWSER")!="1":
        threading.Thread(target=open_when_healthy,args=(url,),daemon=True).start()
    sys.argv=["streamlit","run",str(assets/"streamlit_app.py"),
              "--server.address","127.0.0.1","--server.port",str(port),
              "--global.developmentMode","false","--server.headless","true",
              "--browser.gatherUsageStats","false"]
    cli.main()


if __name__=="__main__":
    main()
