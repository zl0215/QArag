"""给 Web 界面截图 —— 用于改完前端后肉眼验收。

为什么不用 `chrome --headless --screenshot`：
    `--virtual-time-budget` 快进的是**虚拟时钟**，它不等真实网络流。
    对本项目最要紧的那个页面（流式问答）来说，这等于永远拍到空的答案框——
    实测加多大的 budget 都一样，2.7 秒就出图。
    所以这里走 CDP：轮询页面状态，等到答案真正落地再 captureScreenshot。

用法：
    python scripts/shot_ui.py http://127.0.0.1:8000/ -o docs/img/home.png
    python scripts/shot_ui.py "http://127.0.0.1:8000/?ask=隧道断了什么表现" -o chat.png

默认等待条件见 DEFAULT_READY_JS：`.msg-a` 里文本够长且光标消失（流式结束）。
不需要等的时候就 `--ready-js 'true'`。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]
DEBUG_PORT = 9333

# 等到答案生成完：有内容、且流式的光标已经拿掉
DEFAULT_READY_JS = """(() => {
  const a = document.querySelector('.msg-a');
  if (!a) return false;
  if (a.innerText.trim().length < 80) return false;
  return !a.querySelector('.cursor');
})()"""


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    raise SystemExit("找不到 Chrome，改 CHROME_CANDIDATES")


async def capture(page_id: str, url: str, ready_js: str, out: Path,
                  timeout: int, size: tuple[int, int], settle: float) -> bool:
    async with websockets.connect(
        f"ws://127.0.0.1:{DEBUG_PORT}/devtools/page/{page_id}",
        max_size=64 * 1024 * 1024,
    ) as ws:
        counter = 0

        async def call(method: str, params: dict | None = None) -> dict:
            nonlocal counter
            counter += 1
            await ws.send(json.dumps({"id": counter, "method": method,
                                      "params": params or {}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == counter:
                    if "error" in msg:
                        raise RuntimeError(f"{method} 失败：{msg['error']}")
                    return msg.get("result", {})

        await call("Page.enable")
        await call("Emulation.setDeviceMetricsOverride", {
            "width": size[0], "height": size[1],
            "deviceScaleFactor": 1, "mobile": False,
        })
        await call("Page.navigate", {"url": url})

        deadline = time.monotonic() + timeout
        ready = False
        while time.monotonic() < deadline:
            await asyncio.sleep(1.5)
            res = await call("Runtime.evaluate", {
                "expression": ready_js, "returnByValue": True,
            })
            if res.get("result", {}).get("value") is True:
                ready = True
                break

        await asyncio.sleep(settle)   # 让字体/滚动动画落定
        shot = await call("Page.captureScreenshot", {"format": "png"})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(shot["data"]))
        return ready


def main() -> int:
    ap = argparse.ArgumentParser(description="CDP 截图，能等前端真正渲染完")
    ap.add_argument("url")
    ap.add_argument("-o", "--out", default="shot.png")
    ap.add_argument("--ready-js", default=DEFAULT_READY_JS,
                    help="返回 true 表示可以拍了")
    ap.add_argument("--timeout", type=int, default=120, help="最长等待秒数")
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=950)
    ap.add_argument("--settle", type=float, default=1.5,
                    help="就绪后再等几秒，避免拍到动画中途")
    args = ap.parse_args()

    chrome = subprocess.Popen(
        [find_chrome(), "--headless=new", "--disable-gpu", "--no-first-run",
         f"--remote-debugging-port={DEBUG_PORT}", "--remote-allow-origins=*",
         f"--window-size={args.width},{args.height}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        page_id = ""
        for _ in range(40):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{DEBUG_PORT}/json/list"
                ) as resp:
                    tabs = json.load(resp)
            except Exception:
                continue
            page = next((t for t in tabs if t["type"] == "page"), None)
            if page:
                page_id = page["id"]
                break
        if not page_id:
            print("Chrome 没起来（端口被占？换 DEBUG_PORT）", file=sys.stderr)
            return 1

        out = Path(args.out)
        ready = asyncio.run(capture(page_id, args.url, args.ready_js, out,
                                    args.timeout, (args.width, args.height),
                                    args.settle))
        print(f"{'已就绪' if ready else '超时（仍出图）'} → {out}")
        return 0 if ready else 1
    finally:
        chrome.terminate()


if __name__ == "__main__":
    sys.exit(main())
