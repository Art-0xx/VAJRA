import subprocess
import time
import json
import urllib.request
import asyncio
import base64
import websockets

async def capture_tab(ws_url, tab_name, output_path, click_selector=None):
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
        await ws.send(json.dumps({"id": 2, "method": "Runtime.enable"}))
        
        if click_selector:
            # Click selector via Runtime.evaluate
            eval_expr = f"""
                (() => {{
                    const btns = Array.from(document.querySelectorAll('button'));
                    const target = btns.find(b => b.innerText.includes('{click_selector}'));
                    if (target) {{
                        target.click();
                        return true;
                    }}
                    return false;
                }})()
            """
            await ws.send(json.dumps({
                "id": 3,
                "method": "Runtime.evaluate",
                "params": {"expression": eval_expr}
            }))
            await asyncio.sleep(1.0)
            
        # Capture Screenshot
        await ws.send(json.dumps({
            "id": 4,
            "method": "Page.captureScreenshot",
            "params": {"format": "png"}
        }))
        
        while True:
            resp = json.loads(await ws.recv())
            if resp.get("id") == 4 and "result" in resp:
                data = base64.b64decode(resp["result"]["data"])
                with open(output_path, "wb") as f:
                    f.write(data)
                print(f"[+] Saved {tab_name} to {output_path}")
                break

async def main():
    p = subprocess.Popen([
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless",
        "--remote-debugging-port=9225",
        "--disable-gpu",
        "--window-size=1400,1200",
        "http://localhost:3000"
    ])
    time.sleep(2.5)
    try:
        tabs = json.loads(urllib.request.urlopen("http://localhost:9225/json").read().decode())
        target = [t for t in tabs if "localhost:3000" in t["url"]][0]
        ws_url = target["webSocketDebuggerUrl"]
        
        artifact_dir = r"C:\Users\admin\.gemini\antigravity\brain\ab8df76c-0a05-4c82-af72-d4e65548be66"
        
        # 1. Capture Tab 1: Live Capture
        await capture_tab(ws_url, "Tab 1 Live Capture", f"{artifact_dir}\\screenshot_tab1_live.png")
        
        # 2. Capture Tab 2: Red Team
        await capture_tab(ws_url, "Tab 2 Red Team", f"{artifact_dir}\\screenshot_tab2_redteam.png", "2. RED TEAM")
        
        # 3. Capture Tab 3: Blue Team
        await capture_tab(ws_url, "Tab 3 Blue Team", f"{artifact_dir}\\screenshot_tab3_blueteam.png", "3. BLUE TEAM")
        
        # 4. Capture Tab 4: Metrics
        await capture_tab(ws_url, "Tab 4 Metrics", f"{artifact_dir}\\screenshot_tab4_metrics.png", "4. METRICS")
    finally:
        p.kill()

if __name__ == "__main__":
    asyncio.run(main())

