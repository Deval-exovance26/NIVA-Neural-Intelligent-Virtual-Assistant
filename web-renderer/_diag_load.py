import asyncio, base64
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=[
            "--use-gl=angle","--use-angle=swiftshader","--ignore-gpu-blocklist","--no-sandbox"])
        pg = await b.new_page(viewport={"width":640,"height":720})
        logs = []; errs = []
        pg.on("console", lambda m: logs.append(m.type + ": " + m.text))
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("http://localhost:8000/index.html", wait_until="load", timeout=30000)
        await pg.wait_for_timeout(7000)
        st = await pg.evaluate("()=>({hasR:!!window.__r, N: window.__r?window.__r.avatar.N:null, mts: window.__mainThreadSorts})")
        print("STATE", st)
        await pg.evaluate("window.__forceMorph=[0,1]")
        await pg.wait_for_timeout(2500)
        d = await pg.evaluate("()=>document.getElementById('gfx').toDataURL('image/png')")
        print("blink bytes:", len(base64.b64decode(d.split(',')[1])))
        print("PAGE_ERRORS:", errs or "none")
        print("CONSOLE:", logs[-12:])
        await b.close()

try:
    asyncio.run(asyncio.wait_for(run(), timeout=60))
except asyncio.TimeoutError:
    print("TIMEOUT: page hung")
