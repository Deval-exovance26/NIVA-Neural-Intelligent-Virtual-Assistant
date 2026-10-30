import asyncio, base64, io
from playwright.async_api import async_playwright
from PIL import Image
import numpy as np

def log(*a):
    print(*a, flush=True)

MORPHS = [("neu",[0,0]),("talk",[0.35,0]),("blink",[0,1]),("jawfull",[1,0])]

async def cap(pg, tag):
    out = {}
    for name, w in MORPHS:
        await pg.evaluate("window.__forceMorph=%s" % w)
        await pg.wait_for_timeout(1500)
        d = await pg.evaluate("()=>document.getElementById('gfx').toDataURL('image/png')")
        out[name] = base64.b64decode(d.split(",")[1])
        log(tag, name, "cap", len(out[name]))
    return out

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=[
            "--use-gl=angle","--use-angle=swiftshader","--ignore-gpu-blocklist","--no-sandbox"])
        res = {}
        for tag, url in [("NEW","http://localhost:8000/index.html"),
                         ("REF","http://localhost:8001/index.html")]:
            log(tag, "goto")
            pg = await b.new_page(viewport={"width":480,"height":540})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            await pg.goto(url, wait_until="domcontentloaded", timeout=30000)
            log(tag, "loaded")
            await pg.wait_for_timeout(9000)
            log(tag, "warmed")
            mts = await pg.evaluate("()=> window.__mainThreadSorts")
            res[tag] = {"img": await cap(pg, tag), "mts": mts, "err": errs}
            await pg.close()
            log(tag, "closed mts=", mts, "err", errs or "none")
        await b.close()
        log("--- diff NEW vs REF ---")
        worst = 0.0
        for name, _ in MORPHS:
            ia = np.asarray(Image.open(io.BytesIO(res["NEW"]["img"][name])).convert("RGB")).astype("float32")
            ic = np.asarray(Image.open(io.BytesIO(res["REF"]["img"][name])).convert("RGB")).astype("float32")
            mad = float(np.abs(ia-ic).mean()); mx = float(np.abs(ia-ic).max())
            worst = max(worst, mad)
            log("%-8s meanAbsDiff=%.4f/255 maxDiff=%.0f" % (name, mad, mx))
        log("NEW __mainThreadSorts =", res["NEW"]["mts"])
        log("VERDICT:", "PIXEL-EQUIVALENT" if worst < 0.5 else "DIFFERS worst=%.3f" % worst)

asyncio.run(main())
