import asyncio, base64
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(headless=True,args=['--use-gl=angle','--use-angle=swiftshader','--ignore-gpu-blocklist','--no-sandbox'])
        pg=await b.new_page(viewport={'width':640,'height':720})
        logs=[]; pg.on('console',lambda m: logs.append(m.text)); pg.on('pageerror',lambda e: logs.append('ERR '+str(e)))
        await pg.goto('http://localhost:8000/index.html',wait_until='load',timeout=60000); await pg.wait_for_timeout(12000)
        for name,w in [('neu',[0,0]),('blink',[0,1]),('talk',[0.35,0]),('jawfull',[1,0])]:
            await pg.evaluate(f'window.__forceMorph={w}'); await pg.wait_for_timeout(2500)
            d=await pg.evaluate("()=>document.getElementById('gfx').toDataURL('image/png')")
            open(f'/tmp/c3_{name}.png','wb').write(base64.b64decode(d.split(',')[1]))
        print('\n'.join(l for l in logs if 'anim' in l.lower() or 'ERR' in l or 'rror' in l))
        await b.close(); print('ok')
asyncio.run(main())
