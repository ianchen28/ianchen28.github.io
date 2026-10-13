// Usage: node tests/image_browser.cjs BEFORE_URL CANDIDATE_URL OUTPUT_DIRECTORY
// The maintenance candidate is served from an isolated source copy.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [before, after, output] = process.argv.slice(2);
if (!before || !after || !output) throw new Error('Provide two preview URLs and an output directory');
fs.mkdirSync(output, { recursive: true });
(async () => {
  const browser = await chromium.launch({ headless: true });
  const results = [];
  try {
    for (const [label, root] of [['before', before], ['after', after]]) {
      for (const width of [390, 1440]) {
        const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: 'light' });
        await page.addInitScript(() => localStorage.setItem('pref-theme', 'light'));
        let broken = 0;
        for (const slug of ['tech/transformer', 'archive/torchcraftai-intro', 'archive/torchcraftai-experiment', 'archive/torchcraftai-code-learning.md']) {
          const route = '/zh/posts/' + slug + '/';
          assert.equal((await page.goto(root + route, { waitUntil: 'networkidle' })).status(), 200);
          const localImages = await page.locator('.post-content img').evaluateAll(images =>
            images.map(image => image.src).filter(url => new URL(url).origin === location.origin));
          for (const url of localImages) {
            const response = await page.request.get(url);
            if (response.status() >= 400) broken++;
            else assert(response.headers()['content-type'].startsWith('image/'), url);
          }
          if (slug === 'tech/transformer') {
            for (const image of await page.locator('.post-content img').all()) {
              await image.scrollIntoViewIfNeeded();
              await image.evaluate(e => e.complete ? undefined : new Promise(resolve => {
                e.addEventListener('load', resolve, { once: true });
                e.addEventListener('error', resolve, { once: true });
              }));
            }
            await page.evaluate(() => scrollTo(0, 0));
            await page.screenshot({ path: path.join(output, `zh-transformer-${width}-${label}.png`), fullPage: true });
          }
        }
        assert.equal(broken, label === 'before' ? 21 : 0);
        results.push({ label, width, brokenLocalImageReferences: broken });
        await page.close();
      }
    }
    fs.writeFileSync(path.join(output, 'image-browser-results.json'), JSON.stringify(results, null, 2) + '\n');
    console.log(JSON.stringify(results, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
