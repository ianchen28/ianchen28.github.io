// Existing Playwright installation: PLAYWRIGHT_MODULE=/absolute/path/to/playwright
// Usage: node tests/portfolio_browser.cjs URL ARTIFACT_DIRECTORY [--baseline]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [root, output, mode] = process.argv.slice(2);
if (!root || !output) throw new Error('Provide preview URL and artifact directory');
fs.mkdirSync(output, { recursive: true });

function contrast(foreground, background) {
  const luminance = rgb => rgb.match(/[\d.]+/g).slice(0, 3)
    .map(v => Number(v) / 255)
    .map(v => v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)
    .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
  const a = luminance(foreground), b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const results = [], failures = [], errors = [], localLinks = new Set();
  const check = (label, fn) => {
    try { fn(); } catch (error) { failures.push({ label, error: error.message }); }
  };
  try {
    const widths = mode === '--baseline' ? [390, 1440] : [320, 390, 768, 1050, 1440];
    for (const width of widths) for (const theme of ['light', 'dark']) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, colorScheme: theme });
      await context.addInitScript(t => localStorage.setItem('pref-theme', t), theme);
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      for (const prefix of ['', '/zh']) for (const suffix of ['/', '/projects/agent-improvement-lab/', '/projects/', '/about/', '/posts/', '/posts/tech/llm-agents/']) {
        const route = prefix + suffix;
        const response = await page.goto(root + route, { waitUntil: 'networkidle' });
        const label = `${route} ${width} ${theme}`;
        check(label + ' response', () => assert.equal(response.status(), 200));
        const data = await page.evaluate(() => ({
          lang: document.documentElement.lang,
          h1: document.querySelectorAll('h1').length,
          overflow: document.documentElement.scrollWidth - innerWidth,
          menuOverflow: document.querySelector('#menu').scrollWidth - document.querySelector('#menu').clientWidth,
          menuRects: [...document.querySelectorAll('#menu a')].map(a => {
            const r = a.getBoundingClientRect(); return { left: r.left, right: r.right, width: r.width, height: r.height };
          }),
          counterpart: new URL(document.querySelector('.lang-switch a').href).pathname,
          bodyDark: document.body.classList.contains('dark'),
          themeName: document.querySelector('#theme-toggle').getAttribute('aria-label'),
          text: document.querySelector('main').innerText,
          brokenImages: [...document.images].filter(i => !i.complete || !i.naturalWidth).map(i => i.src),
          schema: [...document.querySelectorAll('script[type="application/ld+json"]')].map(s => JSON.parse(s.textContent)),
          cardText: [...document.querySelectorAll('.portfolio-writing p')].map(p => p.textContent),
        }));
        check(label + ' semantics', () => { assert.equal(data.lang, prefix ? 'zh' : 'en'); assert.equal(data.h1, 1); });
        check(label + ' mobile navigation', () => {
          assert(data.overflow <= 1); assert(data.menuOverflow <= 1, `Navigation hides ${data.menuOverflow}px`);
          assert(data.menuRects.every(r => r.left >= 0 && r.right <= width + 1 && r.width >= 24 && r.height >= 24));
        });
        check(label + ' translation', () => assert.equal(data.counterpart, prefix ? suffix : '/zh' + suffix));
        check(label + ' theme name', () => assert.equal(data.themeName, prefix ? '切换明暗主题' : 'Toggle color theme'));
        check(label + ' public content', () => {
          if (!suffix.startsWith('/posts/')) {
            assert(!/owner confirmation|Biography Status|所有者确认|个人信息状态|open.source|开源/i.test(data.text));
          }
          assert.deepEqual(data.brokenImages, []);
        });
        if (await page.locator('.portfolio-featured').count()) {
          const statusStyle = await page.locator('.portfolio-featured .portfolio-status').first().evaluate(e => ({
            color: getComputedStyle(e).color,
            background: getComputedStyle(e.closest('.portfolio-card')).backgroundColor,
          }));
          check(label + ' status contrast', () => assert(contrast(statusStyle.color, statusStyle.background) >= 4.5));
        }
        for (const href of await page.locator('a[href]').evaluateAll(a => a.map(e => e.href))) {
          if (new URL(href).origin === new URL(root).origin) localLinks.add(href.split('#')[0]);
        }
        if (suffix.includes('agent-improvement-lab')) {
          check(label + ' project schema', () => {
            const schema = data.schema.find(s => s['@type'] === 'WebPage');
            assert(schema, 'Project is still represented as a blog post');
            assert.equal(schema.inLanguage, prefix ? 'zh' : 'en');
            assert(!('datePublished' in schema)); assert(!('dateModified' in schema));
            assert(!data.schema.some(s => s['@type'] === 'BlogPosting'));
          });
          const style = await page.locator('.agent-loop-stages small').first().evaluate(e => ({
            font: parseFloat(getComputedStyle(e).fontSize), color: getComputedStyle(e).color,
            background: getComputedStyle(e.parentElement).backgroundColor,
          }));
          check(label + ' diagram readability', () => { assert(style.font >= 13); assert(contrast(style.color, style.background) >= 4.5); });
          assert.equal(await page.locator('.agent-loop-stages > li').count(), 7);
          assert.equal(await page.locator('.heldout-stages > li').count(), 4);
          assert.equal(await page.locator('.agent-loop-track .heldout-evaluation').count(), 0);
          assert(await page.locator(width <= 1100 ? '.agent-feedback-vertical' : '.agent-feedback-horizontal').isVisible());
          await page.locator('.harness-architecture summary').focus(); await page.keyboard.press('Enter');
          assert(await page.locator('.harness-architecture').evaluate(e => e.open));
          await page.keyboard.press('Enter');
        } else if (suffix === '/') {
          if (!prefix) check(label + ' English writing descriptions', () => {
            assert.equal(data.cardText.length, 2);
            assert(data.cardText.every(t => t.trim() && !/[\u3400-\u9fff]/.test(t)));
          });
          const projectLink = page.locator('.portfolio-featured .portfolio-actions a').first();
          await projectLink.focus();
          assert.notEqual(await projectLink.evaluate(e => getComputedStyle(e).outlineStyle), 'none');
        } else if (suffix === '/projects/') {
          check(label + ' project cards', () => assert(data.text.includes(prefix ? '进行中：' : 'In progress:')));
          check(label + ' concise project list', () => assert(!data.text.includes(prefix ? '开发评估器' : 'Development Evaluator')));
        }
        await page.locator('#theme-toggle').click();
        assert.equal(await page.locator('body').evaluate(e => e.classList.contains('dark')), !data.bodyDark);
        await page.locator('#theme-toggle').click();
        if ([390, 1440].includes(width) && !suffix.startsWith('/posts/')) {
          const surface = suffix === '/' ? 'home' : suffix.includes('agent-improvement-lab') ? 'project' : suffix.includes('projects') ? 'projects' : 'about';
          await page.screenshot({ path: path.join(output, `${prefix ? 'zh' : 'en'}-${surface}-${width}-${theme}.png`), fullPage: true });
        }
        results.push({ route, width, theme, status: response.status() });
      }
      await context.close();
    }
    const page = await browser.newPage();
    for (const prefix of ['', '/zh']) {
      await page.goto(root + prefix + '/search/', { waitUntil: 'networkidle' });
      await page.locator('#searchInput').pressSequentially('Agent Improvement Lab');
      await page.waitForFunction(() => document.querySelectorAll('#searchResults li').length > 0);
      assert((await page.locator('#searchResults a').evaluateAll(a => a.map(e => new URL(e.href).pathname)))
        .includes(prefix + '/projects/agent-improvement-lab/'));
      await page.locator('#searchInput').fill('Transformer'); await page.locator('#searchInput').press('ArrowRight');
      assert((await page.locator('#searchResults').innerText()).includes('Transformer'));
    }
    for (const url of localLinks) assert((await page.request.get(url)).status() < 400, url);
    assert.deepEqual(errors, []);
    const report = { cases: results.length, failures, errors, internalLinks: localLinks.size, bilingualSearch: true, results };
    fs.writeFileSync(path.join(output, 'browser-results.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ cases: report.cases, failures, errors, internalLinks: localLinks.size }, null, 2));
    if (failures.length) process.exitCode = 1;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
