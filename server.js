const express = require('express');
const fetch = require('node-fetch');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

const DATA_API = 'https://data-api.polymarket.com';

// ── In-memory cache ──────────────────────────────────────────────────────────
const cache = {};
function getCached(key) {
  const entry = cache[key];
  if (entry && Date.now() - entry.ts < entry.ttl) return entry.data;
  return null;
}
function setCache(key, data, ttlMs) {
  cache[key] = { data, ts: Date.now(), ttl: ttlMs };
}

// ── Fetch with timeout + retry ───────────────────────────────────────────────
async function apiFetch(url, timeoutMs = 12000, retries = 2) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      const res = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: 'application/json', 'User-Agent': 'PolyWhaleTracker/1.0' },
      });
      clearTimeout(timer);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      if (attempt === retries) throw err;
      await new Promise(r => setTimeout(r, 600 * (attempt + 1)));
    }
  }
}

// ── Concurrency-limited batch fetcher ───────────────────────────────────────
async function batchFetch(tasks, concurrency = 5) {
  const results = new Array(tasks.length).fill(null);
  let idx = 0;
  async function worker() {
    while (idx < tasks.length) {
      const i = idx++;
      try { results[i] = await tasks[i](); }
      catch { results[i] = null; }
    }
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
  return results;
}

// ── Serve static files ───────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));

// ── GET /api/leaderboard ─────────────────────────────────────────────────────
app.get('/api/leaderboard', async (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 50, 100);
  const period = req.query.period || 'MONTH';
  const key = `leaderboard:${period}:${limit}`;
  const hit = getCached(key);
  if (hit) return res.json(hit);
  try {
    const data = await apiFetch(`${DATA_API}/leaderboard?period=${period}&limit=${limit}`);
    setCache(key, data, 5 * 60 * 1000); // 5 min
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

// ── GET /api/positions ───────────────────────────────────────────────────────
app.get('/api/positions', async (req, res) => {
  const { user } = req.query;
  if (!user) return res.status(400).json({ error: 'user param required' });
  const key = `positions:${user}`;
  const hit = getCached(key);
  if (hit) return res.json(hit);
  try {
    const data = await apiFetch(`${DATA_API}/positions?user=${user}&limit=100`);
    setCache(key, data, 3 * 60 * 1000); // 3 min
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

// ── GET /api/whale-signals ───────────────────────────────────────────────────
// One-shot endpoint: fetches leaderboard + all positions, aggregates signals.
// Client makes a single request; server handles all fan-out concurrently.
app.get('/api/whale-signals', async (req, res) => {
  const cacheKey = 'whale-signals';
  const hit = getCached(cacheKey);
  if (hit) return res.json(hit);

  try {
    // 1. Leaderboard
    const period = req.query.period || 'MONTH';
    const topN = Math.min(parseInt(req.query.top) || 25, 50);
    const lb = await apiFetch(`${DATA_API}/leaderboard?period=${period}&limit=${topN}`);
    const traders = Array.isArray(lb) ? lb : (lb.data || lb.leaderboard || []);

    if (!traders.length) return res.json({ traders: [], signals: [] });

    // 2. Positions for every trader — batched, 6 concurrent
    const tasks = traders.map(t => () =>
      apiFetch(`${DATA_API}/positions?user=${t.proxyWallet}&limit=100`)
        .catch(() => null)
    );
    const posResults = await batchFetch(tasks, 6);

    // 3. Attach positions to each trader
    const enriched = traders.map((t, i) => {
      const raw = posResults[i];
      const positions = Array.isArray(raw)
        ? raw
        : (raw && Array.isArray(raw.data) ? raw.data : []);
      return { ...t, positions };
    });

    // 4. Aggregate signals by conditionId
    const byMarket = {};
    enriched.forEach((trader, i) => {
      const rank = (typeof trader.rank === 'number' ? trader.rank : i + 1);
      const rankWeight = 1 / rank;
      trader.positions.forEach(pos => {
        if (!pos.conditionId) return;
        if (!byMarket[pos.conditionId]) {
          byMarket[pos.conditionId] = {
            conditionId: pos.conditionId,
            title: pos.title || '—',
            outcome: pos.outcome,
            whales: [],
            yesWeightedValue: 0,
            noWeightedValue: 0,
          };
        }
        const entry = byMarket[pos.conditionId];
        const wv = (pos.currentValue || 0) * rankWeight;
        entry.whales.push({
          address: trader.proxyWallet,
          rank,
          outcome: pos.outcome,
          size: pos.size,
          currentValue: pos.currentValue || 0,
          cashPnl: pos.cashPnl || 0,
          percentPnl: pos.percentPnl || 0,
          avgPrice: pos.avgPrice || 0,
          price: pos.price || pos.currentPrice || 0,
        });
        if ((pos.outcome || '').toUpperCase() === 'YES') entry.yesWeightedValue += wv;
        else entry.noWeightedValue += wv;
      });
    });

    // 5. Build final signal list
    const signals = Object.values(byMarket)
      .filter(m => m.whales.length >= 2)
      .map(m => {
        const totalWV = m.yesWeightedValue + m.noWeightedValue;
        const direction = m.yesWeightedValue >= m.noWeightedValue ? 'YES' : 'NO';
        const yesCount = m.whales.filter(w => w.outcome?.toUpperCase() === 'YES').length;
        const noCount  = m.whales.length - yesCount;
        const avgPrice = m.whales.reduce((s, w) => s + (w.price || w.avgPrice || 0), 0) / m.whales.length;
        const totalValue = m.whales.reduce((s, w) => s + w.currentValue, 0);
        return {
          conditionId: m.conditionId,
          title: m.title,
          direction,
          convictionScore: Math.round(totalWV * 100) / 100,
          whaleCount: m.whales.length,
          yesCount,
          noCount,
          avgPrice: Math.round(avgPrice * 1000) / 1000,
          totalWhaleValue: Math.round(totalValue * 100) / 100,
          whales: m.whales.sort((a, b) => a.rank - b.rank),
        };
      })
      .sort((a, b) => b.convictionScore - a.convictionScore);

    const payload = {
      fetchedAt: new Date().toISOString(),
      period,
      traders: enriched.map(t => ({
        rank: t.rank,
        proxyWallet: t.proxyWallet,
        userName: t.userName || null,
        xUsername: t.xUsername || null,
        pnl: t.pnl ?? t.PnL ?? t.profit ?? 0,
        volume: t.volume ?? 0,
        positionCount: t.positions.length,
        positions: t.positions,
      })),
      signals,
    };

    setCache(cacheKey, payload, 3 * 60 * 1000); // 3 min
    res.json(payload);
  } catch (err) {
    console.error('whale-signals error:', err.message);
    res.status(502).json({ error: err.message });
  }
});

// ── Fallback → SPA ───────────────────────────────────────────────────────────
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`\n  Polymarket Whale Tracker → http://localhost:${PORT}\n`);
});
