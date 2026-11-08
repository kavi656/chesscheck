"""Streamlit dashboard. Runs entirely from cached results in results/ (no API calls).

    streamlit run src/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"

# --- palette (validated: dataviz validate_palette.js --mode dark, all checks pass) ---
SURFACE, PANEL, BORDER = "#141413", "#1f1f1d", "#2e2e2b"
TEXT, TEXT2, MUTED, GRID = "#f0efec", "#c3c2b7", "#8a8984", "#383835"
GOOD = "#0ca30c"
STAGES = ["proposed", "explore", "corrected", "validate", "skeptic", "vault", "REAL"]
KILL = {  # kill category -> (label, colour). Grey = never looked real; colour = looked real, then died.
    "untestable": ("Invalid / duplicate / underpowered", "#5f5e5a"),
    "never": ("Never significant (REJECTED)", MUTED),
    "correction": ("Killed by multiple-testing correction", "#3987e5"),
    "min_effect": ("Killed: effect below min_effect", "#d95926"),
    "validate": ("Killed at validation", "#199e70"),
    "skeptic": ("Killed by skeptic checks", "#c98500"),
    "vault": ("Killed in the vault", "#d55181"),
    "real": ("REAL", GOOD),
}

st.set_page_config(page_title="ChessCheck", layout="wide", page_icon=str(ROOT / "assets" / "logo-64.png"), initial_sidebar_state="collapsed")
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, h5, input, textarea,
button[data-baseweb="tab"] p, [data-testid="stMetricValue"] {{ font-family: 'Inter', sans-serif; }}
code, .mono {{ font-family: 'JetBrains Mono', monospace !important; }}
.stApp {{ background: {SURFACE}; color: {TEXT}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
.hero {{ position: relative; padding: 6px 0 0 0; margin: 0 0 1.2rem 0; }}
@keyframes drift {{ from {{ transform: translate(0, 0); }} to {{ transform: translate(-120px, 60px); }} }}
.hero-title {{ position: relative; font-size: 3rem !important; font-weight: 800; letter-spacing: -0.03em; margin: 0; line-height: 1.1;
               background: linear-gradient(90deg, #f0efec 0%, #8fc0ff 45%, #6fe06f 70%, #f0efec 100%); background-size: 220% auto;
               -webkit-background-clip: text; background-clip: text; color: transparent; animation: shine 6s linear infinite; }}
@keyframes shine {{ to {{ background-position: 220% center; }} }}
.hero-sub {{ position: relative; }}
.hero-row {{ position: relative; display: flex; align-items: center; gap: 16px; }}
.hero-row svg {{ flex: 0 0 auto; filter: drop-shadow(0 6px 16px rgba(0,0,0,0.5)); }}
.hero-tag {{ position: relative; font-size: 1.35rem !important; font-weight: 600; color: {TEXT}; margin: 0.5rem 0 0.2rem 0; }}
@property --n {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
.cnt {{ --n: var(--to); counter-reset: n var(--n); animation: cnt 1.6s cubic-bezier(.17,.67,.2,1) both; }}
.cnt::after {{ content: counter(n); }}
@keyframes cnt {{ from {{ --n: 0; }} }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes glow {{ 0%, 100% {{ box-shadow: 0 0 0 0 rgba(63,191,63,0.0); }} 50% {{ box-shadow: 0 0 22px 2px rgba(63,191,63,0.35); }} }}
.stMarkdown h4 {{ position: relative; padding-left: 14px; }}
.stMarkdown h4::before {{ content: ""; position: absolute; left: 0; top: 18%; bottom: 18%; width: 4px; border-radius: 4px;
                          background: linear-gradient(180deg, #3987e5, #3fbf3f); }}
.hero-sub {{ color: {TEXT2}; font-size: 1.08rem !important; margin: 0.35rem 0 1.4rem 0; max-width: 900px; line-height: 1.5; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 1.6rem; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.kpi {{ --acc: #3987e5; position: relative; overflow: hidden; background: linear-gradient(180deg, rgba(255,255,255,0.025), transparent), {PANEL};
        border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 18px 16px 18px;
        animation: rise 0.6s ease-out both; transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }}
.kpi::before {{ content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; background: var(--acc); }}
.kpi:hover {{ transform: translateY(-3px); border-color: var(--acc); box-shadow: 0 10px 30px -12px var(--acc); }}
.kpi:nth-child(2) {{ animation-delay: .08s; }} .kpi:nth-child(3) {{ animation-delay: .16s; }} .kpi:nth-child(4) {{ animation-delay: .24s; }}
.kpi.good {{ --acc: #3fbf3f; }} .kpi.blue {{ --acc: #3987e5; }} .kpi.orange {{ --acc: #ef8a5c; }} .kpi.violet {{ --acc: #9085e9; }}
.kpi .v {{ font-size: 2.1rem; font-weight: 800; color: {TEXT}; line-height: 1.1; letter-spacing: -0.02em; }}
.kpi .v .of {{ font-size: 1.1rem; color: {MUTED}; font-weight: 600; }}
.kpi .l {{ color: {TEXT}; font-weight: 600; margin-top: 6px; font-size: 0.95rem; }}
.kpi .s {{ color: {MUTED}; font-size: 0.85rem; margin-top: 2px; line-height: 1.35; }}
.kpi.good .v {{ color: #4fd04f; }}
[data-baseweb="tab-list"] {{ gap: 6px; }}
button[data-baseweb="tab"] p {{ font-size: 0.98rem; font-weight: 600; color: {TEXT2}; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color: {TEXT}; }}
.card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px; height: 100%;
         animation: rise 0.55s ease-out both; transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }}
.card:hover {{ transform: translateY(-3px); border-color: #3a3a37; box-shadow: 0 12px 28px -14px rgba(0,0,0,0.8); }}
.card.pos {{ border-top: 3px solid #3fbf3f; }} .card.neg {{ border-top: 3px solid #ef8a5c; }}
.card.pos .num {{ color: #4fd04f; }} .card.neg .num {{ color: #f39a70; }}
.card h5 {{ margin: 0 0 6px 0; font-size: 1rem; font-weight: 700; color: {TEXT}; }}
.card p {{ margin: 0; color: {TEXT2}; font-size: 0.92rem; line-height: 1.5; }}
.card .num {{ font-size: 1.6rem; font-weight: 800; color: {TEXT}; letter-spacing: -0.02em; }}
.claim {{ animation: rise 0.5s ease-out both; transition: border-color .2s ease, transform .2s ease; }}
.claim:hover {{ border-color: #3a3a37; transform: translateX(3px); }}
.claim:nth-child(2) {{ animation-delay: .07s; }} .claim:nth-child(3) {{ animation-delay: .14s; }}
.claim:nth-child(4) {{ animation-delay: .21s; }} .claim:nth-child(5) {{ animation-delay: .28s; }}
.claim {{ display: flex; gap: 14px; align-items: flex-start; background: {PANEL}; border: 1px solid {BORDER};
          border-radius: 14px; padding: 14px 18px; margin-bottom: 10px; }}
.claim .tick {{ flex: 0 0 34px; height: 34px; border-radius: 50%; background: rgba(12,163,12,0.15); color: #3fbf3f;
                display: flex; align-items: center; justify-content: center; font-weight: 800; }}
.claim .warn {{ background: rgba(201,133,0,0.15); color: #e0a020; }}
.claim b {{ color: {TEXT}; }}
.claim div.t {{ color: {TEXT2}; line-height: 1.5; font-size: 0.95rem; }}
.flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 0.4rem 0 1.2rem 0; }}
.flow .st {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 8px 12px; text-align: center; min-width: 92px;
             animation: rise 0.45s ease-out both; }}
.flow .ar {{ animation: rise 0.45s ease-out both; }}
.flow .st .n {{ font-size: 1.25rem; font-weight: 800; color: {TEXT}; }}
.flow .st .k {{ font-size: 0.78rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.04em; }}
.flow .st.real {{ border-color: #0ca30c; background: rgba(12,163,12,0.10); animation: rise 0.45s ease-out both, glow 2.4s ease-in-out 1.2s infinite; }}
.flow .st.real .n {{ color: #4fd04f; }}
.flow .ar {{ color: {MUTED}; font-size: 1.1rem; }}
.pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }}
.pill.real {{ background: rgba(12,163,12,0.18); color: #3fbf3f; }}
.pill.lucky {{ background: rgba(213,81,129,0.18); color: #e57aa0; }}
.pill.rej {{ background: rgba(138,137,132,0.2); color: {TEXT2}; }}
.src {{ color: {MUTED}; font-size: 0.88rem; line-height: 1.7; }}
.src a {{ color: #7fb0f0; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; }}
.topbar {{ display: flex; align-items: center; gap: 12px; margin: 0 0 0.6rem 0; }}
.topbar .brand {{ font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; color: {TEXT}; }}
.topbar .brand-tag {{ color: {MUTED}; font-size: 1rem; margin-left: 6px; }}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {{ position: sticky; top: 0; z-index: 50; padding-top: 6px;
    background: rgba(18,18,17,0.82); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }}
.story {{ max-width: 1100px; }}
.fact {{ min-height: 78vh; display: flex; flex-direction: column; justify-content: center; gap: 18px; padding: 6vh 0; }}
.hero-fact {{ min-height: 86vh; }}
.eyebrow {{ font-size: 0.9rem !important; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #7fb0f0; }}
.mega {{ font-size: clamp(2.8rem, 6.4vw, 5.6rem) !important; font-weight: 800; line-height: 1.04; letter-spacing: -0.035em; color: {TEXT}; }}
.big {{ font-size: clamp(2.2rem, 4.6vw, 4rem) !important; font-weight: 800; line-height: 1.08; letter-spacing: -0.03em; color: {TEXT}; }}
.giant {{ font-size: clamp(4.5rem, 13vw, 10.5rem) !important; font-weight: 800; line-height: 0.95; letter-spacing: -0.05em; color: {TEXT}; }}
.giant .unit {{ font-size: 0.42em; letter-spacing: -0.02em; color: {TEXT2}; font-weight: 700; }}
.giant sup {{ font-size: 0.45em; vertical-align: super; }}
.good {{ color: #4fd04f !important; }} .warm {{ color: #f39a70 !important; }} .blue {{ color: #7fb0f0 !important; }} .violet {{ color: #a99ff0 !important; }}
.grad {{ background: linear-gradient(90deg, #8fc0ff, #8ff08f); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.lede {{ font-size: clamp(1.1rem, 1.6vw, 1.45rem) !important; line-height: 1.55; color: {TEXT2}; max-width: 820px; margin: 0; }}
.lede b {{ color: {TEXT}; }}
.scroll-cue {{ margin-top: 4vh; color: {MUTED}; font-size: 0.85rem; letter-spacing: 0.2em; text-transform: uppercase; display: flex; flex-direction: column; gap: 6px; }}
.scroll-cue span {{ font-size: 1.6rem; animation: bob 1.8s ease-in-out infinite; }}
@keyframes bob {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(8px); }} }}
.big-flow .st {{ min-width: 110px; padding: 12px 14px; }} .big-flow .st .n {{ font-size: 1.7rem; }}
.bars {{ display: flex; flex-direction: column; gap: 14px; max-width: 760px; margin-top: 8px; }}
.barrow {{ display: grid; grid-template-columns: 170px 1fr 70px; gap: 14px; align-items: center; color: {TEXT2}; }}
.bar {{ height: 16px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow: hidden; }}
.bar i {{ display: block; height: 100%; width: 0; border-radius: 999px; transition: width 1.4s cubic-bezier(.2,.7,.2,1) .3s; }}
.bar i.warm {{ background: #ef8a5c; }} .bar i.good {{ background: #4fd04f; }}
.bv {{ font-weight: 700; color: {TEXT}; }}
.fgrid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; margin-top: 10px; }}
.fcard {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; padding: 18px; }}
.fcard.pos {{ border-top: 3px solid #3fbf3f; }} .fcard.neg {{ border-top: 3px solid #ef8a5c; }}
.fcard .fnum {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }}
.fcard.pos .fnum {{ color: #4fd04f; }} .fcard.neg .fnum {{ color: #f39a70; }}
.fcard .ftxt {{ color: {TEXT2}; font-size: 0.98rem; line-height: 1.45; margin-top: 6px; }}
.hash {{ font-family: 'JetBrains Mono', monospace; font-size: clamp(0.8rem, 1.3vw, 1.1rem); color: {TEXT2}; background: {PANEL};
         border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 18px; word-break: break-all; max-width: 900px; }}
.limits {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; margin-top: 10px; }}
.lim {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; padding: 18px; display: flex; flex-direction: column; gap: 6px; }}
.lim b {{ color: {TEXT}; font-size: 1.1rem; }} .lim span {{ color: {TEXT2}; line-height: 1.45; }}
.outro {{ min-height: 60vh; }}
/* scroll reveal: only hidden once the script has loaded (cc-js), so nothing is lost without it */
.cc-js .reveal {{ opacity: 0; transform: translateY(48px); transition: opacity .9s ease, transform .9s cubic-bezier(.2,.7,.2,1); }}
.cc-js .reveal.in {{ opacity: 1; transform: none; }}
.cc-js .reveal .st, .cc-js .reveal .fcard, .cc-js .reveal .lim {{ opacity: 0; transform: translateY(18px);
    transition: opacity .6s ease, transform .6s ease; }}
.cc-js .reveal.in .st, .cc-js .reveal.in .fcard, .cc-js .reveal.in .lim {{ opacity: 1; transform: none; }}
.cc-js .reveal.in .bar i {{ width: var(--w); }}
.cc-js .reveal:not(.in) .cnt {{ animation: none; --n: 0; }}
.cc-js .reveal.in .cnt {{ animation: cnt 1.8s cubic-bezier(.17,.67,.2,1) .15s both; }}
.live-badge {{ display: inline-flex; align-items: center; gap: 8px; font-weight: 800; color: #ff5d5d;
               letter-spacing: 0.08em; font-size: 0.85rem; }}
.live-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #ff5d5d; box-shadow: 0 0 0 0 rgba(255,93,93,0.7);
             animation: pulse 1.6s infinite; }}
@keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(255,93,93,0.6); }} 70% {{ box-shadow: 0 0 0 12px rgba(255,93,93,0); }}
                    100% {{ box-shadow: 0 0 0 0 rgba(255,93,93,0); }} }}
@keyframes slidein {{ from {{ opacity: 0; transform: translateY(-6px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes countup {{ from {{ opacity: 0.2; transform: scale(0.92); }} to {{ opacity: 1; transform: none; }} }}
.kpi .v {{ animation: countup 0.6s ease-out; }}
.ticker {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: 6px 4px; }}
.tick-row {{ display: grid; grid-template-columns: 46px minmax(0,1fr) 40px minmax(0,1fr) 36px; gap: 8px; align-items: center;
             padding: 7px 12px; white-space: nowrap; border-bottom: 1px solid {BORDER}; font-size: 0.9rem; color: {TEXT2};
             animation: slidein 0.5s ease-out both; }}
.tick-row:last-child {{ border-bottom: none; }}
.tick-row .t {{ color: {MUTED}; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }}
.tick-row span {{ overflow: hidden; text-overflow: ellipsis; }}
.tick-row .w {{ text-align: right; }} .tick-row .res {{ text-align: center; font-weight: 700; color: {TEXT}; }}
.tick-row .win {{ color: {TEXT}; font-weight: 700; }}
.tick-row .tc {{ color: {MUTED}; font-size: 0.8rem; text-align: right; }}
[data-testid="stMetricValue"] {{ color: {TEXT}; }}
[data-testid="stMetricLabel"] p {{ color: {TEXT2}; }}
.big {{ font-size: 2.6rem; font-weight: 700; line-height: 1.1; }}
.sub {{ color: {TEXT2}; font-size: 0.95rem; }}
.explain {{ background: {PANEL}; border-left: 3px solid #3987e5; padding: 0.7rem 1rem; margin: 0.4rem 0 1rem 0;
            color: {TEXT2}; border-radius: 4px; font-size: 0.95rem; }}
</style>""", unsafe_allow_html=True)

# Interactive chessboard background: a full-page canvas injected behind the app (the component iframe is
# same-origin, so it can reach the parent page). Squares bulge away from the cursor like a lens.
BOARD_JS = """
<script>
(function () {
  const P = window.parent, D = P.document;
  if (D.getElementById('cc-board')) return;           // inject once per page load
  const style = D.createElement('style');
  style.textContent = `
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], header[data-testid="stHeader"],
    [data-testid="stBottomBlockContainer"] { background: transparent !important; }
    body { background: #121211 !important; }
    #cc-board { position: fixed; inset: 0; z-index: -1; pointer-events: none; }
    iframe[title="st.iframe"][height="0"] { display: none; }`;
  D.head.appendChild(style);
  const c = D.createElement('canvas'); c.id = 'cc-board'; D.body.prepend(c);
  const ctx = c.getContext('2d');
  const CELL = 64, R = 170, STRENGTH = 0.18;
  const reduce = P.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W = 0, H = 0, dpr = 1;
  let mx = -9999, my = -9999, tx = -9999, ty = -9999, amp = 0, tamp = 0, raf = null;
  function resize() {
    dpr = P.devicePixelRatio || 1; W = P.innerWidth; H = P.innerHeight;
    c.width = W * dpr; c.height = H * dpr; c.style.width = W + 'px'; c.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); kick();
  }
  function warp(x, y) {
    const dx = x - mx, dy = y - my, d2 = dx * dx + dy * dy;
    const f = amp * STRENGTH * Math.exp(-d2 / (2 * R * R));
    return [x + dx * f, y + dy * f, f];
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const cols = Math.ceil(W / CELL) + 1, rows = Math.ceil(H / CELL) + 1;
    const pts = [];
    for (let j = 0; j <= rows; j++) { const row = [];
      for (let i = 0; i <= cols; i++) row.push(warp(i * CELL, j * CELL)); pts.push(row); }
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      const a = pts[j][i], b = pts[j][i + 1], d = pts[j + 1][i + 1], e = pts[j + 1][i];
      const light = (i + j) % 2 === 0;
      const lift = Math.min(1, (a[2] + d[2]) * 1.6);       // brighter where the lens is
      const base = light ? 27 : 20;
      const r = base + 3 * lift, g = base + 4 * lift, bl = base + 8 * lift;
      ctx.fillStyle = `rgb(${r|0},${g|0},${bl|0})`;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(d[0], d[1]); ctx.lineTo(e[0], e[1]);
      ctx.closePath(); ctx.fill();
    }
    if (amp > 0.01) {                                      // soft glow under the cursor
      const g = ctx.createRadialGradient(mx, my, 0, mx, my, R * 1.3);
      g.addColorStop(0, `rgba(57,135,229,${0.04 * amp})`); g.addColorStop(1, 'rgba(57,135,229,0)');
      ctx.fillStyle = g; ctx.fillRect(mx - R * 1.3, my - R * 1.3, R * 2.6, R * 2.6);
    }
  }
  function tick() {
    mx += (tx - mx) * 0.18; my += (ty - my) * 0.18; amp += (tamp - amp) * 0.12;
    draw();
    if (Math.abs(tx - mx) > 0.3 || Math.abs(ty - my) > 0.3 || Math.abs(tamp - amp) > 0.005) raf = P.requestAnimationFrame(tick);
    else raf = null;
  }
  function kick() { if (!raf) raf = P.requestAnimationFrame(tick); }
  if (!reduce) {
    P.addEventListener('mousemove', ev => {
      if (mx < -9000) { mx = ev.clientX; my = ev.clientY; }
      tx = ev.clientX; ty = ev.clientY; tamp = 1; kick(); }, { passive: true });
    D.addEventListener('mouseleave', () => { tamp = 0; kick(); });
  }
  P.addEventListener('resize', resize);
  resize();
  // scroll reveal: mark .reveal sections "in" as they enter the viewport (re-armed on every Streamlit rerender)
  D.documentElement.classList.add('cc-js');
  const io = new P.IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.18 });
  const arm = () => D.querySelectorAll('.reveal:not(.in):not([data-io])').forEach(el => { el.dataset.io = 1; io.observe(el); });
  new P.MutationObserver(arm).observe(D.body, { childList: true, subtree: true });
  arm();
})();
</script>
"""
components.html(BOARD_JS, height=0)


def layout(fig: go.Figure, height=420, **kw) -> go.Figure:
    legend = {"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT2}, **kw.pop("legend", {})}
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=TEXT2, size=13),
                      height=height, margin=kw.pop("margin", dict(l=50, r=20, t=40, b=50)),
                      hoverlabel=dict(bgcolor=PANEL, font_color=TEXT, bordercolor=GRID), legend=legend, **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def _mtime(path: str) -> float:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def _load_json(path: str, mtime: float):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def load_json(path: str):
    """Cached, but re-read whenever the file changes (or appears)."""
    return _load_json(path, _mtime(path))


@st.cache_data
def _load_ledger(path: str, mtime: float) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    from src.ledger import Ledger
    return list(Ledger(p).state().values())


def load_ledger(path: str) -> list[dict]:
    return _load_ledger(path, _mtime(path))


def classify(s: dict) -> tuple[str, str]:
    """(stage where it stopped, kill category)."""
    v, at = s.get("verdict"), s.get("killed_at")
    if v in ("INVALID", "DUPLICATE"):
        return "proposed", "untestable"
    if v == "REAL":
        return "REAL", "real"
    if v == "UNDERPOWERED":
        return {"explore": "explore", "validate": "validate", "vault": "vault"}.get(at, "explore"), "untestable"
    if v == "REJECTED":
        return "explore", "never"
    if v == "LUCKY":
        return {"correction": "corrected", "min_effect": "corrected"}.get(at, at), at
    return "explore", "never"


def ledger_frame(states: list[dict]) -> pd.DataFrame:
    rows = []
    for s in states:
        stage, kill = classify(s)
        stg = s["stages"]
        row = {"id": s["id"], "source": s["source"], "verdict": s["verdict"], "killed_at": s.get("killed_at"),
               "reason": s.get("reason"), "statement": s["raw"].get("statement", ""), "stage": stage, "kill": kill,
               "direction": s["raw"].get("direction"), "min_effect": s["raw"].get("min_effect")}
        for name in ("explore", "validate", "vault"):
            r = stg.get(name, {})
            row[f"{name}_effect"] = r.get("effect")
            row[f"{name}_p"] = r.get("p")
            row[f"{name}_n"] = r.get("n_treated")
        row["explore_lo"] = stg.get("explore", {}).get("ci_low")
        row["explore_hi"] = stg.get("explore", {}).get("ci_high")
        row["explore_elo"] = elo(row["explore_effect"])
        row["evidence"] = evidence(row)
        rows.append(row)
    return pd.DataFrame(rows)


ELO_PER_POINT = 1600 / 2.302585  # near equal ratings, d(expected score)/d(Elo) = ln(10)/1600 -> ~695 Elo per point


def elo(x) -> str:
    """Residual effect expressed as an equivalent rating change (exact near equal ratings)."""
    return "—" if x is None or pd.isna(x) else f"{x * ELO_PER_POINT:+.0f} Elo"


def evidence(r) -> str:
    """For non-REAL hypotheses: is this 'no evidence' or 'evidence of no effect'? Uses the explore 95% CI:
    if the whole interval lies inside ±min_effect, any effect that would matter is ruled out (equivalence)."""
    lo, hi, me = r.get("explore_lo"), r.get("explore_hi"), r.get("min_effect")
    if r.get("verdict") == "REAL":
        return "real effect"
    if lo is None or hi is None or me is None or pd.isna(lo) or pd.isna(hi):
        return "—"
    if -me < lo and hi < me:
        return "NO EFFECT (confirmed)"
    return "not shown either way" if r.get("verdict") == "REJECTED" else "looked real, failed later"


FINDING_TITLES = {  # plain-English names for the findings (statements are kept on hover / in the ledger)
    "H0006": "White advantage: players score above their rating as White",
    "H0001": "Tilt: after a loss, re-queueing within 5 minutes hurts",
    "H0019": "Momentum: after a win, re-queueing within 5 minutes helps",
    "H0004": "Losing streaks: after 3+ losses in a row, players underperform",
    "H0039": "Winning streaks (1200–1600): after 2+ wins, players outperform",
    "H0054": "Long sessions (2400+): favourites deep into a session underperform",
    "H0056": "Beginners (<1200): the game after a session-opening loss goes worse",
}


def finding_title(f: dict) -> str:
    for m in [f["lead"], *f["members"]]:
        if m in FINDING_TITLES:
            return FINDING_TITLES[m]
    return f["statement"]


def evidence_text(ev: str, me) -> str:
    return {"NO EFFECT (confirmed)": f"no effect — the whole interval is inside ±{me}, so any effect that would "
                                     "matter is ruled out",
            "not shown either way": "not enough evidence either way"}.get(ev, ev)


def explain(text: str) -> None:
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:+.{nd}f}" if nd == 4 else f"{x:.2g}"


def funnel_figure(df: pd.DataFrame) -> go.Figure:
    """Survivor bars above the axis; one dot per dead hypothesis below, under the stage that killed it."""
    order = {s: i for i, s in enumerate(STAGES)}
    reached = [int(sum(df["stage"].map(order) >= i)) for i in range(len(STAGES))]
    fig = go.Figure()
    fig.add_bar(x=list(range(len(STAGES))), y=reached, marker_color="#4b4a46", marker_line_width=0, width=0.7,
                name="reached this stage", text=reached, textposition="outside", textfont=dict(color=TEXT),
                customdata=STAGES, hovertemplate="%{customdata}: %{y} hypotheses reached this stage<extra></extra>")
    dead = df[df["kill"] != "real"].copy()
    dead["slot"] = dead.groupby("stage").cumcount()
    per_row = 6
    for kill, (label, colour) in KILL.items():
        sub = dead[dead["kill"] == kill] if kill != "real" else df[df["kill"] == "real"].assign(
            slot=lambda d: range(len(d)))
        if sub.empty:
            continue
        xs = [order[s] + ((k % per_row) - (per_row - 1) / 2) * 0.12 for s, k in zip(sub["stage"], sub["slot"])]
        unit = max(reached) / 14  # dot row spacing scales with the tallest bar
        if kill == "real":
            ys = [reached[-1] + unit * (2.2 + (k // per_row)) for k in sub["slot"]]
        else:
            ys = [-unit * (1.2 + (k // per_row)) for k in sub["slot"]]
        hover = [f"<b>{r.id}</b> · {r.verdict}<br>{r.statement[:90]}<br>{label}"
                 f"<br>explore effect {fmt(r.explore_effect)}, p {fmt(r.explore_p, 2)}"
                 f"<br>{(r.reason or '')[:110]}" for r in sub.itertuples()]
        fig.add_scatter(x=xs, y=ys, mode="markers", name=label, text=hover, hovertemplate="%{text}<extra></extra>",
                        marker=dict(size=11, color=colour, line=dict(color=SURFACE, width=2)))
    fig.add_hline(y=0, line_color=TEXT2, line_width=1)
    fig.add_annotation(x=0, y=-max(reached) / 25, xref="x", yref="y", text="graveyard ↓", showarrow=False,
                       font=dict(color=MUTED, size=12), xanchor="left")
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(STAGES))), ticktext=STAGES, range=[-0.6, len(STAGES) - 0.4])
    fig.update_yaxes(visible=False)
    return layout(fig, height=600, barmode="overlay", margin=dict(l=50, r=20, t=90, b=40),
                  legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=TEXT2, size=12)))


# --------------------------------------------------------------------------- header
LEDGER_LABELS = {"ledger_main.jsonl": "Main research run",
                 "ledger_hand.jsonl": "Hand-written hypotheses only",
                 "ledger_naive-null-auto-s0.jsonl": "Naive p-hacker on shuffled data",
                 "ledger_naive-null-llm-s0.jsonl": "Naive Claude on shuffled data",
                 "ledger_naive-real-auto-s0.jsonl": "Naive p-hacker on real data",
                 "ledger_naive-real-llm-s0.jsonl": "Naive Claude on real data"}
ledgers = sorted(RES.glob("ledger_*.jsonl"), key=lambda p: (p.name != "ledger_main.jsonl", p.name))
with st.sidebar:
    st.markdown("### Experiment")
    choice = st.radio("Which ledger to show in Funnel and Ledger", [p.name for p in ledgers], index=0,
                      format_func=lambda n: LEDGER_LABELS.get(n, n)) if ledgers else None
    st.caption("Every experiment keeps its own append-only ledger. The main run is what the demo is about.")
states = load_ledger(str(RES / choice)) if choice else []
df = ledger_frame(states) if states else pd.DataFrame()
cal = load_json(str(RES / "calibration.json"))
findings = load_json(str(RES / "findings.json")) if choice == "ledger_main.jsonl" else None

main_df = ledger_frame(load_ledger(str(RES / "ledger_main.jsonl"))) if (RES / "ledger_main.jsonl").exists() else df
main_findings = load_json(str(RES / "findings.json"))
naive_null = [x for x in (load_json(str(RES / f"naive_null_{r}.json")) for r in ("auto", "llm")) if x]
commit_meta = load_json(str(ROOT / "commitment.json"))

LOGO_SVG = (ROOT / "assets" / "logo.svg").read_text().replace('width="512" height="512"', 'width="44" height="44"')
st.markdown('<div class="topbar">' + LOGO_SVG + '<span class="brand">ChessCheck</span>'
            '<span class="brand-tag">Every chess pattern, checked for luck.</span></div>', unsafe_allow_html=True)


def cnt(n: int) -> str:
    """An integer that counts up from 0 when it appears (pure CSS)."""
    return f'<span class="cnt" style="--to:{int(n)}" aria-label="{int(n)}"></span>'


def kpi(value, label, sub, cls=""):
    return f'<div class="kpi {cls}"><div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div>'


cards = []
if cal:
    cards.append(kpi(f'{cnt(cal["false_real_count"])}<span class="of"> / {cnt(cal["null_hypothesis_tests"])}</span>',
                     "false discoveries on shuffled data",
                     f"Target ≤ 5%. A naive p &lt; 0.05 rule gets {cal['naive_false_rate_per_hypothesis']:.1%}.", "good"))
if not main_df.empty:
    n_real = int((main_df.verdict == "REAL").sum())
    cards.append(kpi(f'{cnt(n_real)}<span class="of"> of {cnt(len(main_df))}</span>', "hypotheses survived as REAL",
                     (f"= {main_findings['n_findings']} distinct findings once duplicates are merged" if main_findings else ""),
                     "blue"))
if naive_null:
    fake = sum(int(round(x["mean_naive_discoveries"] * len(x["runs"]))) for x in naive_null)
    kept = sum(int(round(x["mean_honest_real"] * len(x["runs"]))) for x in naive_null)
    cards.append(kpi(f'{cnt(fake)} → {cnt(kept)}', "naive fakes → kept by the honest pipeline",
                     "p-hacked “discoveries” on shuffled data, all caught", "orange"))
if commit_meta:
    cards.append(kpi(f'{cnt(len(commit_meta["hypotheses"]))}', "predictions locked for the future",
                     f"SHA-256 <span class='mono'>{__import__('hashlib').sha256((ROOT / 'commitment.json').read_bytes()).hexdigest()[:12]}…</span> "
                     f"· tested on games played after {commit_meta['committed_at'][11:16]} UTC", "violet"))

tabs = st.tabs(["Overview", "Funnel", "Ledger", "Naive vs honest", "Calibration", "Live vault", "Ask a question"])

@st.cache_data
def live_games(mtime: float, t0: str) -> pd.DataFrame:
    """Post-commit games from the raw live dump (re-read when the file changes)."""
    f = RES / "live" / "games.jsonl"
    if not f.exists():
        return pd.DataFrame()
    t0_ms = pd.Timestamp(t0).timestamp() * 1000
    rows = []
    for line in f.read_text().splitlines():
        try:
            g = json.loads(line)
        except json.JSONDecodeError:
            continue
        if g.get("createdAt", 0) < t0_ms or g.get("speed") != "blitz" or not g.get("rated"):
            continue
        w, b = g["players"]["white"], g["players"]["black"]
        rows.append({"t": pd.Timestamp(g["createdAt"], unit="ms", tz="UTC"),
                     "white": (w.get("user") or {}).get("name", "?"), "wr": w.get("rating"),
                     "black": (b.get("user") or {}).get("name", "?"), "br": b.get("rating"),
                     "winner": g.get("winner"),
                     "tc": f"{(g.get('clock') or {}).get('initial', 0) // 60}+{(g.get('clock') or {}).get('increment', 0)}"})
    return pd.DataFrame(rows).sort_values("t") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- overview (scroll story)
def fact(inner: str, cls: str = "") -> str:
    return f'<section class="fact reveal {cls}">{inner}</section>'


with tabs[0]:
    parts = []
    parts.append(
        '<section class="fact hero-fact reveal">'
        '<div class="eyebrow">An AI researcher that checks itself</div>'
        '<div class="mega">Finding patterns is easy.<br><span class="grad">Knowing which ones to trust is hard.</span></div>'
        '<p class="lede">Claude proposes ideas about chess players. ChessCheck tests every one on 11 million games, '
        'labels it <b>REAL</b>, <b>LUCKY</b> or <b>REJECTED</b>, and then proves how often it fools itself.</p>'
        '<div class="scroll-cue">scroll<span>↓</span></div></section>')
    parts.append(fact(
        f'<div class="eyebrow">The data</div><div class="giant">{cnt(11)}<span class="unit"> million</span></div>'
        '<p class="lede">rated blitz games from Lichess, October 1–8 2026. Every game seen from both sides: '
        '22 million player-games, each scored against what the Elo ratings predicted.</p>'))
    if cal:
        parts.append(fact(
            '<div class="eyebrow">The problem</div><div class="giant warm">1 in 20</div>'
            '<p class="lede">ideas with <b>no real effect</b> still pass the usual p &lt; 0.05 test by luck. Test enough '
            f'ideas and “discoveries” are guaranteed: a naive researcher finds about <b>{cal["naive_discoveries_per_run_mean"]:.0f}</b> '
            'in pure noise, every single run.</p>'))
    if not main_df.empty:
        order = {s_: i for i, s_ in enumerate(STAGES)}
        reached = [int(sum(main_df["stage"].map(order) >= i)) for i in range(len(STAGES))]
        names = ["Proposed", "Explore", "Corrected", "Validated", "Skeptic", "Vault", "REAL"]
        flow = '<span class="ar">→</span>'.join(
            f'<div class="st{" real" if k == len(names) - 1 else ""}" style="transition-delay:{0.2 + k * 0.12:.2f}s">'
            f'<div class="n">{cnt(n)}</div><div class="k">{nm}</div></div>'
            for k, (nm, n) in enumerate(zip(names, reached)))
        parts.append(fact(
            f'<div class="eyebrow">The method</div><div class="big">{reached[0]} ideas. Six gates. '
            f'<span class="grad">{reached[-1]} survive.</span></div>'
            '<p class="lede">Every idea is written down <b>before</b> it is tested, then must pass a first test, a bar that '
            'rises with every idea tried, a re-test on days it has never seen, automatic sanity checks, and a vault it '
            'can look at exactly once.</p>'
            f'<div class="flow big-flow">{flow}</div>'))
    if cal:
        naive_rate = cal["naive_false_rate_per_hypothesis"]
        parts.append(fact(
            '<div class="eyebrow">Proof 1 · we fed it noise, 50 times</div>'
            f'<div class="giant good">{cnt(cal["false_real_count"])}<span class="unit"> / {cnt(cal["null_hypothesis_tests"])}</span></div>'
            '<p class="lede">false discoveries. We shuffled each player’s results into random order, destroying every '
            'real pattern, and ran every hypothesis through the full pipeline. Target: at most 5%.</p>'
            '<div class="bars">'
            f'<div class="barrow"><span class="bl">Naive p &lt; 0.05</span><span class="bar"><i class="warm" style="--w:{min(naive_rate / 0.06, 1) * 100:.0f}%"></i></span><span class="bv">{naive_rate:.1%}</span></div>'
            '<div class="barrow"><span class="bl">ChessCheck</span><span class="bar"><i class="good" style="--w:0.6%"></i></span><span class="bv">0.0%</span></div>'
            '</div>'))
        pw = {k: v for k, v in cal["power"].items() if not k.startswith("_")}
        if "0.01" in pw:
            parts.append(fact(
                '<div class="eyebrow">Proof 2 · it still finds real things</div>'
                f'<div class="giant blue">{cnt(round(pw["0.01"]["power"] * 100))}<span class="unit">%</span></div>'
                '<p class="lede">of fake effects we planted on purpose, worth about <b>7 rating points</b>, were found and '
                'called REAL, on two different hypotheses. Saying “no” to everything would be easy. This proves it doesn’t.</p>'))
    if naive_null:
        parts.append(fact(
            '<div class="eyebrow">Proof 3 · same AI, safeguards off</div>'
            f'<div class="giant"><span class="warm">{cnt(fake)}</span> <span class="unit">→</span> <span class="good">{cnt(kept)}</span></div>'
            '<p class="lede">With safeguards off, Claude and an automatic p-hacker peeked at results and tweaked their ideas, '
            f'producing {fake} “discoveries” in shuffled data. Claude even wrote confident explanations for patterns that '
            f'were pure noise. ChessCheck kept <b>{kept}</b>.</p>'))
    if main_findings and not main_df.empty:
        vault_eff = main_df.set_index("id")["vault_effect"]
        fs = sorted(main_findings["findings"], key=lambda f: -abs(vault_eff.get(f["lead"], 0) or 0))
        cards_html = "".join(
            f'<div class="fcard {"pos" if (vault_eff.get(f["lead"]) or 0) > 0 else "neg"}" style="transition-delay:{0.15 + i * 0.07:.2f}s">'
            f'<div class="fnum">{elo(vault_eff.get(f["lead"]))}</div><div class="ftxt">{finding_title(f)}</div></div>'
            for i, f in enumerate(fs))
        parts.append(fact(
            f'<div class="eyebrow">What survived</div><div class="big">{main_findings["n_real"]} REAL hypotheses. '
            f'<span class="grad">{main_findings["n_findings"]} distinct findings.</span></div>'
            '<p class="lede">Mirror claims (“White does better” / “Black does worse”) and narrower variants are merged, so '
            'nothing is counted twice. Effects in rating points, measured in the one-shot vault:</p>'
            f'<div class="fgrid">{cards_html}</div>'))
    parts.append(fact(
        '<div class="eyebrow">The bug that proved the method</div><div class="giant warm">p = 10<sup>−213</sup></div>'
        '<p class="lede">Our own first test found an after-a-loss effect in <b>coin flips</b>. Each game was compared with '
        'the player’s average, and that average included the very loss that picked the game: the hot-hand selection bias '
        '(Miller &amp; Sanjurjo, 2018). We fixed the test, and made the noise harder so the self-check would have caught it.</p>'))
    if commit_meta:
        h = __import__("hashlib").sha256((ROOT / "commitment.json").read_bytes()).hexdigest()
        n_live = len(live_games(_mtime(str(RES / "live" / "games.jsonl")), commit_meta["committed_at"]))
        parts.append(fact(
            '<div class="eyebrow">The final exam · games that hadn’t been played yet</div>'
            f'<div class="giant violet">{cnt(len(commit_meta["hypotheses"]))}<span class="unit"> predictions sealed</span></div>'
            f'<p class="lede">at {commit_meta["committed_at"][11:16]} UTC, with a fingerprint published before the games '
            f'existed. Since then: <b>{n_live:,}</b> new games, checked live, with a one-shot verdict fixed in advance.</p>'
            f'<div class="hash">{h}</div>'))
    lims = [("Results, not play quality", "we score the result against the ratings, not the moves."),
            ("One week, one site", "Lichess rated blitz, October 1–8 2026."),
            ("Associations, not causes", "“tilt” could be the kind of player who re-queues instantly."),
            ("Local time is estimated", "from each player’s quietest 6 hours.")]
    parts.append(fact(
        '<div class="eyebrow">What we don’t claim</div><div class="big">Limits, stated up front.</div>'
        '<div class="limits">' + "".join(f'<div class="lim" style="transition-delay:{0.15 + i * 0.08:.2f}s"><b>{t}</b><span>{d}</span></div>'
                                         for i, (t, d) in enumerate(lims)) + '</div>'))
    parts.append(fact(
        '<div class="big">See the evidence for yourself.</div>'
        '<div class="limits">'
        '<div class="lim"><b>Funnel</b><span>where every idea died, and why</span></div>'
        '<div class="lim"><b>Calibration</b><span>the noise test and power curves</span></div>'
        '<div class="lim"><b>Live vault</b><span>predictions being checked right now</span></div>'
        '<div class="lim"><b>Ask a question</b><span>test your own chess hypothesis</span></div></div>', "outro"))
    st.markdown('<div class="story">' + "".join(parts) + "</div>", unsafe_allow_html=True)
    with st.expander("Methods and sources"):
        st.markdown('<div class="src">'
                    "Benjamini &amp; Hochberg (1995): false discovery rate control · "
                    "Miller &amp; Sanjurjo (2018), <i>Econometrica</i>: the hot-hand selection bias we found and fixed · "
                    "Cameron, Gelbach &amp; Miller (2011): two-way cluster-robust standard errors · "
                    "Howard, Ramdas, McAuliffe &amp; Sekhon (2021), <i>Annals of Statistics</i>: confidence sequences · "
                    "Waudby-Smith et al. (2021): asymptotic confidence sequences · "
                    "Elo (1978): expected score · "
                    '<a href="https://database.lichess.org/">Lichess open database</a> (CC0) · '
                    '<a href="https://lichess.org/api">Lichess API</a>. Every design decision is in NOTES.md with a one-line why.'
                    "</div>", unsafe_allow_html=True)

tabs = tabs[1:]

# --------------------------------------------------------------------------- funnel
with tabs[0]:
    if df.empty:
        st.info("No ledger yet. Run `python -m src.judge` or `python -m src.proposer`.")
    else:
        explain("Every idea starts on the left and must pass each stage to move right: a first test "
                "(<b>explore</b>), a stricter bar because we test many ideas (<b>corrected</b>), a re-test on later "
                "days it has never seen (<b>validate</b>), automatic sanity checks (<b>skeptic</b>) and one final "
                "look at held-back data (<b>vault</b>). Ideas that die drop into the graveyard below the line, "
                "coloured by what killed them. Hover any dot for details.")
        st.plotly_chart(funnel_figure(df), width="stretch")
        st.caption("Bars: how many hypotheses reached each stage. Dots below the line: the graveyard — each "
                   "hypothesis sits under the stage that killed it. Grey never looked real; coloured ones looked "
                   "significant and were then exposed as luck. Green dots above: REAL.")
        if findings:
            st.markdown(f"#### {findings['n_real']} REAL hypotheses → {findings['n_findings']} distinct findings")
            explain("Several REAL hypotheses are the same fact said twice (\"White outperforms\" and \"Black "
                    "underperforms\") or narrower versions of each other. Counting them separately would oversell "
                    "the result, so they are merged when their treated games overlap far more than chance would give.")
            vault_eff = df.set_index("id")["vault_effect"]
            st.dataframe(pd.DataFrame([{
                "finding": finding_title(f), "hypotheses": ", ".join(f["members"]),
                "vault effect (lead)": vault_eff.get(f["lead"]), "≈ rating points": elo(vault_eff.get(f["lead"]))}
                for f in findings["findings"]]), hide_index=True, width="stretch",
                column_config={"vault effect (lead)": st.column_config.NumberColumn(format="%.4f")})
        wc = df[df["validate_effect"].notna()].copy()
        if len(wc) >= 3:
            st.markdown("#### Do effects hold up on fresh data?")
            explain("Each dot is a hypothesis that survived the first test and the correction. Across: its effect on "
                    "explore, the data it was chosen on. Up: the same pre-registered effect re-measured on data it had "
                    "never seen. On the dashed line means it replicated at full strength; below the line means it shrank, "
                    "the signature of a lucky pick (the <i>winner's curse</i>). Survivors of the correction mostly sit "
                    "on the line. The flukes were already filtered out by the stricter bar.")
            sign = wc["explore_effect"].apply(lambda e: 1 if e >= 0 else -1)
            fig = go.Figure()
            for name, col, colour in (("validate", "validate_effect", "#3987e5"), ("vault", "vault_effect", "#d55181")):
                d = wc[wc[col].notna()]
                sg = sign[d.index]
                fig.add_scatter(x=d["explore_effect"] * sg, y=d[col] * sg, mode="markers", name=f"re-measured on {name}",
                                marker=dict(size=10, color=colour, line=dict(color=SURFACE, width=2)),
                                text=[f"<b>{r.id}</b> · {r.verdict}<br>{r.statement[:80]}" for r in d.itertuples()],
                                hovertemplate="%{text}<br>explore %{x:.4f} → %{y:.4f}<extra></extra>")
            hi = float(max((wc["explore_effect"].abs()).max(), 0.01)) * 1.1
            fig.add_scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=MUTED, dash="dash", width=1),
                            name="no shrinkage", hoverinfo="skip")
            ratio = ((wc["validate_effect"] * sign).sum() / (wc["explore_effect"] * sign).sum())
            st.plotly_chart(layout(fig, height=420, xaxis_title="effect on explore (in the predicted direction)",
                                   yaxis_title="effect on fresh data", legend=dict(orientation="h", y=-0.2)),
                            width="stretch")
            st.caption(f"Effects re-measured on validate keep {ratio:.0%} of their explore size on average. Dots "
                       "near zero are the ones that died.")

# --------------------------------------------------------------------------- ledger
with tabs[1]:
    if df.empty:
        st.info("No ledger yet.")
    else:
        explain("Every hypothesis ever proposed, written down <b>before</b> it was tested. Effects are in score "
                "points versus the player's own usual level (0.01 ≈ 7 rating points). <b>evidence</b> separates "
                "\"we couldn't tell\" from \"we checked, and any effect that would matter is ruled out\". "
                "Click a row for the full spec, the per-rating-band breakdown and the reasoning behind it.")
        verdicts = st.multiselect("Verdict", sorted(df.verdict.dropna().unique()), default=sorted(df.verdict.dropna().unique()))
        VBADGE = {"REAL": "🟢 REAL", "LUCKY": "🟣 LUCKY", "REJECTED": "⚪ REJECTED", "UNDERPOWERED": "🟠 UNDERPOWERED",
                  "INVALID": "⚫ INVALID", "DUPLICATE": "⚫ DUPLICATE"}
        view = df[df.verdict.isin(verdicts)].assign(verdict_badge=lambda d: d["verdict"].map(VBADGE).fillna(d["verdict"]),
                                                    killed_at=lambda d: d["killed_at"].fillna(""))
        view = view[["id", "verdict_badge", "killed_at", "evidence", "statement", "explore_elo", "explore_effect", "explore_p", "explore_n", "validate_effect",
                                             "validate_p", "vault_effect", "vault_p", "source", "direction", "reason"]]
        labels = {"id": "ID", "verdict_badge": "Verdict", "killed_at": "Died at", "evidence": "Evidence",
                  "statement": st.column_config.TextColumn("Hypothesis", width="large"),
                  "explore_elo": "≈ Elo", "explore_effect": st.column_config.NumberColumn("Explore effect", format="%.4f"),
                  "validate_effect": st.column_config.NumberColumn("Validate effect", format="%.4f"),
                  "vault_effect": st.column_config.NumberColumn("Vault effect", format="%.4f"),
                  "explore_p": st.column_config.NumberColumn("Explore p", format="%.1e"),
                  "explore_n": st.column_config.NumberColumn("Explore n", format="%d"),
                  "validate_p": st.column_config.NumberColumn("Validate p", format="%.1e"),
                  "vault_p": st.column_config.NumberColumn("Vault p", format="%.1e"),
                  "source": "Source", "direction": "Direction", "reason": "Why"}
        ev = st.dataframe(view, width="stretch", height=560, hide_index=True, on_select="rerun",
                          selection_mode="single-row", column_config=labels)
        sel = ev.selection.rows if ev and ev.selection else []
        if sel:
            hid = view.iloc[sel[0]]["id"]
            s = next(x for x in states if x["id"] == hid)
            st.markdown(f"#### {hid} — {s['verdict']}  \n{s['raw'].get('statement')}")
            st.markdown(f"**Rationale (LLM / author):** {s['raw'].get('rationale', '—')}  \n**Reason:** {s.get('reason')}")
            sk = s["stages"].get("skeptic")
            if sk and sk.get("bands"):
                b = pd.DataFrame(sk["bands"]).T.reset_index(names="rating_band")
                b[["lo", "hi"]] = pd.DataFrame(b["ci"].tolist(), index=b.index)
                fig = go.Figure(go.Scatter(x=b["effect"], y=b["rating_band"], mode="markers",
                                           marker=dict(size=10, color="#3987e5"),
                                           error_x=dict(type="data", symmetric=False, array=b["hi"] - b["effect"],
                                                        arrayminus=b["effect"] - b["lo"], color=TEXT2),
                                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
                fig.add_vline(x=0, line_color=MUTED)
                st.markdown("**Per rating band (explore+validate), 95% CI**")
                st.plotly_chart(layout(fig, height=260, xaxis_title="player-adjusted effect on residual"),
                                width="stretch")
                st.write({"skeptic checks": sk["checks"]})
            with st.expander("Full spec and stage results (JSON)"):
                st.json(s)

# --------------------------------------------------------------------------- naive vs honest
with tabs[2]:
    explain("<b>Naive</b> is a sloppy researcher. Claude tests ideas, looks at the results, tweaks the ideas up to 3 "
            "times, and calls anything with p &lt; 0.05 a discovery: no correction, no fresh data, no vault. Each "
            "naive \"discovery\" is then sent through the <b>honest</b> pipeline. On <b>shuffled</b> data, where "
            "every real pattern has been destroyed, every naive discovery is fake by construction.")
    found = False
    FATE_ORDER = [("correction", "killed by the correction", "#3987e5"), ("min_effect", "effect too small", "#d95926"),
                  ("validate", "failed on fresh data", "#199e70"), ("skeptic", "failed skeptic checks", "#c98500"),
                  ("vault", "failed in the vault", "#d55181"), ("REAL", "REAL", GOOD)]
    for data in ("null", "real"):
        st.markdown(f"### {'On shuffled data: every discovery is fake' if data == 'null' else 'On real data'}")
        if data == "real":
            st.caption("On real data many naive discoveries are genuine but tiny. With millions of games almost "
                       "everything is “significant”, which is why every hypothesis must clear a minimum effect.")
        for refiner in ("llm", "auto"):
            nv = load_json(str(RES / f"naive_{data}_{refiner}.json"))
            if not nv:
                continue
            found = True
            who = "Claude as a naive researcher" if refiner == "llm" else "Automatic p-hacker"
            seeds = f" · {len(nv['runs'])} shuffles" if len(nv["runs"]) > 1 else ""
            fate_all = pd.DataFrame([x for r in nv["runs"] for x in r["fate"]])
            n_naive = len(fate_all)
            n_real = int((fate_all["verdict"] == "REAL").sum()) if n_naive else 0
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f'<div class="card"><h5>{who}{seeds}</h5>'
                            f'<div style="display:flex;align-items:baseline;gap:14px;margin-top:8px">'
                            f'<div><div class="num" style="font-size:2.4rem">{n_naive}</div><p>naive “discoveries”</p></div>'
                            f'<div style="font-size:1.6rem;color:{MUTED}">→</div>'
                            f'<div><div class="num" style="font-size:2.4rem;color:{"#3fbf3f" if data == "null" or n_real else TEXT}">'
                            f'{n_real}</div><p>REAL after honest testing</p></div></div></div>', unsafe_allow_html=True)
            with c2:
                if n_naive:
                    key = fate_all.apply(lambda r: "REAL" if r["verdict"] == "REAL" else (r["killed_at"] or "correction"), axis=1)
                    fig = go.Figure()
                    for k, label, colour in FATE_ORDER:
                        n = int((key == k).sum())
                        if n:
                            fig.add_bar(y=[""], x=[n], orientation="h", name=f"{label} ({n})", marker_color=colour,
                                        marker_line=dict(color=SURFACE, width=2),
                                        showlegend=True, hovertemplate=f"{label}: {n}<extra></extra>")
                    st.plotly_chart(layout(fig, height=150, barmode="stack", margin=dict(l=10, r=10, t=10, b=10),
                                           xaxis=dict(visible=False), yaxis=dict(visible=False),
                                           legend=dict(orientation="h", y=-0.15, x=0)),
                                    width="stretch", key=f"fate_{data}_{refiner}")
            fate = pd.DataFrame(nv["runs"][0]["fate"])
            if not fate.empty:
                with st.expander(f"See every naive discovery ({len(fate)}{' in the first shuffle' if seeds else ''}) "
                                 "and what happened to it"):
                    fate["honest fate"] = fate["verdict"] + fate["killed_at"].fillna("").map(lambda k: f" @ {k}" if k else "")
                    st.dataframe(fate[["statement", "naive_p", "naive_effect", "round", "honest fate", "reason"]],
                                 hide_index=True, width="stretch",
                                 column_config={"statement": st.column_config.TextColumn("Naive discovery", width="large"),
                                                "naive_p": st.column_config.NumberColumn("Naive p", format="%.1e"),
                                                "naive_effect": st.column_config.NumberColumn("Effect", format="%.4f"),
                                                "round": st.column_config.NumberColumn("Refinement round"),
                                                "honest fate": "Honest verdict", "reason": "Why"})
    if not found:
        st.info("Run `python -m src.naive` to populate.")

# --------------------------------------------------------------------------- calibration
with tabs[3]:
    if not cal:
        st.info("Run `python -m src.calibrate`.")
    else:
        explain("How do we know the pipeline works? We shuffled each player's results into random order, which "
                "destroys every real pattern, and ran all hypotheses through the pipeline 50 times. Any REAL verdict "
                "there is a false alarm. Then we <b>planted</b> a fake effect of known size and checked the pipeline "
                "finds it (<b>power</b>). Bottom chart: on shuffled data a correct test gives p-values spread evenly "
                "from 0 to 1. A spike near 0 would mean a biased test.")
        st.markdown('<div class="kpis">' + "".join([
            kpi(f'{cal["false_real_rate_per_hypothesis"]:.2%}', "false REAL rate per hypothesis",
                f'{cal["null_hypothesis_tests"]:,} tests on shuffled data · target ≤ 5%', "good"),
            kpi(f'{cal["fraction_runs_with_any_real"]:.0%}', "of shuffled runs with any REAL",
                f'{cal["null_seeds"]} runs of {cal["bank_size"]} hypotheses each (family-wise)', "good"),
            kpi(f'{cal["naive_discoveries_per_run_mean"]:.1f}', "naive discoveries per run",
                "p &lt; 0.05 with no correction, on the same shuffled data"),
            kpi(f'{cal["power"].get("0.01", {}).get("power", 0):.0%}', "power at 0.01 (≈ 7 Elo)",
                "share of planted effects found and called REAL")]) + "</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            rates = {"Naive (p<0.05)": cal["naive_false_rate_per_hypothesis"],
                     "Honest pipeline": cal["false_real_rate_per_hypothesis"]}
            fig = go.Figure(go.Bar(x=list(rates), y=list(rates.values()), marker_color=["#d95926", "#3987e5"],
                                   width=0.5, text=[f"{v:.2%}" for v in rates.values()], textposition="outside",
                                   textfont=dict(color=TEXT), hovertemplate="%{x}: %{y:.3%}<extra></extra>"))
            fig.add_hline(y=0.05, line_dash="dash", line_color=TEXT2, annotation_text="5% target",
                          annotation_font_color=TEXT2)
            st.markdown("**How often does it fool itself?** Share of hypotheses called real on shuffled data")
            st.plotly_chart(layout(fig, height=360, yaxis_tickformat=".0%", yaxis_title="share of hypotheses called real"),
                            width="stretch")
        with right:
            pw = {k: v for k, v in cal["power"].items() if not k.startswith("_")}
            xs = [float(k) for k in pw]
            ys = [v["power"] for v in pw.values()]
            fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#3987e5", width=2),
                                       marker=dict(size=9), name="planted on tilt",
                                       hovertemplate="planted effect %{x}: power %{y:.0%}<extra></extra>"))
            extra = load_json(str(RES / "power_14.json"))
            if extra:
                ex_pw = extra["power"]
                fig.add_scatter(x=[float(k) for k in ex_pw], y=[v["power"] for v in ex_pw.values()],
                                mode="lines+markers", line=dict(color="#d95926", width=2), marker=dict(size=9),
                                name=f"planted on 'after losing on time' ({extra['explore_treated_rows'] / 1e6:.1f}M explore games)",
                                hovertemplate="planted effect %{x}: power %{y:.0%}<extra></extra>")
            st.markdown("**Power: does it find an effect we planted?**")
            st.plotly_chart(layout(fig, height=360, xaxis_title="planted effect on residual (score points)",
                                   yaxis_title="detected as REAL", yaxis_tickformat=".0%", yaxis_range=[0, 1.05],
                                   legend=dict(orientation="h", y=-0.35, x=0),
                                   xaxis_type="log", xaxis_tickvals=xs,
                                   xaxis_ticktext=[f"{x:g}<br>(≈{x * ELO_PER_POINT:.0f} Elo)" for x in xs]),
                            width="stretch")
        hist = cal["null_explore_p_hist"]
        fig = go.Figure(go.Bar(x=[i / 20 + 0.025 for i in range(20)], y=hist, marker_color=MUTED, width=0.045,
                               hovertemplate="p in [%{x:.3f}±0.025]: %{y}<extra></extra>"))
        st.markdown("**Is the test itself fair?** p-values on shuffled data should be spread evenly (flat bars)")
        st.plotly_chart(layout(fig, height=240, xaxis_title="p-value"), width="stretch")
        with st.expander("Raw calibration numbers"):
            st.json({k: v for k, v in cal.items() if k in ("null_fate_counts", "power", "runtime_s")})

# --------------------------------------------------------------------------- live vault
def read_fresh(path: Path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:  # file being rewritten by the live runner
        return None


@st.cache_data
def trajectory(mtime: float, hid: str) -> pd.DataFrame:
    """Running estimate + anytime-valid interval for one committed hypothesis, recomputed at 10-minute
    checkpoints on the games that existed at that time (exactly what a viewer would have seen then)."""
    from src import live as L
    commit = json.loads((ROOT / "commitment.json").read_text())
    one = {**commit, "hypotheses": [h for h in commit["hypotheses"] if h["id"] == hid]}
    one.pop("final_test_at", None)
    games = L.load_games()
    t0 = pd.Timestamp(commit["committed_at"])
    last = max((g["createdAt"] for g in games.values()), default=0)
    out = []
    for t in pd.date_range(t0 + pd.Timedelta(minutes=10), pd.Timestamp(last, unit="ms", tz="UTC"), freq="10min"):
        cut = t.timestamp() * 1000
        r = L.evaluate(one, {k: g for k, g in games.items() if g["createdAt"] <= cut})["results"][0]
        if r["lo"] is not None:
            out.append({"t": t, "effect": r["effect"], "lo": r["lo"], "hi": r["hi"], "n": r["n_treated"]})
    return pd.DataFrame(out)


@st.fragment(run_every=30)
def live_tab():
    commit = read_fresh(ROOT / "commitment.json")
    live = read_fresh(RES / "live" / "live_results.json")
    explain("The vault inside October is still old data. So we locked the surviving hypotheses into "
            "<code>commitment.json</code> and published its SHA-256 fingerprint <b>before</b> the games below were "
            "played. Change one character of the file and the fingerprint changes, so anyone can check nothing was "
            "edited afterwards (<code>python -m src.live verify</code>). The intervals are <b>anytime-valid</b>, so "
            "watching them update does not create false positives. The one-shot <b>final test</b> was fixed in the "
            "commitment in advance. This page refreshes every 30 s.")
    if not commit:
        st.info("No commitment yet. Run `python -m src.live commit`.")
        return
    import hashlib
    h = hashlib.sha256((ROOT / "commitment.json").read_bytes()).hexdigest()
    published = (ROOT / "commitment.sha256").read_text().strip() if (ROOT / "commitment.sha256").exists() else None
    if live and pd.Timestamp(live.get("updated_at", "1970-01-01T00:00:00+00:00")) < pd.Timestamp(commit["committed_at"]):
        live = None  # results from an earlier commitment
    now = pd.Timestamp.now(tz="UTC")
    t_final = pd.Timestamp(commit["final_test_at"]) if commit.get("final_test_at") else None
    ok = published == h
    mins = int((t_final - now).total_seconds() // 60) if t_final is not None else None
    cards = [kpi("✓ verified" if ok else "✗ mismatch", "commitment unchanged",
                 f"SHA-256 <span class='mono'>{h[:20]}…</span>", "good" if ok else ""),
             kpi(commit["committed_at"][11:16] + " UTC", "predictions locked",
                 f"{commit['committed_at'][:10]} · {len(commit['hypotheses'])} hypotheses · {len(commit.get('player_pool', []))} players watched"),
             kpi(f"{len(live_games(_mtime(str(RES / 'live' / 'games.jsonl')), commit['committed_at'])):,}",
                 "games played since",
                 (f"{live['n_games']:,} usable for testing (rated blitz, history known) · updated "
                  f"{live['updated_at'][11:16]} UTC") if live else "waiting for the runner"),
             kpi((f"in {mins // 60}h {mins % 60:02d}m" if mins is not None and mins > 0 else "done") if t_final is not None else "—",
                 "one-shot final test", f"pre-registered for {commit.get('final_test_at', '')[11:16]} UTC")]
    st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)
    gfile = RES / "live" / "games.jsonl"
    gm = _mtime(str(gfile))
    lg = live_games(gm, commit["committed_at"])
    fresh = live and (now - pd.Timestamp(live["updated_at"])).total_seconds() < 25 * 60
    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;margin:4px 0 10px 0">'
                f'<span class="live-badge">{"<span class=live-dot></span>LIVE" if fresh else "● PAUSED"}</span>'
                f'<span style="color:{MUTED};font-size:0.85rem">runner polls 300 players in rounds of ~10 min · '
                f'page refreshes every 30 s</span></div>', unsafe_allow_html=True)
    if not lg.empty:
        left, right = st.columns([3, 2])
        with left:
            cum = lg.set_index("t").resample("5min").size().cumsum().reset_index(name="games")
            fig = go.Figure(go.Scatter(x=cum["t"], y=cum["games"], mode="lines", line=dict(color="#3987e5", width=2),
                                       fill="tozeroy", fillcolor="rgba(57,135,229,0.15)",
                                       hovertemplate="%{x|%H:%M} UTC: %{y:,} games<extra></extra>"))
            if t_final is not None:
                fig.add_vline(x=t_final, line_dash="dash", line_color=TEXT2)
                fig.add_annotation(x=t_final, y=1, yref="paper", text="final test", showarrow=False,
                                   font=dict(color=TEXT2), xanchor="left", yanchor="top")
            st.markdown("**Games played since the predictions were locked**")
            st.plotly_chart(layout(fig, height=300, margin=dict(l=50, r=20, t=10, b=40),
                                   xaxis_range=[pd.Timestamp(commit["committed_at"]),
                                                max(lg["t"].max(), t_final if t_final is not None else lg["t"].max())
                                                + pd.Timedelta(minutes=10)]), width="stretch")
        with right:
            st.markdown("**Latest games**")
            rowsh = []
            for i, g in enumerate(lg.tail(8).iloc[::-1].itertuples()):
                res = "1–0" if g.winner == "white" else "0–1" if g.winner == "black" else "½–½"
                wcls = "win" if g.winner == "white" else ""
                bcls = "win" if g.winner == "black" else ""
                rowsh.append(f'<div class="tick-row" style="animation-delay:{i * 0.06:.2f}s"><span class="t">{g.t:%H:%M}</span>'
                             f'<span class="w {wcls}">{g.white} <span style="color:{MUTED}">{g.wr}</span></span>'
                             f'<span class="res">{res}</span>'
                             f'<span class="{bcls}">{g.black} <span style="color:{MUTED}">{g.br}</span></span>'
                             f'<span class="tc">{g.tc}</span></div>')
            st.markdown(f'<div class="ticker">{"".join(rowsh)}</div>', unsafe_allow_html=True)
        tr = trajectory(gm, "H0006")
        if len(tr) >= 2:
            pred = next(h["validate"]["effect"] for h in commit["hypotheses"] if h["id"] == "H0006")
            fig = go.Figure()
            fig.add_scatter(x=list(tr["t"]) + list(tr["t"][::-1]), y=list(tr["hi"]) + list(tr["lo"][::-1]),
                            fill="toself", fillcolor="rgba(57,135,229,0.18)", line=dict(width=0),
                            name="95% anytime-valid interval", hoverinfo="skip")
            fig.add_scatter(x=tr["t"], y=tr["effect"], mode="lines+markers", line=dict(color="#3987e5", width=2),
                            marker=dict(size=6), name="live estimate", customdata=tr["n"],
                            hovertemplate="%{x|%H:%M}: %{y:+.3f} (%{customdata} White games)<extra></extra>")
            fig.add_hline(y=pred, line_dash="dash", line_color="#d95926",
                          annotation_text=f"predicted {pred:+.3f} ({elo(pred)})", annotation_font_color="#d95926")
            fig.add_hline(y=0, line_color=MUTED)
            st.markdown("**Watching one prediction converge: White advantage**")
            st.caption("The shaded band is where the true effect can still be. It narrows as games arrive. When it "
                       "no longer covers zero, the prediction is confirmed live. Peeking every 30 s is safe "
                       "because the band is built for continuous monitoring.")
            st.plotly_chart(layout(fig, height=320, margin=dict(l=50, r=20, t=10, b=40),
                                   yaxis_range=[-0.5, 0.5], legend=dict(orientation="h", y=1.02, yanchor="bottom")),
                            width="stretch")
    with st.expander("Full SHA-256 and how to verify"):
        st.code(f"{h}\n\n$ python -m src.live verify\nVERIFIED: matches the published hash; commitment unchanged.", language=None)
    ft = (live or {}).get("final_test") or {}
    if ft.get("status") == "done":
        n_ok = sum(r["verdict"] == "CONFIRMED" for r in ft["results"])
        st.markdown(f"#### Final test: {n_ok} of {len(ft['results'])} predictions confirmed")
        st.caption(f"Games started between the commit and {ft['at'][11:16]} UTC ({ft['n_rows']:,} player-games). "
                   "One-sided test in the committed direction, Benjamini–Hochberg q = 0.05 across all committed hypotheses.")
        st.dataframe(pd.DataFrame(ft["results"]), hide_index=True, width="stretch")
    if not live:
        st.info("No post-commit games yet. Start `python -m src.live run`.")
        return
    rows = pd.DataFrame(live["results"])
    rows["title"] = rows["id"].map(FINDING_TITLES).fillna(rows["statement"])
    st.markdown("#### Running estimates")
    st.caption("Each prediction is compared with the games coming in. Intervals are 95% anytime-valid confidence "
               "sequences, wide at first and narrowing as games arrive. “Confirmed” means the interval has moved "
               "entirely to the predicted side of zero.")
    STATUS = {"confirmed": "🟢 confirmed", "contradicted": "🔴 contradicted", "inconclusive": "⚪ inconclusive",
              "waiting for data": "⏳ waiting for data"}
    table = pd.DataFrame({
        "ID": rows["id"], "Prediction": rows["title"],
        "Predicted": rows["predicted_effect"].map(elo),
        "Live estimate": rows["effect"].map(elo),
        "Status": rows["status"].map(STATUS).fillna(rows["status"]),
        "Games matching": rows["n_treated"]})
    st.dataframe(table, hide_index=True, width="stretch", height=min(40 + 35 * len(table), 620),
                 column_config={"Prediction": st.column_config.TextColumn(width="large"),
                                "Games matching": st.column_config.ProgressColumn(
                                    "Games matching (of 500 target)", min_value=0, max_value=500, format="%d")})
    shown = rows[rows["lo"].notna()]
    if len(shown):
        fig = go.Figure()
        for i, r in enumerate(shown.itertuples()):
            fig.add_scatter(x=[r.effect], y=[r.id], mode="markers", marker=dict(size=10, color="#3987e5"),
                            name="live estimate", showlegend=i == 0,
                            error_x=dict(type="data", symmetric=False, array=[r.hi - r.effect],
                                         arrayminus=[r.effect - r.lo], color=TEXT2),
                            hovertemplate=f"{r.id}: {str(r.title)[:80]}<br>live %{{x:.4f}} [{r.lo:.3f}, {r.hi:.3f}]<extra></extra>")
            fig.add_scatter(x=[r.predicted_effect], y=[r.id], mode="markers", showlegend=i == 0,
                            name="committed estimate", marker=dict(symbol="diamond-open", size=12, color="#d95926"),
                            hovertemplate="committed %{x:.4f}<extra></extra>")
        fig.add_vline(x=0, line_color=MUTED)
        st.plotly_chart(layout(fig, height=120 + 50 * len(shown), xaxis_title="effect on score (player-adjusted)",
                               xaxis_range=[-0.3, 0.3], legend=dict(orientation="h", y=1.02, yanchor="bottom")),
                        width="stretch")
        st.caption("Only predictions with enough matching games are drawn. The axis is clipped at ±0.3; early "
                   "intervals can be wider.")


with tabs[4]:
    live_tab()

# --------------------------------------------------------------------------- ask
def describe(clauses) -> str:
    if not clauses:
        return "everyone"
    ops = {"==": "is", "!=": "is not", "<": "<", "<=": "≤", ">": ">", ">=": "≥", "in": "is one of", "not_in": "is not one of"}
    out = []
    for c in clauses:
        v = c["value"]
        if c["op"] == "between":
            out.append(f"<b>{c['feature']}</b> between {v[0]} and {v[1]}")
        else:
            out.append(f"<b>{c['feature']}</b> {ops.get(c['op'], c['op'])} {', '.join(map(str, v)) if isinstance(v, list) else v}")
    return " and ".join(out)


def stage_path(res: dict) -> str:
    names = ["Explore", "Correction", "Validate", "Skeptic", "Vault"]
    idx = {"explore": 0, "correction": 1, "min_effect": 1, "validate": 2, "skeptic": 3, "vault": 4}
    dead = None if res["verdict"] == "REAL" else idx.get(res.get("killed_at") or "explore", 0)
    chips = []
    for i, n in enumerate(names):
        if dead is None or i < dead:
            chips.append(f'<div class="st real"><div class="n" style="color:#3fbf3f">✓</div><div class="k">{n}</div></div>')
        elif i == dead:
            chips.append(f'<div class="st" style="border-color:#d55181"><div class="n" style="color:#e57aa0">✗</div><div class="k">{n}</div></div>')
        else:
            chips.append(f'<div class="st" style="opacity:0.35"><div class="n">·</div><div class="k">{n}</div></div>')
    return '<div class="flow">' + '<span class="ar">→</span>'.join(chips) + "</div>"


with tabs[5]:
    explain("Type a hypothesis in plain English. Claude turns it into a precise, testable spec, you check it, and "
            "the full honest pipeline runs (about 30 s). The question is logged <b>before</b> it is tested and counts "
            "toward the correction, so asking many questions makes the bar stricter. <b>Test mode</b> (on by default) "
            "uses a copy of the main ledger, so demo questions can't change the committed results. Time-of-day "
            "questions can use each player's <b>estimated local time</b>.")
    examples = ["Do players perform worse after midnight their local time?",
                "Do players do worse in their 20th game of a session?",
                "Is the Sicilian (ECO B) better for 1600-2000 players?",
                "Do players on a 2-game losing streak do worse in 3+0 games?"]
    st.markdown("**Try one:**")
    ecols = st.columns(2)
    for i, ex_q in enumerate(examples):
        if ecols[i % 2].button(ex_q, key=f"ex{i}", width="stretch"):
            st.session_state["ask_q"] = ex_q
            st.session_state.pop("ask_spec", None)
            st.session_state.pop("ask_res", None)
    q = st.text_input("Your hypothesis", key="ask_q",
                      placeholder="e.g. Players do worse in their third game after midnight")
    c1, c2 = st.columns([1, 3])
    sandbox = c2.toggle("Test mode (sandbox ledger)", value=True)
    if c1.button("Convert to a testable spec", type="primary", disabled=not q):
        with st.spinner("Claude is translating your question…"):
            try:
                from src.ask import translate
                st.session_state["ask_spec"] = translate(q)
                st.session_state.pop("ask_res", None)
            except Exception as e:  # no API key or network
                st.error(f"Could not reach the LLM: {e}")
    spec = st.session_state.get("ask_spec")
    if spec:
        comp = spec.get("comparison")
        control = "the rest of that group" if comp in (None, "rest_of_subset") else describe(comp.get("condition", []))
        arrow = {"less": "worse than", "greater": "better than", "two-sided": "differently from"}.get(spec.get("direction"), "vs")
        st.markdown(f'<div class="card" style="margin:10px 0"><h5>{spec.get("statement")}</h5><p>'
                    f'Population: {describe(spec.get("subset", []))}<br>'
                    f'Treated games: {describe(spec.get("condition", []))}<br>'
                    f'Compared with: {control}<br>'
                    f'Prediction: treated games go <b>{arrow}</b> the comparison, by at least '
                    f'<b>{spec.get("min_effect")}</b> score points (≈ {spec.get("min_effect", 0) * ELO_PER_POINT:.0f} Elo)'
                    f'</p></div>', unsafe_allow_html=True)
        with st.expander("Exact JSON spec"):
            st.json(spec)
        if st.button("Looks right: run the pipeline", type="primary"):
            from src.ask import run_question
            with st.spinner("Running explore → correction → validate → skeptic → vault…"):
                st.session_state["ask_res"] = run_question(spec, sandbox=sandbox)
            st.cache_data.clear()
    res = st.session_state.get("ask_res")
    if res and spec:
        pill = {"REAL": "real", "LUCKY": "lucky"}.get(res["verdict"], "rej")
        st.markdown(f'#### Verdict: <span class="pill {pill}" style="font-size:1rem">{res["verdict"]}</span>',
                    unsafe_allow_html=True)
        st.markdown(stage_path(res), unsafe_allow_html=True)
        ex = res["stages"].get("explore", {})
        if ex.get("effect") is not None:
            ev = evidence({"verdict": res["verdict"], "explore_lo": ex.get("ci_low"), "explore_hi": ex.get("ci_high"),
                           "min_effect": spec.get("min_effect")})
            st.markdown(f"Explore effect **{ex['effect']:+.4f}** ({elo(ex['effect'])}), 95% CI "
                        f"[{ex.get('ci_low', 0):+.4f}, {ex.get('ci_high', 0):+.4f}], {ex.get('n_treated', 0):,} games. "
                        f"Evidence: **{evidence_text(ev, spec.get('min_effect'))}**.")
        st.caption(f"Why: {res.get('reason')}")
