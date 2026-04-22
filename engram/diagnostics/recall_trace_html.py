"""Render a :class:`RecallTrace` as a self-contained HTML dashboard.

Single-file output: all CSS + JS inline, no external dependencies, no
server. Open the artifact directly in a browser or attach it to an
investigation ticket. The dashboard embeds the trace JSON verbatim so
the artifact is both a visual tool and a machine-readable snapshot.

**Visual structure.**

- **Header.** Query, intent verdict + confidence, timing summary.
- **Stage tabs.** [1] INTENT, [2] SEED, [3] EXPAND, [4] SCORE,
  [5] ASSEMBLE — click to switch.
- **Stage [1].** Per-intent cosine bar chart (CSS flexbox, no Chart.js).
  Chosen intent highlighted; margin vs threshold annotated.
- **Stage [2].** Query entity NER resolution status; merged seed table
  sortable by score / source / granularity with text previews.
- **Stage [3].** BFS step controls — Prev / Next / slider drill into
  each depth. Shows frontier-in size, edges considered/traversed,
  per-edge-type counts, cap hits, newly-reached nodes.
- **Stage [4].** Ranked passages with granularity color coding;
  expandable dropped-non-granule section.
- **Stage [5].** Fact breakdown + derived-rebuild indicator.

The raw JSON is toggleable at the bottom for deep inspection.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.diagnostics.recall_trace import RecallTrace


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>RecallTrace — {query_safe}</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: #0b1220; color: #d8e1ef;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 14px; line-height: 1.45;
}}
code, pre, .mono {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12.5px; }}
a {{ color: #7ab8ff; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

.wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}

header.trace-header {{
  background: linear-gradient(180deg, #162039 0%, #0e1730 100%);
  border: 1px solid #23314c;
  border-radius: 10px;
  padding: 18px 22px;
  margin-bottom: 20px;
}}
header.trace-header h1 {{
  margin: 0 0 6px 0; font-size: 16px; letter-spacing: 0.02em;
  color: #9fc3ff; font-weight: 600;
}}
header.trace-header .query {{
  font-size: 18px; color: #f2f5fb;
  overflow-wrap: anywhere;
}}
.meta-grid {{
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 14px;
}}
.meta-card {{
  background: #0c1428; border: 1px solid #1e2a45; border-radius: 8px;
  padding: 10px 12px;
}}
.meta-label {{ color: #7b8aa7; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }}
.meta-value {{ color: #e6ecf7; font-size: 15px; margin-top: 2px; font-weight: 600; }}
.meta-value.accent {{ color: #9fe8b1; }}
.meta-value.warn {{ color: #ffb072; }}
.meta-value.crit {{ color: #ff7b7b; }}

nav.tabs {{
  display: flex; gap: 4px; margin-bottom: 16px;
  border-bottom: 1px solid #23314c;
}}
nav.tabs button {{
  background: transparent; color: #8ba2c4; border: 0;
  padding: 10px 14px; cursor: pointer; font-size: 13px;
  border-bottom: 2px solid transparent;
  font-weight: 500;
}}
nav.tabs button:hover {{ color: #c7d5ef; }}
nav.tabs button.active {{ color: #7ab8ff; border-bottom-color: #4d8cf5; }}

.stage {{ display: none; }}
.stage.active {{ display: block; }}

.card {{
  background: #0f1830; border: 1px solid #1e2a45; border-radius: 10px;
  padding: 18px 20px; margin-bottom: 14px;
}}
.card h2 {{
  margin: 0 0 14px 0; font-size: 14px; color: #9fc3ff;
  text-transform: uppercase; letter-spacing: 0.06em;
}}
.card h3 {{ margin: 18px 0 8px 0; font-size: 13px; color: #bccbe4; }}

/* Intent bars */
.intent-row {{ display: flex; align-items: center; margin-bottom: 6px; }}
.intent-label {{ width: 170px; color: #c7d5ef; font-size: 13px; }}
.intent-bar-track {{
  flex: 1; background: #0a1223; border-radius: 4px; height: 22px; position: relative;
  overflow: hidden;
}}
.intent-bar-fill {{
  background: linear-gradient(90deg, #2a5aac 0%, #4d8cf5 100%);
  height: 100%;
}}
.intent-bar-fill.chosen {{
  background: linear-gradient(90deg, #2e7d5a 0%, #5fd3a0 100%);
}}
.intent-bar-fill.negative {{ background: linear-gradient(90deg, #6b2f2f 0%, #c9665a 100%); }}
.intent-bar-value {{
  position: absolute; right: 8px; top: 2px; color: #d8e1ef; font-size: 12px;
  font-family: "SF Mono", Menlo, Consolas, monospace;
}}
.margin-line {{
  display: flex; gap: 12px; align-items: center; margin-top: 12px;
  font-size: 13px; color: #bccbe4;
}}
.pill {{
  padding: 3px 9px; border-radius: 12px; font-size: 11px;
  font-weight: 600; letter-spacing: 0.04em;
}}
.pill.ok {{ background: #143d2a; color: #5fd3a0; }}
.pill.warn {{ background: #5a3a15; color: #ffb072; }}
.pill.crit {{ background: #4a1c1c; color: #ff7b7b; }}
.pill.neutral {{ background: #1e2a45; color: #9fc3ff; }}

/* Tables */
table.data {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
table.data th {{
  text-align: left; padding: 8px 10px; color: #7b8aa7;
  font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em;
  border-bottom: 1px solid #23314c;
}}
table.data td {{
  padding: 6px 10px; border-bottom: 1px solid #152036;
  color: #d8e1ef;
}}
table.data tr:hover td {{ background: #0a1223; }}
table.data td.num {{ text-align: right; font-family: "SF Mono", Menlo, Consolas, monospace; }}
table.data td.score {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-weight: 600; }}
.granularity-tag {{
  display: inline-block; padding: 1px 7px; border-radius: 10px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
}}
.granularity-tag.turn {{ background: #1a3a5c; color: #7ab8ff; }}
.granularity-tag.sentence {{ background: #1f4d3c; color: #5fd3a0; }}
.granularity-tag.ngram {{ background: #4c3a1a; color: #f2c268; }}
.granularity-tag.unknown {{ background: #2a2f3e; color: #8ba2c4; }}
.source-tag {{
  display: inline-block; padding: 1px 7px; border-radius: 4px;
  font-size: 10.5px; font-weight: 500;
  background: #142038; color: #9fc3ff;
}}
.source-tag.entity {{ background: #2f1a3e; color: #cd8cf2; }}
.source-tag.entity_granule {{ background: #2a1f42; color: #b08cff; }}
.source-tag.both {{ background: #224a2f; color: #9fe8b1; }}

/* BFS step controls */
.bfs-stepper {{
  display: flex; align-items: center; gap: 12px; padding: 14px 16px;
  background: #0a1223; border: 1px solid #1e2a45; border-radius: 8px;
  margin-bottom: 16px;
}}
.bfs-stepper button {{
  background: #1e2a45; color: #d8e1ef; border: 0; border-radius: 5px;
  padding: 7px 14px; font-size: 13px; cursor: pointer; font-weight: 500;
}}
.bfs-stepper button:hover:not(:disabled) {{ background: #2a3a5c; }}
.bfs-stepper button:disabled {{ opacity: 0.35; cursor: not-allowed; }}
.bfs-stepper .depth-label {{
  font-size: 14px; color: #9fc3ff; font-weight: 600;
  min-width: 110px;
}}
.bfs-stepper input[type=range] {{ flex: 1; }}

.step-detail {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
  margin-bottom: 16px;
}}
.stat-card {{
  background: #0a1223; border: 1px solid #1e2a45; border-radius: 8px;
  padding: 12px 14px;
}}
.stat-card .label {{ color: #7b8aa7; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
.stat-card .value {{ color: #e6ecf7; font-size: 20px; font-weight: 600; margin-top: 4px; }}
.stat-card .sub {{ color: #8ba2c4; font-size: 11px; margin-top: 2px; }}

.edge-type-grid {{
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px;
}}
.edge-row {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 4px 10px; background: #0a1223; border-radius: 4px;
}}
.edge-row .type {{ color: #c7d5ef; font-size: 12px; }}
.edge-row .w {{ color: #7b8aa7; font-size: 11px; }}
.edge-row .count {{ color: #9fe8b1; font-weight: 600; font-size: 12px; }}
.edge-row.zero .count {{ color: #6b7690; }}

.preview {{
  color: #9aabc7; font-size: 12px; font-style: italic;
}}

.details-toggle {{
  background: none; border: 1px dashed #2a3a5c; color: #8ba2c4;
  padding: 6px 12px; border-radius: 5px; cursor: pointer; font-size: 12px;
}}
.details-toggle:hover {{ border-color: #4d8cf5; color: #c7d5ef; }}
.collapsible {{ display: none; margin-top: 10px; }}
.collapsible.open {{ display: block; }}

.fact-kind-list {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.fact-kind {{
  padding: 6px 12px; background: #0a1223; border: 1px solid #1e2a45;
  border-radius: 6px;
}}
.fact-kind .k {{ color: #9fc3ff; font-size: 12px; }}
.fact-kind .n {{ color: #e6ecf7; font-weight: 600; font-size: 16px; margin-left: 6px; }}

.raw-json {{
  background: #070c18; border: 1px solid #1e2a45; border-radius: 8px;
  padding: 14px; max-height: 400px; overflow: auto; font-size: 11.5px;
  color: #bccbe4;
}}

footer {{
  margin-top: 24px; padding-top: 14px; border-top: 1px solid #1e2a45;
  color: #6b7690; font-size: 11px; text-align: center;
}}
</style>
</head>
<body>
<div class="wrap">

<header class="trace-header">
  <h1>RecallTrace</h1>
  <div class="query" id="hdr-query"></div>
  <div class="meta-grid">
    <div class="meta-card">
      <div class="meta-label">Intent</div>
      <div class="meta-value" id="hdr-intent"></div>
    </div>
    <div class="meta-card">
      <div class="meta-label">Intent Confidence</div>
      <div class="meta-value" id="hdr-intent-conf"></div>
    </div>
    <div class="meta-card">
      <div class="meta-label">Passages / Facts</div>
      <div class="meta-value" id="hdr-passfacts"></div>
    </div>
    <div class="meta-card">
      <div class="meta-label">Total</div>
      <div class="meta-value" id="hdr-total"></div>
    </div>
  </div>
</header>

<nav class="tabs">
  <button data-stage="intent" class="active">[1] Intent</button>
  <button data-stage="seed">[2] Seed</button>
  <button data-stage="expand">[3] Expand</button>
  <button data-stage="score">[4] Score</button>
  <button data-stage="assemble">[5] Assemble</button>
  <button data-stage="raw">Raw JSON</button>
</nav>

<section id="stage-intent" class="stage active"></section>
<section id="stage-seed" class="stage"></section>
<section id="stage-expand" class="stage"></section>
<section id="stage-score" class="stage"></section>
<section id="stage-assemble" class="stage"></section>
<section id="stage-raw" class="stage"></section>

<footer>engram · recall_trace dashboard</footer>
</div>

<script id="trace-data" type="application/json">{trace_json}</script>
<script>
(function() {{
  const TRACE = JSON.parse(document.getElementById("trace-data").textContent);

  function fmt(n, digits) {{
    if (n === null || n === undefined) return "—";
    const d = digits === undefined ? 3 : digits;
    return (typeof n === "number") ? n.toFixed(d) : String(n);
  }}
  function escapeHtml(s) {{
    return (s || "").replace(/[&<>"']/g, ch => ({{
      "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
    }})[ch]);
  }}

  // Header
  document.getElementById("hdr-query").textContent = TRACE.query || "";
  document.getElementById("hdr-intent").textContent = (TRACE.intent && TRACE.intent.chosen) || "—";
  if (TRACE.intent && TRACE.intent.fell_back) {{
    document.getElementById("hdr-intent").classList.add("warn");
    document.getElementById("hdr-intent").textContent += " (fallback)";
  }}
  const conf = TRACE.intent ? TRACE.intent.margin : 0;
  const confEl = document.getElementById("hdr-intent-conf");
  confEl.textContent = fmt(conf, 4);
  if (TRACE.intent && !TRACE.intent.used_hint) {{
    if (conf < TRACE.intent.margin_threshold) confEl.classList.add("crit");
    else if (conf < TRACE.intent.margin_threshold * 2) confEl.classList.add("warn");
    else confEl.classList.add("accent");
  }}
  const passCount = (TRACE.score && TRACE.score.selected) ? TRACE.score.selected.length : 0;
  const factCount = TRACE.assemble ? TRACE.assemble.facts_assembled : 0;
  document.getElementById("hdr-passfacts").textContent = passCount + " / " + factCount;
  document.getElementById("hdr-total").textContent = fmt(TRACE.timing_ms && TRACE.timing_ms.total_ms, 1) + "ms";

  // Tab routing
  document.querySelectorAll("nav.tabs button").forEach(btn => {{
    btn.addEventListener("click", () => {{
      document.querySelectorAll("nav.tabs button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".stage").forEach(s => s.classList.remove("active"));
      document.getElementById("stage-" + btn.dataset.stage).classList.add("active");
    }});
  }});

  // ============================================================
  // [1] INTENT
  // ============================================================
  (function renderIntent() {{
    const el = document.getElementById("stage-intent");
    const t = TRACE.intent;
    if (!t) {{ el.innerHTML = '<div class="card">no intent trace</div>'; return; }}

    let html = '<div class="card"><h2>[1] Intent Classification</h2>';

    if (t.used_hint) {{
      html += '<p>Intent hint supplied by caller — classification skipped.</p>';
      html += '<div><span class="pill neutral">chosen: ' + escapeHtml(t.chosen) + '</span></div>';
    }} else {{
      const entries = Object.entries(t.scores_by_intent || {{}}).sort((a,b) => b[1] - a[1]);
      const maxAbs = Math.max(...entries.map(([,v]) => Math.abs(v)), 1e-9);
      for (const [name, score] of entries) {{
        const pct = Math.abs(score) / maxAbs * 100;
        const negClass = score < 0 ? " negative" : "";
        const chosenClass = (name === t.chosen && !t.fell_back) ? " chosen" : "";
        html += '<div class="intent-row">';
        html += '<div class="intent-label">' + escapeHtml(name) + '</div>';
        html += '<div class="intent-bar-track">';
        html +=   '<div class="intent-bar-fill' + negClass + chosenClass + '" style="width:' + pct + '%"></div>';
        html +=   '<div class="intent-bar-value">' + fmt(score, 4) + '</div>';
        html += '</div></div>';
      }}
      html += '<div class="margin-line">';
      html += '<span>margin = ' + fmt(t.margin, 4) + '</span>';
      html += '<span>threshold = ' + fmt(t.margin_threshold, 4) + '</span>';
      if (t.fell_back) {{
        html += '<span class="pill crit">BELOW → fallback: ' + escapeHtml(t.fallback_intent) + '</span>';
      }} else if (t.margin < t.margin_threshold * 2) {{
        html += '<span class="pill warn">precarious — within 2× threshold</span>';
      }} else {{
        html += '<span class="pill ok">clear margin</span>';
      }}
      html += '</div>';
    }}
    html += '</div>';
    el.innerHTML = html;
  }})();

  // ============================================================
  // [2] SEED
  // ============================================================
  (function renderSeed() {{
    const el = document.getElementById("stage-seed");
    const t = TRACE.seed;
    if (!t) {{ el.innerHTML = '<div class="card">no seed trace</div>'; return; }}

    const surfaces = t.query_entity_surfaces || [];
    const resolved = (t.query_entity_ids || []).length;
    const unresolved = surfaces.length - resolved;

    let html = '<div class="card"><h2>[2] Seeding</h2>';
    html += '<div class="step-detail">';
    html += '<div class="stat-card"><div class="label">Semantic Seeds</div><div class="value">' + t.semantic_seed_count + '</div></div>';
    html += '<div class="stat-card"><div class="label">Entity Seeds</div><div class="value">' + t.entity_seed_count + '</div>' +
            (unresolved > 0 ? '<div class="sub">⚠ ' + unresolved + ' query NER mention(s) unresolved in registry</div>' : '') + '</div>';
    html += '<div class="stat-card"><div class="label">Merged (cap=' + t.total_cap + ')</div><div class="value">' + t.merged_seed_count +
            (t.was_capped ? ' <span class="pill warn">capped</span>' : '') + '</div></div>';
    html += '</div>';

    html += '<h3>Granularity Weights</h3>';
    const gw = t.granularity_weights || {{}};
    html += '<div style="display:flex;gap:8px">';
    for (const [g, w] of Object.entries(gw).sort()) {{
      html += '<div class="stat-card" style="flex:1"><div class="label">' + escapeHtml(g) + '</div><div class="value">' + fmt(w, 2) + '</div></div>';
    }}
    html += '</div>';

    if (surfaces.length) {{
      html += '<h3>Query NER</h3>';
      const resolvedIds = new Set(t.query_entity_ids || []);
      html += '<div style="display:flex;gap:6px;flex-wrap:wrap">';
      for (const surface of surfaces) {{
        html += '<span class="pill neutral">' + escapeHtml(surface) + '</span>';
      }}
      html += '</div>';
      html += '<div style="margin-top:6px;font-size:12px;color:#8ba2c4">resolved_ids: ' + resolved + ' of ' + surfaces.length + '</div>';
    }}

    html += '<h3>Merged Seeds (top ' + Math.min(50, (t.merged || []).length) + ')</h3>';
    html += '<table class="data"><thead><tr>';
    html += '<th>#</th><th>score</th><th>source</th><th>granularity</th><th>preview</th>';
    html += '</tr></thead><tbody>';
    (t.merged || []).slice(0, 50).forEach((e, i) => {{
      html += '<tr>';
      html += '<td>' + (i + 1) + '</td>';
      html += '<td class="score">' + fmt(e.score, 3) + '</td>';
      html += '<td><span class="source-tag ' + escapeHtml(e.source || "") + '">' + escapeHtml(e.source || "—") + '</span></td>';
      const g = e.granularity || "unknown";
      html += '<td><span class="granularity-tag ' + g + '">' + g + '</span></td>';
      html += '<td class="preview">' + escapeHtml((e.text_preview || "").slice(0, 120)) + '</td>';
      html += '</tr>';
    }});
    html += '</tbody></table></div>';
    el.innerHTML = html;
  }})();

  // ============================================================
  // [3] EXPAND  (with step controls)
  // ============================================================
  (function renderExpand() {{
    const el = document.getElementById("stage-expand");
    const t = TRACE.expand;
    if (!t) {{ el.innerHTML = '<div class="card">no expand trace</div>'; return; }}

    const steps = t.steps || [];
    const maxStep = steps.length;  // 0 = "summary" view

    let html = '<div class="card"><h2>[3] Bounded Typed-Edge BFS</h2>';

    html += '<div class="step-detail">';
    html += '<div class="stat-card"><div class="label">Seeds In</div><div class="value">' + t.seed_count + '</div></div>';
    html += '<div class="stat-card"><div class="label">Final Node Count</div><div class="value">' + t.final_node_count + '</div></div>';
    html += '<div class="stat-card"><div class="label">Total Edges Traversed</div><div class="value">' + t.total_edges_traversed + '</div></div>';
    html += '</div>';

    html += '<div class="bfs-stepper">';
    html += '<button id="bfs-prev">◀ Prev</button>';
    html += '<div class="depth-label" id="bfs-depth-label"></div>';
    html += '<input type="range" id="bfs-range" min="0" max="' + maxStep + '" value="0" />';
    html += '<button id="bfs-next">Next ▶</button>';
    html += '</div>';

    html += '<div id="bfs-step-view"></div>';

    html += '<h3>Edge Weights for This Intent</h3>';
    html += '<div class="edge-type-grid">';
    const weights = t.edge_weights || {{}};
    for (const [et, w] of Object.entries(weights).sort()) {{
      html += '<div class="edge-row' + (w === 0 ? ' zero' : '') + '">';
      html += '<span class="type">' + escapeHtml(et) + '</span>';
      html += '<span class="w">w=' + fmt(w, 2) + '</span>';
      html += '</div>';
    }}
    html += '</div>';

    html += '<div style="margin-top:12px;font-size:12px;color:#8ba2c4">';
    html += 'max_depth=' + t.max_depth + ' · max_frontier=' + t.max_frontier;
    html += '</div>';

    html += '</div>';
    el.innerHTML = html;

    // Step controller
    const range = document.getElementById("bfs-range");
    const prev = document.getElementById("bfs-prev");
    const next = document.getElementById("bfs-next");
    const label = document.getElementById("bfs-depth-label");
    const view = document.getElementById("bfs-step-view");

    function renderStep(idx) {{
      if (idx === 0) {{
        label.textContent = "All depths";
        let h = '<div class="step-detail">';
        for (const step of steps) {{
          h += '<div class="stat-card">';
          h += '<div class="label">Depth ' + step.depth + '</div>';
          h += '<div class="value">' + step.newly_reached + '</div>';
          h += '<div class="sub">new · ' + step.edges_traversed + ' edges · ';
          h += step.frontier_out_size + (step.was_capped ? ' (capped)' : '') + ' frontier</div>';
          h += '</div>';
        }}
        h += '</div>';
        view.innerHTML = h;
      }} else {{
        const s = steps[idx - 1];
        label.textContent = "Depth " + s.depth + " / " + steps.length;
        let h = '<div class="step-detail">';
        h += '<div class="stat-card"><div class="label">Frontier In</div><div class="value">' + s.frontier_in_size + '</div></div>';
        h += '<div class="stat-card"><div class="label">Edges Traversed</div><div class="value">' + s.edges_traversed + '</div><div class="sub">of ' + s.edges_considered + ' considered</div></div>';
        h += '<div class="stat-card"><div class="label">Newly Reached</div><div class="value">' + s.newly_reached + '</div></div>';
        h += '</div>';
        h += '<div class="step-detail">';
        h += '<div class="stat-card"><div class="label">Frontier Out (pre-cap)</div><div class="value">' + s.frontier_out_size_before_cap + '</div></div>';
        h += '<div class="stat-card"><div class="label">Frontier Out</div><div class="value">' + s.frontier_out_size +
             (s.was_capped ? ' <span class="pill warn">capped</span>' : '') + '</div></div>';
        h += '<div class="stat-card"><div class="label">Cap</div><div class="value">' + TRACE.expand.max_frontier + '</div></div>';
        h += '</div>';
        h += '<h3>Edges Traversed by Type</h3>';
        h += '<div class="edge-type-grid">';
        const et = Object.entries(s.edges_by_type || {{}}).sort((a,b) => b[1] - a[1]);
        if (!et.length) h += '<div class="edge-row zero"><span class="type">(none)</span><span class="count">0</span></div>';
        for (const [name, count] of et) {{
          h += '<div class="edge-row"><span class="type">' + escapeHtml(name) + '</span><span class="count">' + count + '</span></div>';
        }}
        h += '</div>';
        view.innerHTML = h;
      }}
      range.value = idx;
      prev.disabled = idx === 0;
      next.disabled = idx === maxStep;
    }}

    range.addEventListener("input", () => renderStep(parseInt(range.value, 10)));
    prev.addEventListener("click", () => renderStep(Math.max(0, parseInt(range.value, 10) - 1)));
    next.addEventListener("click", () => renderStep(Math.min(maxStep, parseInt(range.value, 10) + 1)));
    renderStep(0);
  }})();

  // ============================================================
  // [4] SCORE
  // ============================================================
  (function renderScore() {{
    const el = document.getElementById("stage-score");
    const t = TRACE.score;
    if (!t) {{ el.innerHTML = '<div class="card">no score trace</div>'; return; }}

    let html = '<div class="card"><h2>[4] Scoring + Passage Selection</h2>';
    html += '<div class="step-detail">';
    html += '<div class="stat-card"><div class="label">Walk Nodes</div><div class="value">' + t.walk_node_count + '</div></div>';
    html += '<div class="stat-card"><div class="label">Granules Considered</div><div class="value">' + t.granules_considered + '</div></div>';
    html += '<div class="stat-card"><div class="label">Selected</div><div class="value">' + (t.selected || []).length + '</div><div class="sub">of max ' + t.max_passages + '</div></div>';
    html += '</div>';

    html += '<h3>Ranked Passages</h3>';
    html += '<table class="data"><thead><tr><th>#</th><th>score</th><th>granularity</th><th>preview</th></tr></thead><tbody>';
    (t.selected || []).forEach((g, i) => {{
      html += '<tr>';
      html += '<td>' + (i + 1) + '</td>';
      html += '<td class="score">' + fmt(g.score, 3) + '</td>';
      html += '<td><span class="granularity-tag ' + g.granularity + '">' + g.granularity + '</span></td>';
      html += '<td class="preview">' + escapeHtml((g.text_preview || "").slice(0, 160)) + '</td>';
      html += '</tr>';
    }});
    html += '</tbody></table>';

    if ((t.dropped_non_granules || []).length) {{
      html += '<h3>Top Walk Hits That Didn\'t Surface as Passages</h3>';
      html += '<p style="color:#8ba2c4;font-size:12px">Entities, claims, preferences, and time anchors reached by the walk but not routed to a granule. These are the "scaffolding" — their scores are usable signal but they aren\'t returned to the agent as passages.</p>';
      html += '<table class="data"><thead><tr><th>score</th><th>label</th><th>node_id</th></tr></thead><tbody>';
      (t.dropped_non_granules || []).forEach(d => {{
        html += '<tr>';
        html += '<td class="score">' + fmt(d.score, 3) + '</td>';
        html += '<td>' + escapeHtml(d.label) + '</td>';
        html += '<td class="mono" style="font-size:11px;color:#8ba2c4">' + escapeHtml(d.node_id) + '</td>';
        html += '</tr>';
      }});
      html += '</tbody></table>';
    }}
    html += '</div>';
    el.innerHTML = html;
  }})();

  // ============================================================
  // [5] ASSEMBLE
  // ============================================================
  (function renderAssemble() {{
    const el = document.getElementById("stage-assemble");
    const t = TRACE.assemble;
    if (!t) {{ el.innerHTML = '<div class="card">no assemble trace</div>'; return; }}

    let html = '<div class="card"><h2>[5] Assembly</h2>';
    html += '<div class="step-detail">';
    html += '<div class="stat-card"><div class="label">Passages Assembled</div><div class="value">' + t.passages_assembled + '</div></div>';
    html += '<div class="stat-card"><div class="label">Facts Assembled</div><div class="value">' + t.facts_assembled + '</div></div>';
    html += '<div class="stat-card"><div class="label">Derived Index</div><div class="value">' +
            (t.derived_rebuilt ? 'rebuilt' : 'fresh') +
            '</div><div class="sub">' + (t.derived_was_stale ? 'stale on entry' : 'fingerprint matched') + '</div></div>';
    html += '</div>';

    if (t.facts_by_type && Object.keys(t.facts_by_type).length) {{
      html += '<h3>Facts by Kind</h3>';
      html += '<div class="fact-kind-list">';
      for (const [k, n] of Object.entries(t.facts_by_type).sort()) {{
        html += '<div class="fact-kind"><span class="k">' + escapeHtml(k) + '</span><span class="n">' + n + '</span></div>';
      }}
      html += '</div>';
    }} else {{
      html += '<p style="color:#8ba2c4">No facts pulled from derived indexes for this intent.</p>';
    }}

    html += '<h3>Stage Timings</h3>';
    html += '<div class="step-detail" style="grid-template-columns:repeat(6,1fr)">';
    for (const [k, v] of Object.entries(TRACE.timing_ms || {{}}).sort()) {{
      html += '<div class="stat-card"><div class="label">' + escapeHtml(k.replace("_ms", "")) + '</div><div class="value">' + fmt(v, 1) + '</div><div class="sub">ms</div></div>';
    }}
    html += '</div>';

    html += '</div>';
    el.innerHTML = html;
  }})();

  // ============================================================
  // RAW JSON
  // ============================================================
  (function renderRaw() {{
    const el = document.getElementById("stage-raw");
    const blob = JSON.stringify(TRACE, null, 2);
    el.innerHTML = '<div class="card"><h2>Raw Trace</h2><pre class="raw-json">' + escapeHtml(blob) + '</pre></div>';
  }})();

}})();
</script>
</body>
</html>
"""


def render_html(trace: "RecallTrace") -> str:
    """Render ``trace`` as a self-contained HTML dashboard.

    The returned string is a complete HTML document — inline CSS + JS, no
    external fetches. Write it to a file, open it in a browser.
    """
    trace_dict = trace.to_dict()
    # Embed JSON. We use a <script type="application/json"> tag; escape the
    # only real hazard (</script> appearing inside the blob).
    trace_json = json.dumps(trace_dict, separators=(",", ":"))
    trace_json = trace_json.replace("</", "<\\/")
    query_safe = html.escape(trace.query or "")
    return _TEMPLATE.format(trace_json=trace_json, query_safe=query_safe)


__all__ = ["render_html"]
