"""
Builds the standalone interactive HTML headline explorer report — a
searchable/filterable client-side table of the most recent headlines, their
keep/drop status, and their scored sentiment/impact values.
"""
import json

import pandas as pd

from config import HTML_REPORT_PATH
from src.classify import clean_source_domain, human_drop_reason

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>GDELT Headline Explorer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com"/>
    <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap" rel="stylesheet"/>
    <style>
        :root {
            --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0; --accent: #2563eb;
            --accent2: #dc2626; --text: #1e293b; --muted: #64748b;
            --kept-bg: rgba(37,99,235,0.04); --drop-bg: rgba(220,38,38,0.03);
            --pos: #16a34a; --neg: #dc2626; --neu: #475569; --radius: 6px;
        }
        body { font-family: 'DM Mono', monospace; background: var(--bg); color: var(--text); margin: 0; padding: 0; }
        #header { background: #0f172a; color: white; padding: 1.5rem 2.5rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid var(--accent); }
        #header h1 { margin: 0; font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.3rem; letter-spacing: -0.02em; }
        #stats-bar { font-size: 0.75rem; color: #94a3b8; }
        #controls { display: flex; gap: 1.5rem; background: var(--surface); padding: 1rem 2.5rem; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
        .ctrl-group { display: flex; flex-direction: column; gap: 0.3rem; }
        .ctrl-group label { font-size: 0.65rem; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
        .ctrl-group input, .ctrl-group select { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 0.4rem 0.6rem; font-family: inherit; font-size: 0.75rem; color: var(--text); outline: none; }
        .btn-group { display: flex; gap: 0.25rem; }
        .btn-group button { background: var(--bg); border: 1px solid var(--border); padding: 0.4rem 0.8rem; font-family: inherit; font-size: 0.72rem; cursor: pointer; border-radius: 4px; color: var(--text); transition: all 0.1s; }
        .btn-group button.active { background: #0f172a; color: white; border-color: #0f172a; }
        .reset-btn { margin-left: 0.5rem; color: var(--accent2) !important; border-color: rgba(220,38,38,0.2) !important; }
        #result-count { padding: 0.6rem 2.5rem; font-size: 0.72rem; color: var(--muted); border-bottom: 1px solid var(--border); }
        #headline-list { padding: 1rem 2.5rem 3rem; display: flex; flex-direction: column; gap: 0.5rem; }
        .card { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); padding: 0.75rem 1rem; display: grid; grid-template-columns: 1fr auto; gap: 0.3rem 1rem; }
        .card.kept { background: var(--kept-bg); border-left: 3px solid var(--accent); }
        .card.dropped { background: var(--drop-bg); border-left: 3px solid var(--border); }
        .card-title { font-size: 0.88rem; color: #0f172a; line-height: 1.45; }
        .card-meta { font-size: 0.7rem; color: var(--muted); display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.2rem; }
        .card-meta .cat { color: var(--accent); background: rgba(37,99,235,0.08); border-radius: 3px; padding: 0 0.4rem; font-size: 0.65rem; }
        .badge { font-size: 0.65rem; padding: 0.15rem 0.4rem; border-radius: 3px; font-weight: 500; display: inline-block; }
        .badge.selected { background: rgba(22,163,74,0.1); color: var(--pos); }
        .badge.dropped { background: rgba(220,38,38,0.1); color: var(--accent2); }
        .badge.pos { background: var(--pos); color: white; }
        .badge.neg { background: var(--neg); color: white; }
        .badge.neu { background: var(--neu); color: white; }
        #no-results { display: none; padding: 3rem; text-align: center; color: var(--muted); font-size: 0.85rem; }
    </style>
</head>
<body>
    <div id="header">
        <h1>GDELT News Scored Data Feed</h1>
        <div id="stats-bar"></div>
    </div>
    <div id="controls">
        <div class="ctrl-group"><label>Search Headlines</label><input type="text" id="search-input" placeholder="e.g. fed, cpi, bitcoin..." autocomplete="off"/></div>
        <div class="ctrl-group"><label>Source Feed</label><select id="source-filter"><option value="">All feeds</option>__SOURCE_OPTIONS__</select></div>
        <div class="ctrl-group"><label>Date From</label><input type="date" id="date-from"/></div>
        <div class="ctrl-group"><label>Date To</label><input type="date" id="date-to"/></div>
        <div class="ctrl-group"><label>Pipeline Status</label><div class="btn-group"><button id="btn-all" class="active" onclick="setStatusFilter('all')">All</button><button id="btn-selected" onclick="setStatusFilter('selected')">Selected</button><button id="btn-dropped" onclick="setStatusFilter('dropped')">Dropped</button><button class="reset-btn" onclick="resetFilters()">Reset</button></div></div>
    </div>
    <div id="result-count"></div>
    <div id="headline-list"></div>
    <div id="no-results">No headlines match your filters.</div>
    <script>
        const RAW_DATA = __ROWS_JSON__; let statusFilter = 'all';
        function setStatusFilter(val) { statusFilter = val; document.querySelectorAll('#controls button').forEach(b => b.classList.remove('active')); const el = document.getElementById('btn-' + val); if (el) el.classList.add('active'); render(); }
        function resetFilters() { document.getElementById('search-input').value = ''; document.getElementById('source-filter').value = ''; document.getElementById('date-from').value = ''; document.getElementById('date-to').value = ''; setStatusFilter('all'); }
        function sentimentClass(s) { return s === 'positive' ? 'pos' : (s === 'negative' ? 'neg' : 'neu'); }
        function buildCard(row) {
            const statusBadge = row.kept ? `<span class="badge selected">\u2713 Kept</span>` : `<span class="badge dropped">\u2715 Dropped</span>`;
            const sentBadge = row.kept && row.sentiment ? `<span class="badge ${sentimentClass(row.sentiment)}">${row.sentiment}</span>` : '';
            const impactStr = row.kept ? `<b style="color:var(--accent)">Llama Score: ${row.opinion.toFixed(3)} | Impact Score: ${row.impact.toFixed(3)}</b>` : `<span style="color:var(--muted)">${row.drop_reason}</span>`;
            const catTag = row.kept && row.category ? `<span class="cat">${row.category}</span>` : ''; return `<div class="card ${row.kept ? 'kept' : 'dropped'}"><div class="card-title"><a href="${row.url}" target="_blank">${row.title}</a><div class="card-meta"><span class="src">${row.source}</span><span class="date">${row.date}</span>${catTag}</div></div><div class="badge-container" style="text-align:right">${statusBadge} ${sentBadge}<br/><div style="font-size:0.72rem;margin-top:0.4rem">${impactStr}</div></div></div>`; }
        function render() {
            const q = document.getElementById('search-input').value.toLowerCase(); const srcF = document.getElementById('source-filter').value;
            const dFrom = document.getElementById('date-from').value;
            const dTo = document.getElementById('date-to').value;
            const list = document.getElementById('headline-list');
            const none = document.getElementById('no-results'); const filtered = RAW_DATA.filter(r => {
                if (statusFilter === 'selected' && !r.kept) return false;
                if (statusFilter === 'dropped' && r.kept) return false;
                if (q && !r.title.toLowerCase().includes(q)) return false;
                if (srcF && r.source !== srcF) return false;
                if (dFrom && r.date_iso.slice(0,10) < dFrom) return false;
                if (dTo && r.date_iso.slice(0,10) > dTo) return false;
                return true;
            }); document.getElementById('result-count').innerHTML = `Showing <b>${filtered.length.toLocaleString()}</b> entries matches.`;
            if(filtered.length === 0) { list.innerHTML = ''; none.style.display='block'; } else { none.style.display='none'; list.innerHTML = filtered.map(buildCard).join(''); }
        }
        function updateStatsBar() {
            const total = RAW_DATA.length; const kept = RAW_DATA.filter(r => r.kept).length;
            document.getElementById('stats-bar').innerHTML = `Total Records Evaluated: <b>${total.toLocaleString()}</b> | Selected: <b>${kept.toLocaleString()}</b> | Dropped: <b>${(total-kept).toLocaleString()}</b>`; }
        updateStatsBar(); render(); </script>
</body>
</html>"""


def build_html_explorer(df_full, score_lookup, output_path=HTML_REPORT_PATH, max_rows=4000):
    """Render the interactive headline explorer to a single self-contained HTML file."""
    rows = []
    df_view = df_full.sort_values("dt", ascending=False).head(max_rows)

    for row in df_view.itertuples(index=False):
        title = str(getattr(row, "title", ""))
        source = str(getattr(row, "source", ""))
        url = str(getattr(row, "url", ""))
        dt_val = getattr(row, "dt", pd.NaT)
        dt_str = dt_val.strftime("%Y-%m-%d %H:%M") if pd.notna(dt_val) else "N/A"
        dt_iso = dt_val.isoformat() if pd.notna(dt_val) else ""
        kept = bool(getattr(row, "keep", False))
        reason_code = str(getattr(row, "drop_reason", ""))
        drop_reason = "Kept" if kept else human_drop_reason(reason_code)

        scored = score_lookup.get(url, {})
        if kept and scored:
            sentiment = scored.get("sentiment", "")
            opinion = float(scored.get("opinion", 0.0))
            impact = float(scored.get("impact", 0.0))
            category = scored.get("category", "")
        else:
            sentiment = ""
            opinion = 0.0
            impact = 0.0
            category = reason_code if kept else ""

        clean_src = clean_source_domain(source) or source
        rows.append({
            "title": title, "source": clean_src, "date": dt_str, "date_iso": dt_iso,
            "url": url, "kept": bool(kept), "drop_reason": drop_reason,
            "sentiment": sentiment, "impact": round(impact, 4),
            "opinion": round(opinion, 4), "category": category,
        })

    rows_json = json.dumps(rows, ensure_ascii=False)
    unique_sources = sorted(set(r["source"] for r in rows if r["source"]))
    source_options = "\n".join(f'<option value="{s}">{s}</option>' for s in unique_sources)

    html = _TEMPLATE.replace("__SOURCE_OPTIONS__", source_options).replace("__ROWS_JSON__", rows_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
