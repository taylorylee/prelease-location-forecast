"""Interactive map of pre-registered forecasts with their uncertainty.

    python make_map.py predictions/preregistration_<date>.csv  ->  reports/forecast_map.html

Circle color = P(open at 24 months). Ring size = width of the model's uncertainty band, so a big
faint ring means 'the model has seen few sites like this; trust it less'.
"""
import sys, json
import pandas as pd
from src import config

src = sys.argv[1]
d = pd.read_csv(src)
pts = d[["site_id", "address", "category", "lat", "lon", "p_open_24mo", "p_band_10", "p_band_90",
         "conformal_80", "forecast_date"]].to_dict("records")

html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pre-lease Forecasts</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{{--bg:#fafaf7;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e4e2dc}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#151514;--fg:#ecebe6;--muted:#9a9993;--card:#1f1f1d;--line:#33332f}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
header{{padding:16px}} h1{{font-size:20px;margin:0 0 4px}} p{{margin:0;color:var(--muted)}}
#map{{height:72vh;margin:0 16px;border-radius:10px;border:1px solid var(--line)}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;padding:10px 16px;color:var(--muted);font-size:13px}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle}}
</style></head><body>
<header><h1>Pre-lease survival forecasts</h1>
<p>P(still open 24 months after opening). Pre-registered {d.forecast_date.iloc[0]}; scored against DataSF at 6, 12 and 24 months.</p></header>
<div id="map"></div>
<div class="legend"><span><i class="sw" style="background:#2e7d4f"></i>&ge; 65%</span>
<span><i class="sw" style="background:#d39b2a"></i>45&ndash;65%</span><span><i class="sw" style="background:#b8453a"></i>&lt; 45%</span>
<span>Outer ring = uncertainty band (10th&ndash;90th pct of trees)</span></div>
<script>
const pts = {json.dumps(pts)};
const map = L.map('map').setView([37.765,-122.43],13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{attribution:'&copy; OpenStreetMap &copy; CARTO'}}).addTo(map);
const col = p => p>=.65?'#2e7d4f':p>=.45?'#d39b2a':'#b8453a';
const b=[];
pts.forEach(s=>{{
  const w=(s.p_band_90-s.p_band_10);
  L.circleMarker([s.lat,s.lon],{{radius:8+w*60,color:col(s.p_open_24mo),weight:1,fillOpacity:.12}}).addTo(map);
  L.circleMarker([s.lat,s.lon],{{radius:7,color:'#fff',weight:1.5,fillColor:col(s.p_open_24mo),fillOpacity:.95}})
   .bindPopup(`<b>${{s.site_id}}</b> &middot; ${{s.category}}<br>${{s.address}}<br>
   <b>${{Math.round(s.p_open_24mo*100)}}%</b> open at 24 mo (band ${{Math.round(s.p_band_10*100)}}&ndash;${{Math.round(s.p_band_90*100)}}%)<br>
   80% conformal: ${{s.conformal_80}}`).addTo(map);
  b.push([s.lat,s.lon]);
}});
if(b.length) map.fitBounds(b,{{padding:[40,40]}});
</script></body></html>"""
out = config.REPORTS / "forecast_map.html"
out.write_text(html)
print(f"wrote {out}")
