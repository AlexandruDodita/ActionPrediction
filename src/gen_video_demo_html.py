"""Generate a bare-bones (no styling effort) local HTML page for the 20-clip demo:
pose-only / flow-only / combined GIFs + confidence, with two %% inputs to reweight
the combined confidence live. Run after export_video_demo.py.

Usage:
    python src/gen_video_demo_html.py
"""
import json

DATA_PATH = "demo/fusion_demo_data.json"
OUT_PATH = "demo/video_demo.html"

data = json.load(open(DATA_PATH))

rows_html = []
for c in data["demo_clips"]:
    name = c["name"]
    rows_html.append(f"""
  <div class="clip" data-name="{name}">
    <h3>{name} &mdash; true: <span class="true">{data['classes'][c['true']]}</span></h3>
    <div class="three">
      <div>
        <img src="videos/{name}_pose.gif" width="320">
        <p>key pose &mdash; pred: <span class="pose-pred"></span> &mdash; confidence: <span class="pose-conf"></span></p>
      </div>
      <div>
        <img src="videos/{name}_flow.gif" width="320">
        <p>optical flow &mdash; pred: <span class="flow-pred"></span> &mdash; confidence: <span class="flow-conf"></span></p>
      </div>
      <div>
        <img src="videos/{name}_combined.gif" width="320">
        <p>combined &mdash; pred: <span class="comb-pred"></span> &mdash; confidence: <span class="comb-conf"></span></p>
      </div>
    </div>
  </div>""")

html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>pose+flow fusion demo</title>
<style>
body {{ font-family: monospace; margin: 20px; }}
.three {{ display: flex; gap: 20px; }}
.clip {{ border-top: 1px solid #999; padding-top: 10px; margin-top: 20px; }}
.correct {{ color: green; }}
.wrong {{ color: red; }}
label {{ margin-right: 20px; }}
</style>
</head>
<body>

<h1>pose vs optical flow vs combined - confidence demo</h1>
<p>pose-only accuracy (full {data['n_test']}-clip test set): {data['acc_pose_only']*100:.1f}%<br>
flow-only accuracy: {data['acc_flow_only']*100:.1f}%</p>

<p>
<label>pose weight % <input id="wPose" type="number" min="0" max="100" value="50" step="1"></label>
<label>flow weight % <input id="wFlow" type="number" min="0" max="100" value="50" step="1"></label>
</p>

<p id="totals"></p>

<div id="clips">
{''.join(rows_html)}
</div>

<script>
const DATA = {json.dumps(data)};

function argmax(a) {{
  let bi = 0, bv = -1;
  for (let i = 0; i < a.length; i++) if (a[i] > bv) {{ bv = a[i]; bi = i; }}
  return [bi, bv];
}}

function pct(x) {{ return (x * 100).toFixed(1) + "%"; }}

function render() {{
  const wp = parseFloat(document.getElementById("wPose").value) || 0;
  const wf = parseFloat(document.getElementById("wFlow").value) || 0;
  const denom = (wp + wf) || 1;

  let nPose = 0, nFlow = 0, nComb = 0;
  const clipDivs = document.querySelectorAll(".clip");
  DATA.demo_clips.forEach((c, i) => {{
    const div = clipDivs[i];
    const [pi, pv] = argmax(c.pose_probs);
    const [fi, fv] = argmax(c.flow_probs);
    const comb = c.pose_probs.map((p, k) => (wp * p + wf * c.flow_probs[k]) / denom);
    const [ci, cv] = argmax(comb);

    const set = (cls, idx, val, correct) => {{
      const el = div.querySelector(cls);
      el.textContent = DATA.classes[idx] + " (" + pct(val) + ")";
      el.className = cls.slice(1) + " " + (correct ? "correct" : "wrong");
    }};
    set(".pose-pred", pi, pv, pi === c.true);
    set(".flow-pred", fi, fv, fi === c.true);
    set(".comb-pred", ci, cv, ci === c.true);
    div.querySelector(".pose-conf").textContent = pct(pv);
    div.querySelector(".flow-conf").textContent = pct(fv);
    div.querySelector(".comb-conf").textContent = pct(cv);

    nPose += pi === c.true;
    nFlow += fi === c.true;
    nComb += ci === c.true;
  }});

  document.getElementById("totals").textContent =
    `correct out of ${{DATA.demo_clips.length}}: pose=${{nPose}}  flow=${{nFlow}}  combined=${{nComb}}  (weights ${{wp}}/${{wf}})`;
}}

document.getElementById("wPose").addEventListener("input", render);
document.getElementById("wFlow").addEventListener("input", render);
render();
</script>

</body>
</html>
"""

import os
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
with open(OUT_PATH, "w") as f:
    f.write(html)
print(f"wrote {OUT_PATH}")
