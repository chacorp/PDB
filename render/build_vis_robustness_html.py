"""Build vis_CBD/vis_comparison/vis_robustness.html: topology robustness of PDB
(hole / decimate / subdiv) on ICT / Multiface / COMA test sets, same page style as
vis_comparison-3.html (metrics tables, then the qualitative figure).

Sources:
    metrics      : eval_robust/<ckpt>/{ict,mf,coma}/*.json   (eval_CBD_robust.py, eval_robust.sh)
    figure cells : vis_CBD/vis_comparison/robust_figure/<dataset>/  (render/cache_robust_figure.py)

Run from repo root:
    python render/build_vis_robustness_html.py
"""
import sys
import json
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import build_comparison_figure_html as fig_mod  # noqa: E402
from eval_CBD_robust import aggregate, CONDITION_ORDER  # noqa: E402

CKPT_NAME = "2026-04-27-15-36-49-NGBCv5"
EVAL_ROOT = REPO_ROOT / "eval_robust" / CKPT_NAME
FIG_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "robust_figure"
OUT_HTML = REPO_ROOT / "vis_CBD" / "vis_comparison" / "vis_robustness.html"
SCALE = 1e4

DATASETS = [  # (key, title, eval sub-folder, glob)
    ("ict", "ICT (ict-cap, test 20명 × 캡처 2개)", "ict", "ict-cap_*.json"),
    ("mf", "Multiface (mf_SEN + mf_ROM, 프레임 수 가중 평균)", "mf", "mf_*.json"),
    ("mf_SEN", "Multiface mf_SEN", "mf", "mf_SEN.json"),
    ("mf_ROM", "Multiface mf_ROM", "mf", "mf_ROM.json"),
    ("coma", "COMA (test 2명)", "coma", "coma_*.json"),
]
FIG_DATASETS = [("ict", "ICT"), ("mf", "Multiface"), ("coma", "COMA")]
COND_LABEL = {
    "clean": "clean", "hole_s": "hole S", "hole_l": "hole L", "hole_m": "hole ×5",
    "dec50": "decimate 50%", "dec25": "decimate 25%", "subdiv1": "subdiv ×1",
}
SIDE_LABEL = {"src": "source만", "tgt": "target만", "both": "모두"}
ROWS = [
    ("neutral", "Perturbed neutral"), ("gt", "GT expression"),
    ("self", "Self retargeting"), ("self_err", "Self error |pred−GT|"),
    ("cyc", "Cyclic (A→B→A)"), ("cyc_err", "Cyclic error |pred−GT|"),
    ("route", "Route |direct−via|"),
]

CSS = fig_mod.CSS.replace("background:#111; color:#eee;", "background:#fff; color:#111;") \
    .replace("nav { position:sticky; top:0; background:#111;", "nav { position:sticky; top:0; background:#fff;") + """
.metrics { margin-bottom:32px; }
.mblock { margin-top:18px; }
.mlabel { font-size:12px; font-weight:600; color:#444; margin:0 0 6px; }
.legend { font-size:12.5px; color:#555; line-height:1.55; max-width:1100px; }
.legend b { color:#222; }
dl.terms { font-size:12.5px; color:#444; display:grid; grid-template-columns:150px 1fr; gap:4px 12px; max-width:1100px; }
dl.terms dt { font-weight:600; color:#222; }
dl.terms dd { margin:0; }
table.mtable { border-collapse:collapse; font-size:12.5px; font-variant-numeric:tabular-nums; margin-bottom:8px; }
table.mtable th, table.mtable td { padding:5px 10px; text-align:right; white-space:nowrap; }
table.mtable thead th { font-size:10.5px; color:#888; border-bottom:2px solid #ddd; font-weight:500; }
table.mtable thead tr.group th { color:#333; font-size:11px; border-bottom:1px solid #ddd; text-align:center; }
table.mtable th.rowhead { text-align:left; color:#111; font-weight:600; }
table.mtable tbody td, table.mtable tbody th { border-bottom:1px solid #eee; }
table.mtable td.sep, table.mtable th.sep { border-left:1px solid #ddd; }
table.mtable .d { display:block; font-size:10.5px; }
.up1 { color:#b8860b; } .up2 { color:#d2691e; } .up3 { color:#c00; font-weight:600; } .flat { color:#999; } .down { color:#2a7; }
table.ov td { text-align:center; min-width:64px; }
table.grid td .rescale { font-size:10.5px; color:#c00; }
.errv { font-size:10.5px; color:#666; }
"""


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def load_group(sub, pattern):
    files = sorted((EVAL_ROOT / sub).glob(pattern))
    if not files:
        return None, 0, 0
    n = sum(json.load(open(f))["n_frames"] for f in files)
    return aggregate(files), n, len(files)


def baseline(agg, task, side, cond, subset):
    c = agg.get((task, side, cond, "clean_sub", subset)) if subset == "all" else None
    return c if c is not None else agg.get((task, "src", "clean", "pert", "all"))


def dcls(d):
    if d is None:
        return "flat"
    if d < -2:
        return "down"
    if d < 5:
        return "flat"
    if d < 25:
        return "up1"
    if d < 100:
        return "up2"
    return "up3"


def fmt_delta(d):
    if d is None:
        return ""
    if abs(d) >= 1000:
        return f"{d / 100 + 1:.0f}×"
    return f"{d:+.1f}%"


def cell(p, c, m, scale=True):
    if p is None:
        return '<td class="pending">-</td>'
    v = p[m] * (SCALE if scale else 1)
    d = None if c is None else (p[m] - c[m]) / (abs(c[m]) + 1e-20) * 100
    return f'<td>{v:.3f}<span class="d {dcls(d)}">{fmt_delta(d)}</span></td>'


def task_table(agg, task, metrics, sides):
    conds = CONDITION_ORDER
    h = ['<table class="mtable"><thead>']
    if len(sides) > 1:
        h.append('<tr class="group"><th></th>' + "".join(
            f'<th class="sep" colspan="{len(metrics)}">{SIDE_LABEL[s]} 변형</th>' for s in sides) + "</tr>")
    h.append('<tr><th class="rowhead">condition</th>' + "".join(
        f'<th class="{"sep" if i == 0 else ""}">{m}</th>' for _ in sides for i, (m, _) in enumerate(metrics)) + "</tr>")
    h.append("</thead><tbody>")
    for cond in conds:
        for subset in ("all", "orig"):
            if cond == "clean" and subset == "orig":
                continue
            if subset == "orig" and agg.get((task, sides[0], cond, "pert", "orig")) is None:
                continue
            label = COND_LABEL[cond] + (" (원본 vertex만)" if subset == "orig" else "")
            row = [f'<th class="rowhead">{label}</th>']
            for s in sides:
                if cond == "clean":
                    p = agg.get((task, "src", "clean", "pert", "all"))
                    c = p
                else:
                    p = agg.get((task, s, cond, "pert", subset))
                    c = baseline(agg, task, s, cond, subset)
                for i, (m, scale) in enumerate(metrics):
                    td = cell(p, c, m, scale)
                    row.append(td.replace("<td", '<td class="sep"', 1) if i == 0 else td)
            h.append("<tr>" + "".join(row) + "</tr>")
    h.append("</tbody></table>")
    return "\n".join(h)


def overview_table(groups):
    keys = [("self", "src", "MSE-in", "Self MSE-in"), ("cyc", "both", "MSE-in", "Cyclic MSE-in (모두)"),
            ("route", "both", "MSE", "Route MSE (모두)")]
    h = ['<table class="mtable ov"><thead><tr class="group"><th></th>' + "".join(
        f'<th class="sep" colspan="{len(CONDITION_ORDER) - 1}">{lab}</th>' for *_, lab in keys) + "</tr>"]
    h.append('<tr><th class="rowhead">dataset</th>' + "".join(
        f'<th class="{"sep" if i == 0 else ""}">{COND_LABEL[c]}</th>'
        for _ in keys for i, c in enumerate(CONDITION_ORDER[1:])) + "</tr></thead><tbody>")
    for key, title, agg in groups:
        if key in ("mf_SEN", "mf_ROM"):
            continue
        row = [f'<th class="rowhead">{title.split(" (")[0]}</th>']
        for task, side, m, _ in keys:
            for i, cond in enumerate(CONDITION_ORDER[1:]):
                p = agg.get((task, side, cond, "pert", "all"))
                c = baseline(agg, task, side, cond, "all")
                d = None if (p is None or c is None) else (p[m] - c[m]) / (abs(c[m]) + 1e-20) * 100
                sep = ' class="sep"' if i == 0 else ""
                row.append(f'<td{sep}><span class="{dcls(d)}">{fmt_delta(d) or "-"}</span></td>')
        h.append("<tr>" + "".join(row) + "</tr>")
    h.append("</tbody></table>")
    return "\n".join(h)


def attribution_table():
    cols = [("kw_clean+kd_clean", "kw clean · kd clean"), ("kw_pert+kd_clean", "kw 변형 · kd clean"),
            ("kw_clean+kd_pert", "kw clean · kd 변형"), ("kw_pert+kd_pert", "kw 변형 · kd 변형")]
    h = ['<table class="mtable"><thead><tr><th class="rowhead">dataset / condition</th>' +
         "".join(f"<th>{lab}</th>" for _, lab in cols) + "</tr></thead><tbody>"]
    for key, name in FIG_DATASETS:
        mp = FIG_ROOT / key / "meta.json"
        if not mp.exists():
            h.append(f'<tr><th class="rowhead">{name}</th><td class="pending" colspan="4">pending</td></tr>')
            continue
        meta = json.load(open(mp))
        for cond, v in meta["attribution"].items():
            base = v["kw_clean+kd_clean"]
            tds = []
            for k, _ in cols:
                d = (v[k] - base) / base * 100
                tds.append(f'<td>{v[k] * SCALE:.3f}<span class="d {dcls(d)}">{fmt_delta(d) if k != cols[0][0] else ""}</span></td>')
            h.append(f'<tr><th class="rowhead">{name} · {COND_LABEL[cond]}</th>' + "".join(tds) + "</tr>")
    h.append("</tbody></table>")
    return "\n".join(h)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def img(path):
    if not path.exists():
        return '<div class="missing">missing</div>'
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    tag = f'<div class="clip"><img src="data:image/png;base64,{b64}" loading="lazy"></div>'
    txt = path.with_suffix(".rescale.txt")
    if txt.exists():
        tag += f'<div class="rescale">&times;{txt.read_text().strip()}</div>'
    return tag


def figure_frame(key, frame, meta):
    d = FIG_ROOT / key / f"frame_{frame:06d}"
    fm = meta[f"frame_{frame:06d}"]
    p = ['<div class="figure-row">', '<table class="grid ref-grid"><tbody>',
         f'<tr><th class="rowhead">Source Neutral Mesh</th><td class="imgcell">{img(d / "ref_neutral.png")}</td></tr>',
         f'<tr><th class="rowhead">Source Expression Mesh</th><td class="imgcell">{img(d / "ref_expression.png")}</td></tr>',
         "</tbody></table>", '<table class="grid"><tbody>']
    for row, label in ROWS:
        p.append(f'<tr><th class="rowhead">{label}</th>')
        for c in CONDITION_ORDER:
            extra = ""
            if row in ("self_err", "cyc_err", "route"):
                extra = f'<div class="errv">{fm["mean_err"][c][row] * SCALE:.2f}</div>'
            p.append(f'<td class="imgcell">{img(d / f"{row}_{c}.png")}{extra}</td>')
        p.append("</tr>")
    p.append('</tbody><tfoot><tr><th class="rowhead-col"></th>' +
             "".join(f"<th>{COND_LABEL[c]}</th>" for c in CONDITION_ORDER) + "</tr></tfoot></table></div>")
    return "\n".join(p)


# ---------------------------------------------------------------------------
def main():
    groups = []
    blocks = []
    for key, title, sub, pattern in DATASETS:
        agg, n, nseq = load_group(sub, pattern)
        if agg is None:
            blocks.append(f'<h3 id="m_{key}">{title}</h3><p class="legend">pending (평가 결과 없음)</p>')
            continue
        groups.append((key, title, agg))
        blocks.append(f'<h3 id="m_{key}">{title} &mdash; {n:,} frames, {nseq} sequences</h3>')
        blocks.append('<div class="mblock"><p class="mlabel">Self retargeting (A → A, GT와 비교)</p>' +
                      task_table(agg, "self", [("MSE", True), ("MSE-in", True), ("MSE-out", True)], ["src"]) + "</div>")
        blocks.append('<div class="mblock"><p class="mlabel">Cyclic retargeting (A → B → A, GT와 비교)</p>' +
                      task_table(agg, "cyc", [("MSE", True), ("MSE-in", True), ("MSE-out", True)],
                                 ["src", "tgt", "both"]) + "</div>")
        blocks.append('<div class="mblock"><p class="mlabel">Route consistency (s → c vs s → b → c; nMSE는 ×1 그대로)</p>' +
                      task_table(agg, "route", [("MSE", True), ("MSE-in", True), ("nMSE", False)],
                                 ["src", "tgt", "both"]) + "</div>")

    fig_parts = []
    for key, name in FIG_DATASETS:
        mp = FIG_ROOT / key / "meta.json"
        if not mp.exists():
            fig_parts.append(f'<h2 id="fig_{key}">{name}</h2><p class="legend">pending</p>')
            continue
        meta = json.load(open(mp))
        fig_parts.append(f'<h2 id="fig_{key}">{name} &mdash; {meta["seq"]} '
                         f'(A={meta["src"]}, B={meta["cyc_B"]}; route b={meta["route_b"]}, c={meta["route_c"]})</h2>')
        for fr in meta["frames"]:
            fig_parts.append(f"<h3>frame {fr}</h3>")
            fig_parts.append(figure_frame(key, fr, meta))

    nav = ('<nav><a href="#terms">용어</a><a href="#overview">개요</a>' +
           "".join(f'<a href="#m_{k}">{k}</a>' for k, *_ in DATASETS) +
           '<a href="#attrib">원인 분리</a>' + "".join(f'<a href="#fig_{k}">figure {n}</a>' for k, n in FIG_DATASETS) + "</nav>")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>PDB topology robustness</title>
<style>{CSS}</style></head><body>
<h1>PDB topology robustness: hole / decimate / subdiv (ICT · Multiface · COMA test set)</h1>
<p class="legend">모델: <b>ckpts_CBD/{CKPT_NAME}/model_best.pth</b> (version 5, relu, align_latent).
평가: <code>eval_CBD_robust.py</code> (<code>eval_robust.sh</code>), figure: <code>render/cache_robust_figure.py</code>.
모든 test 프레임 사용.</p>
{nav}
<div class="metrics">
<h2 id="terms">용어 정의</h2>
<dl class="terms">
<dt>Self retargeting</dt><dd>identity A의 표정을 A 자신의 neutral에 다시 입힌 결과를 A의 GT와 비교.</dd>
<dt>Cyclic</dt><dd>A의 표정 → identity B → 다시 A로 보낸 결과를 A의 GT와 비교.</dd>
<dt>Route consistency</dt><dd>source s의 표정을 c로 직접 보낸 결과(direct)와 b를 거쳐 보낸 결과(via)의 차이. GT 불필요.
nMSE = MSE(direct, via) / MSE(direct, c의 neutral), 즉 움직임 크기 대비 불일치 비율.</dd>
<dt>hole S / L / ×5</dt><dd>얼굴 영역(plateau mask=1) vertex의 2% / 10% / 2%×5개를 k-ring(이웃 고리)으로 제거. 같은 topology면 같은 위치.</dd>
<dt>decimate 50% / 25%</dt><dd>igl.decimate(edge collapse)로 face를 50% / 25%로 줄이고, 남은 vertex를 원본 vertex로 snap → 원본 vertex의 부분집합.</dd>
<dt>subdiv ×1</dt><dd>smoothing 없는 midpoint subdivision(삼각형 1개 → 4개). 새 vertex = edge 중점.</dd>
<dt>P (변형 연산자)</dt><dd>원본 vertex → 변형 메쉬 vertex로 보내는 sparse 행렬. 변형 메쉬의 GT = P·GT (정확).</dd>
<dt>변형 위치</dt><dd><b>source만</b>: A(또는 s)만 변형. <b>target만</b>: B(또는 b, c)만 변형. <b>모두</b>: 등장하는 모든 메쉬 변형. self는 메쉬가 하나라 source만.</dd>
<dt>Δ%</dt><dd>(변형 결과 − clean 결과) / clean 결과. clean 결과는 <b>같은 vertex 부분집합</b>에서 계산(clean 출력에 P 적용).
평가 메쉬가 변형되지 않은 경우(cyclic target만, route source만)는 clean 전체 값과 비교. 1000% 이상은 배수(×)로 표시.</dd>
<dt>MSE / MSE-in / MSE-out</dt><dd>기존 eval_CBD / eval_CBD_cyc와 동일: 프레임별 MSE의 프레임 평균. in/out은 plateau hat mask(얼굴 안/밖)를 곱한 값
(self: r0=1.0, r1=2.25 / cyclic, route: r0=1.25, r1=2.5). MSE-out은 출력이 얼굴 밖에서 neutral과 얼마나 다른지.</dd>
<dt>kw / kd</dt><dd>kw = key_weight (key_weight_model이 neutral에서 예측하는 vertex별 cage 좌표, [V, 512]).
kd = key_d (exp_z_model + key_d_model이 표정 메쉬에서 인코딩한 cage 변위, [512, 3]). 출력 = kw · kd.</dd>
</dl>
<p class="legend">표의 값은 &times;1e-4 (표 값 &times; 1e-4 = 실제 MSE), 아래 작은 글씨가 Δ%.
색: <span class="flat">±5% 이내</span> · <span class="up1">+5~25%</span> · <span class="up2">+25~100%</span> ·
<span class="up3">+100% 이상</span> · <span class="down">감소</span>.
Multiface는 mf_SEN / mf_ROM을 프레임 수로 가중 평균(모든 프레임 값의 평균). 쌍 구성: ICT i → B=i+1, route b=i+1, c=i+2 (mod 20);
Multiface는 train ID 0을 B와 b로, train ID 1을 c로 사용; COMA는 다른 test ID를 B와 b로, COMA train ID 0을 c로 사용 (경유/타깃은 neutral만 사용).</p>

<h2 id="overview">개요: clean 대비 증가율 (Δ%)</h2>
{overview_table(groups)}

<h2 id="metrics">데이터셋별 상세 지표</h2>
{"".join(blocks)}

<h2 id="attrib">Decimation 원인 분리: key_weight vs 표정 인코딩</h2>
<p class="legend">대표 시퀀스 앞 256프레임, 모든 조건에서 self retargeting MSE(&times;1e-4)를 측정했습니다. kw와 kd를 각각 clean 메쉬 결과로 쓰거나(kw는 P를 적용한 뒤 행 합이 1이 되도록 정규화)
변형 메쉬에서 새로 예측해서 조합했습니다. 작은 글씨는 kw clean · kd clean 대비 Δ%입니다. <b>kw 변형 열이 kw 변형 · kd 변형 열과 비슷하면
오차의 주원인은 key_weight_model입니다.</b></p>
{attribution_table()}
</div>

<section class="figure-card figure-section"><h2 id="figure">Qualitative figure</h2>
<p class="legend">왼쪽: clean source neutral / GT 표정(neutral 대비 변위 heatmap). 오른쪽: 열 = 변형 조건(모든 메쉬 변형 = "모두"),
행 = 변형된 neutral / 변형 메쉬 위의 GT / self 결과 / self 오차 / cyclic 결과 / cyclic 오차 / route 불일치.
결과 행(GT, self, cyclic)의 heatmap(YlOrRd)은 neutral 대비 변위로, 셀마다 각자 정규화합니다.
<b>오차 행(self 오차, cyclic 오차, route)은 조건 간 비교를 위해 같은 스케일을 씁니다</b>: 상한 = clean 셀 값의 99퍼센타일 &times; 3이고, 넘는 값은 진한 빨강으로 잘립니다.
오차 셀 아래 숫자 = 그 프레임의 MSE(&times;1e-4). 화면 밖으로 나가는 출력은 중심을 맞춰 축소해서 그렸고, 빨간 &times;숫자가 축소 비율입니다.
Route 행은 최종 타깃 c의 메쉬 위에 그렸습니다. 프레임 = GT 변위가 가장 큰 프레임과 75퍼센타일 프레임.</p>
{"".join(fig_parts)}
</section></body></html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
