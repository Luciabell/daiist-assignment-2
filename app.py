"""uv run python main.py app

Dashboard. Loads weights/best.pt and weights/eval_artifacts.json. Trains nothing.
"""
import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import costing as C

A = json.load(open("weights/eval_artifacts.json"))
model = YOLO("weights/best.pt")
POS = A["positive_class_id"]
scores = np.array(A["cases"]["scores"])
is_viol = np.array(A["cases"]["is_violation"], bool)
REC, ASM = A["recommended"], A["assumptions"]


def predict(image, t_low, t_high):
    if image is None:
        return None, "Upload an image."
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    r = model(image, conf=t_low, verbose=False)[0]
    viol = [float(c) for c, k in zip(r.boxes.conf, r.boxes.cls) if int(k) == POS]
    s = max(viol, default=0.0)
    decision = "FAIL" if s >= t_high else "HUMAN REVIEW" if s >= t_low else "PASS"
    text = (f"### Decision: {decision}\n"
            f"- Highest '{A['config']['positive_class']}' confidence: {s:.2f}\n"
            f"- Thresholds: review from {t_low:.2f}, fail from {t_high:.2f}")
    return r.plot()[:, :, ::-1], text


def update(t_low, t_high, base, c_fn, c_fp, c_rev, cases, minutes):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    kw = dict(base_rate=base, c_fn=c_fn, c_fp=c_fp, c_review=c_rev)
    res = C.evaluate(scores, is_viol, t_low, t_high, **kw)
    ts, costs = C.sweep(scores, is_viol, **kw)
    t_best, c_best = float(ts[np.argmin(costs)]), float(costs.min())

    labels_r = ["Violation (no hardhat)", "Compliant"]
    cols = ["Pass", "Human review", "Fail"]
    cm_1000 = pd.DataFrame(res["per1000"].round(1), index=labels_r, columns=cols).reset_index()
    cm_raw = pd.DataFrame(res["cm"].astype(int), index=labels_r, columns=cols).reset_index()

    n_rev, hours, people = C.staffing(res["review_share"], cases, minutes)
    summary = (f"### Expected cost per 1,000 cases: **{res['cost_per_1000']:,.0f} EUR**\n"
               f"- Sent to human review: **{res['review_share']:.1%}** "
               f"({n_rev:.0f} cases/day, {hours:.1f} h/day, {people:.2f} people)\n"
               f"- Best single threshold (no review): {t_best:.2f} -> {c_best:,.0f} EUR / 1,000")

    strategies = pd.DataFrame([
        ["Do nothing (pass everyone)", base * 1000 * c_fn, 0.0],
        ["Human checks everything (assumes perfect human)", 1000 * c_rev,
         C.staffing(1.0, cases, minutes)[2]],
        [f"Model alone, best threshold {t_best:.2f}", c_best, 0.0],
        ["Model + review band (current sliders)", res["cost_per_1000"], people],
    ], columns=["Strategy", "Cost per 1,000 cases (EUR)", "People needed"]).round(1)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ts, costs, label="Cost (single threshold)")
    ax.scatter([t_best], [c_best], color="red", zorder=3, label=f"Min. single threshold: {t_best:.2f}")
    ax.axvspan(REC["t_low"], REC["t_high"], color="red", alpha=0.12,
               label=f"Chosen operating band ({REC['t_low']:.2f}-{REC['t_high']:.2f})")
    ax.axvline(t_low, ls="--", color="orange", label=f"t_low {t_low:.2f}")
    ax.axvline(t_high, ls="--", color="green", label=f"t_high {t_high:.2f}")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("EUR per 1,000 cases")
    ax.legend(fontsize=8); fig.tight_layout()
    return cm_1000, cm_raw, summary, strategies, fig


# ---- Static tables ----
pc = pd.DataFrame(A["per_class"]).T.round(3).reset_index().rename(columns={"index": "class"})
overall = pd.DataFrame([A["overall"]]).round(3)
stats_rows = []
for split, s in A["dataset_stats"].items():
    stats_rows.append({"split": split, "images": s["images"], **s["instances"],
                       "median box area (% of image)": s["median_box_area_pct_of_image"]})
stats = pd.DataFrame(stats_rows)

with gr.Blocks(title="Hard hat gate - cost dashboard") as demo:
    gr.Markdown("# Hard hat gate check\nPass / human review / fail, priced in euros.")
    with gr.Row():
        t_low = gr.Slider(0.05, 0.95, REC["t_low"], step=0.05, label="t_low (below = pass)")
        t_high = gr.Slider(0.05, 0.95, REC["t_high"], step=0.05, label="t_high (above = fail)")
    with gr.Accordion("Cost assumptions", open=False):
        with gr.Row():
            base = gr.Slider(0.005, 0.3, ASM["base_rate"], step=0.005, label="Base rate of violations")
            c_fn = gr.Number(ASM["c_fn"], label="Cost false negative (EUR)")
            c_fp = gr.Number(ASM["c_fp"], label="Cost false positive (EUR)")
            c_rev = gr.Number(ASM["c_review"], label="Cost per human review (EUR)")
        with gr.Row():
            cases = gr.Number(ASM["cases_per_day"], label="Cases per day")
            minutes = gr.Number(ASM["min_per_review"], label="Minutes per review")
    inputs = [t_low, t_high, base, c_fn, c_fp, c_rev, cases, minutes]

    with gr.Tabs():
        with gr.Tab("Predict"):
            img_in = gr.Image(type="pil", label="Upload image")
            btn = gr.Button("Run", variant="primary")
            img_out = gr.Image(label="Predictions")
            txt_out = gr.Markdown()
            btn.click(predict, [img_in, t_low, t_high], [img_out, txt_out])
        with gr.Tab("Cost & review"):
            summary = gr.Markdown()
            with gr.Row():
                cm1000 = gr.Dataframe(label="Confusion matrix: expected cases per 1,000 (at base rate)")
                cmraw = gr.Dataframe(label="Confusion matrix: raw test-set counts")
            plot = gr.Plot(label="Cost vs threshold")
            strat = gr.Dataframe(label="Strategy comparison")
        with gr.Tab("Per-class results"):
            gr.Markdown("Test split. Positive class: **" + A["config"]["positive_class"] + "**")
            gr.Dataframe(pc, label="Per class")
            gr.Dataframe(overall, label="Overall")
            gr.Dataframe(stats, label="Dataset description")
            if Path("weights/confusion_matrix.png").exists():
                gr.Image("weights/confusion_matrix.png", label="Detector confusion matrix (test)")

    outs = [cm1000, cmraw, summary, strat, plot]
    for comp in inputs:
        comp.change(update, inputs, outs)
    demo.load(update, inputs, outs)

if __name__ == "__main__":
    demo.launch()