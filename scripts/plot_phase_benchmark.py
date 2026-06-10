#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path


PHASES = ["commit", "sumcheck", "e2e_prove"]
PHASE_LABEL = {
    "commit": "commit",
    "sumcheck": "sumcheck",
    "e2e_prove": "e2e prove",
}
BACKEND_LABEL = {
    "cpu": "CPU",
    "gpu-metal": "GPU Metal",
}
BACKEND_COLOR = {
    "cpu": "#2f4b7c",
    "gpu-metal": "#d95f02",
}
FOLD_COLOR = {
    1: "#2f4b7c",
    2: "#d95f02",
    3: "#1b9e77",
    4: "#7570b3",
    6: "#e7298a",
}
PROFILE_COLOR = {
    "total": "#1f2937",
    "command wait": "#7570b3",
    "upload": "#d95f02",
    "readback": "#1b9e77",
    "blit": "#e7298a",
}


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_index(rows):
    out = {}
    for r in rows:
        key = (r["backend"], r["phase"], r["log_size"], r["fold"], r["rate"])
        out[key] = r
    return out


def paired_rows(rows):
    idx = row_index(rows)
    keys = sorted(
        {
            (r["phase"], r["log_size"], r["fold"], r["rate"])
            for r in rows
        },
        key=lambda k: (k[1], k[2], k[3], k[0]),
    )
    pairs = []
    for phase, log_size, fold, rate in keys:
        cpu = idx.get(("cpu", phase, log_size, fold, rate))
        gpu = idx.get(("gpu-metal", phase, log_size, fold, rate))
        if cpu or gpu:
            pairs.append((phase, log_size, fold, rate, cpu, gpu))
    return pairs


def gib(x):
    return x / 1024.0 / 1024.0 / 1024.0


def mib(x):
    return x / 1024.0 / 1024.0


def fmt_ms(ms):
    if ms is None:
        return "OOM"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def fmt_speedup(x):
    if x is None:
        return "OOM"
    return f"{x:.2f}x"


def fmt_gib(x):
    if x is None:
        return "OOM"
    return f"{gib(x):.1f}"


def nice_ticks(vmin, vmax, log_y=False, count=5):
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= 0:
        return []
    if log_y:
        lo = math.floor(math.log10(max(vmin, 1e-9)))
        hi = math.ceil(math.log10(vmax))
        ticks = []
        for p in range(lo, hi + 1):
            for m in (1, 2, 5):
                value = m * (10 ** p)
                if vmin <= value <= vmax:
                    ticks.append(value)
        if len(ticks) > 7:
            stride = max(1, math.ceil(len(ticks) / 7))
            ticks = ticks[::stride]
        return ticks
    if vmax == vmin:
        return [vmin]
    raw = (vmax - vmin) / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 5, 10), key=lambda s: abs(raw - s * mag)) * mag
    start = math.floor(vmin / step) * step
    ticks = []
    value = start
    while value <= vmax + step * 0.5:
        if value >= vmin - step * 0.5:
            ticks.append(value)
        value += step
    return ticks[:8]


def svg_text(x, y, text, size=12, anchor="middle", weight="400", fill="#111827"):
    text = html.escape(str(text))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{text}</text>'
    )


def svg_poly(points, color, width=2.2, dash=False):
    if len(points) < 2:
        return ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = ' stroke-dasharray="5 4"' if dash else ""
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
    )


def render_line_chart(
    path,
    title,
    panels,
    y_label,
    x_label="log2(size)",
    log_y=False,
    shared_y=True,
    width=1280,
    panel_height=260,
):
    panel_count = len(panels)
    height = 96 + panel_count * panel_height + 28
    margin_l, margin_r, margin_t, margin_b = 76, 24, 74, 46
    plot_w = width - margin_l - margin_r
    all_x = [
        x
        for panel in panels
        for series in panel["series"].values()
        for x, y in series
        if y is not None and y > 0
    ]
    xmin, xmax = min(all_x), max(all_x)

    def y_range(panel):
        values = [
            y
            for series in panel["series"].values()
            for x, y in series
            if y is not None and y > 0 and math.isfinite(y)
        ]
        if not values:
            return 0.1, 1.0
        lo, hi = min(values), max(values)
        if log_y:
            return lo / 1.35, hi * 1.35
        pad = (hi - lo) * 0.12 if hi > lo else max(1.0, hi * 0.1)
        return max(0.0, lo - pad), hi + pad

    shared_range = None
    if shared_y:
        vals = []
        for panel in panels:
            a, b = y_range(panel)
            vals.extend([a, b])
        shared_range = min(vals), max(vals)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 32, title, 21, weight="700"),
        svg_text(width / 2, height - 8, x_label, 12),
        svg_text(18, height / 2, y_label, 12, anchor="middle"),
    ]
    parts.append(f'<g transform="rotate(-90 18 {height / 2:.1f})"></g>')

    legend_items = {}
    for panel in panels:
        for name, color in panel.get("colors", {}).items():
            legend_items[name] = color
    lx = margin_l
    ly = 54
    for name, color in legend_items.items():
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(svg_text(lx + 34, ly + 4, name, 12, anchor="start"))
        lx += 150

    for i, panel in enumerate(panels):
        top = margin_t + i * panel_height
        bottom = top + panel_height - margin_b
        y0 = top + 26
        plot_h = bottom - y0
        vmin, vmax = shared_range if shared_y else y_range(panel)
        if log_y:
            vmin = max(vmin, 1e-9)
            lmin, lmax = math.log10(vmin), math.log10(vmax)

            def sy(v):
                return bottom - (math.log10(max(v, 1e-9)) - lmin) / (lmax - lmin) * plot_h

        else:

            def sy(v):
                return bottom - (v - vmin) / (vmax - vmin) * plot_h if vmax > vmin else bottom

        def sx(x):
            return margin_l + (x - xmin) / (xmax - xmin) * plot_w if xmax > xmin else margin_l + plot_w / 2

        parts.append(svg_text(margin_l, top + 14, panel["title"], 15, anchor="start", weight="700"))
        parts.append(f'<line x1="{margin_l}" y1="{bottom}" x2="{width - margin_r}" y2="{bottom}" stroke="#111827" stroke-width="1"/>')
        parts.append(f'<line x1="{margin_l}" y1="{y0}" x2="{margin_l}" y2="{bottom}" stroke="#111827" stroke-width="1"/>')

        x_span = int(xmax) - int(xmin)
        x_step = 1 if x_span <= 18 else 2 if x_span <= 36 else 4
        for x in range(int(xmin), int(xmax) + 1, x_step):
            px = sx(x)
            parts.append(f'<line x1="{px:.1f}" y1="{bottom}" x2="{px:.1f}" y2="{bottom + 5}" stroke="#111827"/>')
            parts.append(svg_text(px, bottom + 20, str(x), 11))
        for tick in nice_ticks(vmin, vmax, log_y=log_y):
            py = sy(tick)
            parts.append(f'<line x1="{margin_l}" y1="{py:.1f}" x2="{width - margin_r}" y2="{py:.1f}" stroke="#e5e7eb"/>')
            label = f"{tick:g}" if tick < 1000 else f"{tick / 1000:g}k"
            parts.append(svg_text(margin_l - 9, py + 4, label, 10, anchor="end", fill="#374151"))

        for name, series in panel["series"].items():
            color = panel.get("colors", {}).get(name, "#111827")
            points = [(sx(x), sy(y)) for x, y in series if y is not None and y > 0]
            parts.append(svg_poly(points, color))
            for px, py in points:
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.1" fill="{color}" stroke="#ffffff" stroke-width="1"/>')

        for note in panel.get("annotations", []):
            x, y, text = note
            parts.append(svg_text(sx(x), sy(y), text, 11, fill="#b91c1c", weight="700"))

    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_csv(path, rows):
    pairs = paired_rows(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "log_size",
                "size",
                "fold",
                "rate",
                "phase",
                "cpu_ms",
                "gpu_ms",
                "speedup",
                "cpu_peak_gib",
                "gpu_peak_gib",
                "gpu_upload_gib",
                "gpu_upload_ms",
                "gpu_readback_mib",
                "gpu_readback_ms",
                "gpu_command_wait_ms",
                "gpu_blit_gib",
                "gpu_blit_wait_ms",
            ]
        )
        for phase, log_size, fold, rate, cpu, gpu in pairs:
            cpu_ms = cpu and cpu.get("duration_ms")
            gpu_ms = gpu and gpu.get("duration_ms")
            speedup = cpu_ms / gpu_ms if cpu_ms and gpu_ms else None
            writer.writerow(
                [
                    log_size,
                    1 << log_size,
                    fold,
                    rate,
                    phase,
                    f"{cpu_ms:.6f}" if cpu_ms else "",
                    f"{gpu_ms:.6f}" if gpu_ms else "",
                    f"{speedup:.6f}" if speedup else "",
                    f"{gib(cpu['peak_allocated_bytes']):.6f}" if cpu else "",
                    f"{gib(gpu['peak_allocated_bytes']):.6f}" if gpu else "",
                    f"{gib(gpu.get('metal_upload_bytes', 0)):.6f}" if gpu else "",
                    f"{gpu.get('metal_upload_ms', 0):.6f}" if gpu else "",
                    f"{mib(gpu.get('metal_readback_bytes', 0)):.6f}" if gpu else "",
                    f"{gpu.get('metal_readback_ms', 0):.6f}" if gpu else "",
                    f"{gpu.get('metal_command_wait_ms', 0):.6f}" if gpu else "",
                    f"{gib(gpu.get('metal_blit_bytes', 0)):.6f}" if gpu else "",
                    f"{gpu.get('metal_blit_wait_ms', 0):.6f}" if gpu else "",
                ]
            )


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--report", default="outputs/article_phase_benchmark_16_27_hybrid_report.md")
    parser.add_argument("--paired-csv", default="outputs/article_phase_benchmark_16_27_hybrid_paired.csv")
    parser.add_argument("--requested-min-log-size", type=int)
    parser.add_argument("--requested-max-log-size", type=int)
    args = parser.parse_args()

    rows = load_rows(args.jsonl)
    idx = row_index(rows)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.paired_csv, rows)

    logs = sorted({r["log_size"] for r in rows})
    folds = sorted({r["fold"] for r in rows})
    rates = sorted({r["rate"] for r in rows})

    def get(backend, phase, log_size, fold, rate):
        return idx.get((backend, phase, log_size, fold, rate))

    def series_backend(backend, phase, fold, rate, metric):
        points = []
        for log_size in logs:
            r = get(backend, phase, log_size, fold, rate)
            points.append((log_size, metric(r) if r else None))
        return points

    e2e_panels = []
    for fold in folds:
        annotations = []
        if not get("cpu", "e2e_prove", 27, fold, 1) or not get("gpu-metal", "e2e_prove", 27, fold, 1):
            max_y = max(
                [
                    r["duration_ms"]
                    for b in ("cpu", "gpu-metal")
                    for log_size in logs
                    for r in [get(b, "e2e_prove", log_size, fold, 1)]
                    if r
                ]
                or [1]
            )
            annotations.append((27, max_y, "missing/killed"))
        e2e_panels.append(
            {
                "title": f"fold={fold}, rate=1",
                "series": {
                    "CPU": series_backend("cpu", "e2e_prove", fold, 1, lambda r: r["duration_ms"] if r else None),
                    "GPU Metal": series_backend("gpu-metal", "e2e_prove", fold, 1, lambda r: r["duration_ms"] if r else None),
                },
                "colors": {"CPU": BACKEND_COLOR["cpu"], "GPU Metal": BACKEND_COLOR["gpu-metal"]},
                "annotations": annotations,
            }
        )
    render_line_chart(
        plot_dir / "e2e_runtime_rate1.svg",
        "E2E prove runtime, rate=1",
        e2e_panels,
        "milliseconds, log scale",
        log_y=True,
        shared_y=True,
    )

    speed_series = {}
    for fold in folds:
        pts = []
        for log_size in logs:
            cpu = get("cpu", "e2e_prove", log_size, fold, 1)
            gpu = get("gpu-metal", "e2e_prove", log_size, fold, 1)
            pts.append((log_size, cpu["duration_ms"] / gpu["duration_ms"] if cpu and gpu else None))
        speed_series[f"fold={fold}"] = pts
    render_line_chart(
        plot_dir / "e2e_speedup_rate1.svg",
        "E2E prove speedup, rate=1",
        [{"title": "CPU time / GPU Metal time", "series": speed_series, "colors": {f"fold={f}": FOLD_COLOR[f] for f in folds}}],
        "speedup",
        log_y=False,
        shared_y=False,
        panel_height=360,
    )

    phase_panels = []
    for phase in PHASES:
        phase_panels.append(
            {
                "title": f"{PHASE_LABEL[phase]}, fold=1, rate=1",
                "series": {
                    "CPU": series_backend("cpu", phase, 1, 1, lambda r: r["duration_ms"] if r else None),
                    "GPU Metal": series_backend("gpu-metal", phase, 1, 1, lambda r: r["duration_ms"] if r else None),
                },
                "colors": {"CPU": BACKEND_COLOR["cpu"], "GPU Metal": BACKEND_COLOR["gpu-metal"]},
            }
        )
    render_line_chart(
        plot_dir / "phase_runtime_fold1_rate1.svg",
        "Phase runtime, fold=1, rate=1",
        phase_panels,
        "milliseconds, log scale",
        log_y=True,
        shared_y=False,
    )

    mem_panels = []
    for fold in folds:
        mem_panels.append(
            {
                "title": f"fold={fold}, rate=1",
                "series": {
                    "CPU": series_backend("cpu", "e2e_prove", fold, 1, lambda r: gib(r["peak_allocated_bytes"]) if r else None),
                    "GPU Metal": series_backend("gpu-metal", "e2e_prove", fold, 1, lambda r: gib(r["peak_allocated_bytes"]) if r else None),
                },
                "colors": {"CPU": BACKEND_COLOR["cpu"], "GPU Metal": BACKEND_COLOR["gpu-metal"]},
            }
        )
    render_line_chart(
        plot_dir / "peak_memory_e2e_rate1.svg",
        "Peak allocated memory during E2E prove, rate=1",
        mem_panels,
        "GiB",
        log_y=False,
        shared_y=True,
    )

    max_mem = {"CPU": [], "GPU Metal": []}
    for log_size in logs:
        cpu_vals = [
            r["peak_allocated_bytes"]
            for fold in folds
            for r in [get("cpu", "e2e_prove", log_size, fold, 1)]
            if r
        ]
        gpu_vals = [
            r["peak_allocated_bytes"]
            for fold in folds
            for r in [get("gpu-metal", "e2e_prove", log_size, fold, 1)]
            if r
        ]
        max_mem["CPU"].append((log_size, gib(max(cpu_vals)) if cpu_vals else None))
        max_mem["GPU Metal"].append((log_size, gib(max(gpu_vals)) if gpu_vals else None))
    render_line_chart(
        plot_dir / "peak_memory_max_e2e_rate1.svg",
        "Max peak allocated memory by size, E2E prove rate=1",
        [{"title": "max across successful folds", "series": max_mem, "colors": {"CPU": BACKEND_COLOR["cpu"], "GPU Metal": BACKEND_COLOR["gpu-metal"]}}],
        "GiB",
        log_y=False,
        shared_y=False,
        panel_height=360,
    )

    profile = defaultdict(list)
    for log_size in logs:
        r = get("gpu-metal", "e2e_prove", log_size, 4, 1)
        profile["total"].append((log_size, r["duration_ms"] if r else None))
        profile["command wait"].append((log_size, r.get("metal_command_wait_ms", 0) if r else None))
        profile["upload"].append((log_size, r.get("metal_upload_ms", 0) if r else None))
        profile["readback"].append((log_size, r.get("metal_readback_ms", 0) if r else None))
        profile["blit"].append((log_size, r.get("metal_blit_wait_ms", 0) if r else None))
    render_line_chart(
        plot_dir / "gpu_profile_e2e_fold4_rate1.svg",
        "GPU E2E profile counters, fold=4, rate=1",
        [{"title": "wall time components", "series": dict(profile), "colors": PROFILE_COLOR}],
        "milliseconds, log scale",
        log_y=True,
        shared_y=False,
        panel_height=360,
    )

    speedups = []
    for fold in folds:
        for log_size in logs:
            cpu = get("cpu", "e2e_prove", log_size, fold, 1)
            gpu = get("gpu-metal", "e2e_prove", log_size, fold, 1)
            if cpu and gpu:
                speedups.append(cpu["duration_ms"] / gpu["duration_ms"])
    speedups_sorted = sorted(speedups)
    median_speedup = speedups_sorted[len(speedups_sorted) // 2] if speedups_sorted else None

    e2e_speed_table = []
    for log_size in logs:
        row = [f"2^{log_size}"]
        for fold in folds:
            if fold > log_size:
                row.append("n/a")
                continue
            cpu = get("cpu", "e2e_prove", log_size, fold, 1)
            gpu = get("gpu-metal", "e2e_prove", log_size, fold, 1)
            row.append(fmt_speedup(cpu["duration_ms"] / gpu["duration_ms"]) if cpu and gpu else "missing")
        e2e_speed_table.append(row)

    mem_table = []
    for log_size in logs:
        cpu_vals = [
            r["peak_allocated_bytes"]
            for fold in folds
            for r in [get("cpu", "e2e_prove", log_size, fold, 1)]
            if r
        ]
        gpu_vals = [
            r["peak_allocated_bytes"]
            for fold in folds
            for r in [get("gpu-metal", "e2e_prove", log_size, fold, 1)]
            if r
        ]
        c = max(cpu_vals) if cpu_vals else None
        g = max(gpu_vals) if gpu_vals else None
        mem_table.append([f"2^{log_size}", fmt_gib(c), fmt_gib(g), f"{c / g:.2f}x" if c and g else "OOM"])

    phase_table = []
    phase_sample_logs = [x for x in [8, 16, 20, 24, 27] if x in logs]
    for log_size in phase_sample_logs:
        for phase in PHASES:
            cpu = get("cpu", phase, log_size, 4, 1)
            gpu = get("gpu-metal", phase, log_size, 4, 1)
            phase_table.append(
                [
                    f"2^{log_size}",
                    PHASE_LABEL[phase],
                    fmt_ms(cpu["duration_ms"] if cpu else None),
                    fmt_ms(gpu["duration_ms"] if gpu else None),
                    fmt_speedup(cpu["duration_ms"] / gpu["duration_ms"] if cpu and gpu else None),
                ]
            )

    profile_table = []
    for log_size in phase_sample_logs:
        r = get("gpu-metal", "e2e_prove", log_size, 4, 1)
        profile_table.append(
            [
                f"2^{log_size}",
                fmt_ms(r["duration_ms"] if r else None),
                fmt_ms(r.get("metal_command_wait_ms") if r else None),
                f"{gib(r.get('metal_upload_bytes', 0)):.2f} GiB / {r.get('metal_upload_ms', 0):.1f} ms" if r else "OOM",
                f"{mib(r.get('metal_readback_bytes', 0)):.2f} MiB / {r.get('metal_readback_ms', 0):.1f} ms" if r else "OOM",
                f"{gib(r.get('metal_blit_bytes', 0)):.2f} GiB / {r.get('metal_blit_wait_ms', 0):.1f} ms" if r else "OOM",
            ]
        )

    expected = {
        (r["log_size"], r["fold"], r["rate"])
        for r in rows
    }
    missing = []
    for log_size, fold, rate in sorted(expected):
        for phase in PHASES:
            if not get("cpu", phase, log_size, fold, rate):
                missing.append(f"CPU {PHASE_LABEL[phase]} 2^{log_size} fold={fold} rate={rate}")
            if not get("gpu-metal", phase, log_size, fold, rate):
                missing.append(f"GPU {PHASE_LABEL[phase]} 2^{log_size} fold={fold} rate={rate}")

    report = []
    report.append("# WHIR CPU vs Metal GPU phase benchmark")
    report.append("")
    requested_min = args.requested_min_log_size if args.requested_min_log_size is not None else min(logs)
    requested_max = args.requested_max_log_size if args.requested_max_log_size is not None else max(logs)
    report.append("Hardware: Apple M4 Max, 40-core GPU, 48 GiB unified memory.")
    report.append(f"Parameters: requested `log_size={requested_min}..{requested_max}`, actual rows `log_size={min(logs)}..{max(logs)}`, folds `1,2,3,4,6`, rates `1,2,3` where the article grid fits memory and `rate=1` for the largest sizes, phases `commit,sumcheck,e2e`, `pow_bits=20`, `security_level=128`.")
    if requested_min < min(logs):
        report.append(f"`2^{requested_min}` has no rows because the benchmark rejects the configured folds there (`fold` must be `<= n`).")
    report.append("")
    report.append(f"Raw rows: `{len(rows)}`. Paired CSV: `{args.paired_csv}`.")
    if missing:
        report.append("Missing/killed rows: " + "; ".join(missing) + ".")
    else:
        report.append("Missing/killed rows: none.")
    if speedups:
        report.append(f"E2E rate=1 paired speedup: min `{min(speedups):.2f}x`, median `{median_speedup:.2f}x`, max `{max(speedups):.2f}x`.")
    report.append("")
    report.append("## Charts")
    for name in [
        "e2e_runtime_rate1.svg",
        "e2e_speedup_rate1.svg",
        "phase_runtime_fold1_rate1.svg",
        "peak_memory_e2e_rate1.svg",
        "peak_memory_max_e2e_rate1.svg",
        "gpu_profile_e2e_fold4_rate1.svg",
    ]:
        report.append(f"- `{plot_dir / name}`")
    report.append("")
    report.append("## E2E speedup, rate=1")
    report.append(markdown_table(["size"] + [f"fold={f}" for f in folds], e2e_speed_table))
    report.append("")
    report.append("## Max peak allocated memory, E2E rate=1")
    report.append(markdown_table(["size", "CPU GiB", "GPU GiB", "CPU/GPU"], mem_table))
    report.append("")
    report.append("## Phase speedup, fold=4 rate=1")
    report.append(markdown_table(["size", "phase", "CPU", "GPU", "CPU/GPU"], phase_table))
    report.append("")
    report.append("## GPU transfer/profile counters, E2E fold=4 rate=1")
    report.append(markdown_table(["size", "total", "command wait", "upload", "readback", "blit"], profile_table))
    report.append("")
    report.append("Notes: standalone `sumcheck` measures the phase in isolation, so large rows include host-to-GPU upload cost. The E2E path keeps more data resident and is the better signal for real prover throughput. Peak memory is allocator peak from the benchmark process; Metal counters are the explicit upload/readback/blit instrumentation emitted by the benchmark.")
    Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
