"""Build the 2-page technical report PDF from docs/report/technical_report.md.

    /scratch/m25csa016/envs/denoise/bin/python scripts/build_report_pdf.py

1. Draws the report figure (docs/report/fig_tradeoff.png) from the fp32 evaluation results in runs/ and logs/:
   (a) PSNR vs DISTS on the proxy test set: the fidelity(2a) <-> lambda40 weight blends of OmniDN-S and OmniDN-M
       (runs/interp_{s,m}.csv), the direct lambda15 fine-tunes (fp32 organiser-style evaluation parsed from
       logs/dn-eval-p15-44158.out), the S lambda15 <-> lambda40 blends (runs/interp_s_p15_p40.csv; column `alpha` is the
       lambda40 weight) and the submitted blend 0.55*lambda15 + 0.45*lambda40 (SUBMITTED_BLEND);
   (b) mean PSNR per degradation family: input vs public SCUNet vs submitted OmniDN-S
       (runs/baselines/*_proxy_test*_v2.csv, runs/final_verify_proxy_test*_v2.csv). The script checks that the
       final_verify CSVs hold the submitted model (their means must equal the SUBMITTED_BLEND row of the blend CSV).
2. Renders the Markdown subset used by the report (title, ## headings, paragraphs, "- " bullets, pipe tables,
   **bold**, *italic*, `code`, ![caption](image)) to an A4 PDF with reportlab and prints the page count.

Only reads the repository; writes docs/report/fig_tradeoff.png and docs/report/technical_report.pdf.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT / "docs" / "report"
MD_PATH = REPORT_DIR / "technical_report.md"
PDF_PATH = REPORT_DIR / "technical_report.pdf"
FIG_PATH = REPORT_DIR / "fig_tradeoff.png"
RUNS = PROJECT / "runs"
P15_EVAL_LOG = PROJECT / "logs" / "dn-eval-p15-44158.out"   # fp32 organiser-style eval of the direct lambda15 runs
SUBMITTED_BLEND = 0.45                                        # submission = (1-b)*lambda15 + b*lambda40 (v3_final)

FONT_DIRS = {
    "sans": [("/usr/share/fonts/liberation-sans", "LiberationSans-{}.ttf", ["Regular", "Bold", "Italic", "BoldItalic"]),
             ("/usr/share/fonts/dejavu", "DejaVuSansCondensed{}.ttf", ["", "-Bold", "-Oblique", "-BoldOblique"])],
    "mono": [("/usr/share/fonts/liberation-mono", "LiberationMono-{}.ttf", ["Regular", "Bold", "Italic", "BoldItalic"]),
             ("/usr/share/fonts/dejavu", "DejaVuSansMono{}.ttf", ["", "-Bold", "-Oblique", "-BoldOblique"])],
}

# chart palette (validated categorical slots 1-3 + neutral chrome)
C_S, C_M, C_SC = "#2a78d6", "#eb6834", "#1baf7a"
C_INPUT, INK, INK2, GRID, AXIS = "#a9a7a0", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def find_fonts(kind):
    for d, pat, variants in FONT_DIRS[kind]:
        paths = [Path(d) / pat.format(v) for v in variants]
        if all(p.exists() for p in paths):
            return paths
    raise FileNotFoundError(f"no {kind} TTF font family found")


# ============================================================================================ figure
def load_p15_eval() -> dict:
    """{(model, set): {psnr, ssim, dists}} from the fp32 organiser-style evaluation log of the direct lambda15 runs."""
    import json

    out = {}
    if P15_EVAL_LOG.exists():
        for m in re.finditer(r"^P15 (s|m) (\S+) (\{.*\})\s*$", P15_EVAL_LOG.read_text(), flags=re.M):
            out[(m.group(1), m.group(2))] = json.loads(m.group(3))
    return out


def submitted_point() -> dict:
    """The submitted blend's proxy-test metrics, cross-checked against the submission-folder verification CSVs."""
    b = pd.read_csv(RUNS / "interp_s_p15_p40.csv")
    row = b[(b.set == "proxy_test_v2") & (b.alpha == SUBMITTED_BLEND)].iloc[0]
    fv = pd.read_csv(RUNS / "final_verify_proxy_test_v2.csv")[["psnr", "ssim", "dists"]].mean()
    for k, tol in (("psnr", 1e-3), ("ssim", 1e-4), ("dists", 1e-4)):
        if abs(fv[k] - row[k]) > tol:
            raise SystemExit(f"runs/final_verify_proxy_test_v2.csv ({k} {fv[k]:.4f}) does not hold the submitted blend "
                             f"({row[k]:.4f}); re-run scripts/verify_submission.py before building the report")
    return dict(psnr=float(row.psnr), ssim=float(row.ssim), dists=float(row.dists))


def make_figure(out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for p in find_fonts("sans"):
        font_manager.fontManager.addfont(str(p))
    fam = font_manager.FontProperties(fname=str(find_fonts("sans")[0])).get_name()
    plt.rcParams.update({
        "font.family": fam, "font.size": 7.2, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2,
        "ytick.major.size": 2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlesize": 7.6,
        "axes.titlecolor": INK, "legend.fontsize": 6.7, "legend.frameon": False, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.32, 1.6), gridspec_kw=dict(width_ratios=[1, 1.15]))

    # ---- (a) weight blends: PSNR vs DISTS on the proxy test set (fp32 organiser-style evaluation throughout)
    p15 = load_p15_eval()
    for tag, col, gf in (("s", C_S, 2.96), ("m", C_M, 11.36)):
        d = pd.read_csv(RUNS / f"interp_{tag}.csv")
        d = d[d.set == "proxy_test_v2"].sort_values("alpha")
        ax1.plot(d.dists, d.psnr, "-o", color=col, lw=1.5, ms=4.2, mec="white", mew=0.7, zorder=3,
                 label=f"OmniDN-{tag.upper()}: 2a ↔ λ40 ({gf:.2f} G)")
        for a, ha, dx in ((0.0, "left", 0.0012), (1.0, "right", -0.0012)):
            r = d[d.alpha == a].iloc[0]
            dy, va = (0.045, "bottom") if tag == "s" else (-0.045, "top")  # M labels below: legend sits above
            ax1.annotate(f"α={a:g}", (r.dists, r.psnr), xytext=(r.dists + dx, r.psnr + dy), fontsize=6.6,
                         color=INK2, ha=ha, va=va)
        if (tag, "proxy_test_v2") in p15:  # direct lambda15 fine-tune (same init as the lambda40 run)
            r = p15[(tag, "proxy_test_v2")]
            ax1.plot(r["dists"], r["psnr"], "D", ms=4.6, mfc="white", mec=col, mew=1.1, zorder=5)
    if p15:  # neutral legend proxy: the colour already says which model
        ax1.plot([], [], "D", ms=4.6, mfc="white", mec=INK2, mew=1.1, ls="none", label="direct λ=15")
    # S blends lambda15 <-> lambda40: measured points (solid) and the unmeasured stretch to the lambda40 end (dotted)
    sub = submitted_point()
    b = pd.read_csv(RUNS / "interp_s_p15_p40.csv")
    b = b[b.set == "proxy_test_v2"].sort_values("alpha")
    s40 = pd.read_csv(RUNS / "interp_s.csv").query("set == 'proxy_test_v2' and alpha == 1.0").iloc[0]
    xs = [p15[("s", "proxy_test_v2")]["dists"]] if ("s", "proxy_test_v2") in p15 else []
    ys = [p15[("s", "proxy_test_v2")]["psnr"]] if ("s", "proxy_test_v2") in p15 else []
    xs, ys = xs + b.dists.tolist(), ys + b.psnr.tolist()
    ax1.plot(xs, ys, "--", color=INK, lw=0.8, zorder=4)
    ax1.plot(xs[1:], ys[1:], "x", color=INK, ms=3.2, mew=0.8, zorder=4)
    ax1.plot([], [], "--x", color=INK, lw=0.8, ms=3.2, mew=0.8, label="S: λ15 ↔ λ40")
    ax1.plot([xs[-1], s40.dists], [ys[-1], s40.psnr], ":", color=INK, lw=0.8, zorder=4)
    ax1.plot(sub["dists"], sub["psnr"], "o", ms=10, mfc="none", mec=INK, mew=0.9, zorder=4)
    ax1.annotate(f"submitted: 0.55·λ15 + 0.45·λ40\n{sub['psnr']:.2f} dB / {sub['dists']:.4f}",
                 (sub["dists"], sub["psnr"]), xytext=(sub["dists"] + 0.0085, 33.74), fontsize=6.6,
                 color=INK, ha="left", va="center", linespacing=1.1,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=5))
    ax1.set_xlabel("DISTS (lower is better)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("(a) Weight blends, proxy test set", loc="left")
    ax1.set_xlim(0.094, 0.153)
    ax1.set_ylim(33.62, 34.98)
    ax1.legend(loc="upper left", ncol=2, handlelength=1.6, borderaxespad=0.1, labelspacing=0.2, columnspacing=0.8,
               handletextpad=0.4)  # two rows above the M curve

    # ---- (b) PSNR per degradation family: input / SCUNet / OmniDN-S (submitted)
    def fam_means(prefix_main, prefix_ood):
        d = pd.read_csv(prefix_main)
        m = d.groupby("family").psnr.mean()
        m["ood"] = pd.read_csv(prefix_ood).psnr.mean()
        return m

    series = [
        ("Input", C_INPUT, fam_means(RUNS / "baselines/identity_proxy_test_v2.csv",
                                                      RUNS / "baselines/identity_proxy_test_ood_v2.csv")),
        ("SCUNet, public (79 G)", C_SC, fam_means(RUNS / "baselines/scunet_proxy_test_v2.csv",
                                                       RUNS / "baselines/scunet_proxy_test_ood_v2.csv")),
        ("OmniDN-S, submitted (2.96 G)", C_S, fam_means(RUNS / "final_verify_proxy_test_v2.csv",
                                                             RUNS / "final_verify_proxy_test_ood_v2.csv")),
    ]
    n_fam = pd.read_csv(RUNS / "final_verify_proxy_test_v2.csv").family.value_counts()
    n_fam["ood"] = len(pd.read_csv(RUNS / "final_verify_proxy_test_ood_v2.csv"))
    fams = [("noise", "Noise"), ("blur", "Blur"), ("jpeg", "JPEG"), ("chromatic", "Chromatic"), ("mixed", "Mixtures"),
            ("ood", "Held-out corr.")]
    offs = (-0.24, 0.0, 0.24)
    for (name, col, m), dx in zip(series, offs):
        xs = [i + dx for i in range(len(fams))]
        ys = [m[f] for f, _ in fams]
        ax2.plot(xs, ys, "o", ms=5, color=col, mec="white", mew=0.7, ls="none", zorder=3, label=name)
        if col != C_INPUT:  # direct value labels (relief for the low-contrast aqua slot): SCUNet below, ours above
            below = col == C_SC
            for x, y in zip(xs, ys):
                ax2.text(x, y - 1.0 if below else y + 0.9, f"{y:.1f}", fontsize=6.5, color=INK2, ha="center",
                         va="top" if below else "bottom")
    for i in range(len(fams)):
        if i:
            ax2.axvline(i - 0.5, color=GRID, lw=0.5, zorder=0)
    ax2.set_xticks(range(len(fams)))
    ax2.set_xticklabels([f"{lab}\n(n={n_fam[f]})" for f, lab in fams], fontsize=6.8, linespacing=1.05)
    ax2.tick_params(axis="x", length=0)
    ax2.set_xlim(-0.55, len(fams) - 0.45)
    ax2.set_ylim(18, 57)
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("(b) PSNR by degradation family", loc="left")
    ax2.legend(loc="upper left", handletextpad=0.1, borderaxespad=0.1, ncol=3, columnspacing=0.9)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=1.6)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


# ============================================================================================ markdown -> flowables
def inline(text: str, mono_font: str) -> str:
    """Markdown inline subset -> reportlab mini-HTML."""
    codes = []

    def _code(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _code, text)
    text = text.replace("\\*", "\x01")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"(?<=\w)\^(γ|\d+)", r"<super>\1</super>", text)              # x^γ
    text = text.replace("\x01", "*")
    for i, c in enumerate(codes):
        c = c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00{i}\x00", f'<font name="{mono_font}" size="-0.6">{c}</font>')
    return text


def parse_blocks(md: str):
    """-> list of (kind, payload): title, subtitle, h2, para, bullets, table, image."""
    lines = md.splitlines()
    blocks, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("# "):
            blocks.append(("title", s[2:].strip()))
            i += 1
            continue
        if s.startswith("## "):
            blocks.append(("h2", s[3:].strip()))
            i += 1
            continue
        m = re.match(r"!\[(.*)\]\((.+?)\)\s*$", s)
        if m:
            blocks.append(("image", (m.group(1), m.group(2))))
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in re.split(r"(?<!\\)\|", r.strip("|"))] for r in rows]
            cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            blocks.append(("table", cells))
            continue
        if s.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip():
                t = lines[i].strip()
                if t.startswith("- "):
                    items.append(t[2:])
                else:
                    items[-1] += " " + t
                i += 1
            blocks.append(("bullets", items))
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"\s*(#|\||- |!\[)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        text = " ".join(para)
        kind = "subtitle" if blocks and blocks[-1][0] == "title" else "para"
        blocks.append((kind, text))
    return blocks


def optimise_widths(paras, mins, maxs, avail, step=2.0, hpad=6.0):
    """Greedy column widths that minimise the table height: start every column at its longest unbreakable word, then
    repeatedly widen the column whose smallest useful widening saves the most height per point."""
    ncol, nrow = len(mins), len(paras)
    scale = min(1.0, avail / sum(mins))
    widths = [m * scale for m in mins]
    cache = {}

    def col_h(j, wj):
        key = (j, round(wj, 1))
        if key not in cache:
            cache[key] = [paras[r][j].wrap(max(wj - hpad, 1), 1e6)[1] for r in range(nrow)]
        return cache[key]

    def total(ws):
        hs = [col_h(j, ws[j]) for j in range(ncol)]
        return sum(max(hs[j][r] for j in range(ncol)) for r in range(nrow))

    while avail - sum(widths) >= step:
        base, left, best = total(widths), avail - sum(widths), None
        for j in range(ncol):
            if widths[j] >= maxs[j]:
                continue
            d = step
            while d <= min(left, maxs[j] - widths[j] + step):
                trial = widths[:]
                trial[j] += d
                gain = base - total(trial)
                if gain > 0.01:
                    if best is None or gain / d > best[0]:
                        best = (gain / d, j, d)
                    break
                d += step
        if best is None:  # no widening shortens the table any more: share the rest by remaining natural width
            flex = [max(mx - w, 0.0) for w, mx in zip(widths, maxs)]
            tot = sum(flex) or float(ncol)
            widths = [w + left * (f / tot if sum(flex) else 1.0 / ncol) for w, f in zip(widths, flex)]
            break
        widths[best[1]] += best[2]
    left = avail - sum(widths)
    if left > 0:
        widths[-1] += left
    return widths


def build_pdf(md_path: Path, pdf_path: Path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    # ---- fonts (TTF with Greek / math glyphs)
    sans, mono = find_fonts("sans"), find_fonts("mono")
    names = ["Body", "Body-Bold", "Body-Italic", "Body-BoldItalic"]
    for n, p in zip(names, sans):
        pdfmetrics.registerFont(TTFont(n, str(p)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                  boldItalic="Body-BoldItalic")
    for n, p in zip(["Mono", "Mono-Bold", "Mono-Italic", "Mono-BoldItalic"], mono):
        pdfmetrics.registerFont(TTFont(n, str(p)))
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono-Italic",
                                  boldItalic="Mono-BoldItalic")

    md = md_path.read_text(encoding="utf-8")
    face = pdfmetrics.getFont("Body").face
    missing = sorted({ch for ch in md if ord(ch) > 126 and ord(ch) not in face.charToGlyph})
    if missing:
        print("WARNING: glyphs missing from body font:", " ".join(f"{c}(U+{ord(c):04X})" for c in missing))

    accent = colors.HexColor("#1c5cab")
    ink, ink2 = colors.HexColor("#111111"), colors.HexColor("#4a4946")
    rule, shade, hilite = colors.HexColor("#9c9a93"), colors.HexColor("#f3f2ee"), colors.HexColor("#e3eefb")

    BODY = 9.0
    LEAD = 10.75
    st = {
        "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=13.6, leading=15.8, textColor=ink,
                                spaceAfter=1.5),
        "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=8.3, leading=10.0, textColor=ink2,
                                   spaceAfter=3),
        "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=10.2, leading=11.8, textColor=accent,
                             spaceBefore=3.6, spaceAfter=1.0),  # orphan control: CondPageBreak before each heading
        "para": ParagraphStyle("para", fontName="Body", fontSize=BODY, leading=LEAD, textColor=ink,
                               alignment=TA_LEFT, spaceAfter=1.7, allowWidows=0),
        "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=BODY, leading=LEAD, textColor=ink,
                                 leftIndent=10, bulletIndent=1.5, spaceAfter=1.0),
        "caption": ParagraphStyle("caption", fontName="Body", fontSize=7.9, leading=9.3, textColor=ink2,
                                  spaceBefore=1.5, spaceAfter=2.2),
        "figcap": ParagraphStyle("figcap", fontName="Body", fontSize=7.9, leading=9.3, textColor=ink2,
                                 spaceBefore=1, spaceAfter=3.5),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=7.5, leading=8.7, textColor=ink),
        "cellh": ParagraphStyle("cellh", fontName="Body-Bold", fontSize=7.5, leading=8.7, textColor=ink),
        "small": ParagraphStyle("small", fontName="Body-Italic", fontSize=7.4, leading=8.8, textColor=ink2,
                                spaceBefore=2),
    }

    page_w, page_h = A4
    margin = 1.2 * cm
    frame_w = page_w - 2 * margin
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=1.0 * cm,
                            bottomMargin=1.2 * cm, title="OmniDN-S technical report", author="m25csa016 (IIT Jodhpur)",
                            subject="Mini Competition: A Small Generative Model for Image Denoising")

    def on_page(canvas, d):
        canvas.saveState()
        canvas.setFont("Body", 7.2)
        canvas.setFillColor(ink2)
        canvas.drawString(margin, 0.65 * cm, "OmniDN-S — technical report — m25csa016, IIT Jodhpur")
        canvas.drawRightString(page_w - margin, 0.65 * cm, f"page {d.page}")
        canvas.restoreState()

    def table_flowable(rows):
        cs, chs = st["cell"], st["cellh"]
        ncol = len(rows[0])
        paras = [[Paragraph(inline(c, "Mono"), chs if r == 0 else cs) for c in row] for r, row in enumerate(rows)]
        # column widths: start from the longest unbreakable word, distribute the rest by natural text width
        def w(txt, font):
            plain = re.sub(r"<[^>]+>", "", inline(txt, "Mono")).replace("&amp;", "&").replace("&lt;", "<")
            return plain, pdfmetrics.stringWidth(plain, font, cs.fontSize)

        pad = 6.5
        mins, maxs = [0.0] * ncol, [0.0] * ncol
        for r, row in enumerate(rows):
            for j, c in enumerate(row):
                font = "Body-Bold" if (r == 0 or c.startswith("**")) else "Body"
                plain, full = w(c, font)
                longest = max((pdfmetrics.stringWidth(t, font, cs.fontSize) for t in plain.split()), default=0)
                mins[j] = max(mins[j], longest + pad)
                maxs[j] = max(maxs[j], full + pad)
        avail = frame_w
        if sum(maxs) <= avail:
            widths = [m * avail / sum(maxs) for m in maxs]
        else:
            widths = optimise_widths(paras, mins, maxs, avail)
        t = Table(paras, colWidths=widths, repeatRows=1, hAlign="LEFT")
        style = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.9),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, rule), ("LINEBELOW", (0, 0), (-1, 0), 0.5, rule),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, rule), ("BACKGROUND", (0, 0), (-1, 0), shade),
        ]
        for r in range(1, len(rows)):
            if r > 1:
                style.append(("LINEABOVE", (0, r), (-1, r), 0.25, colors.HexColor("#dcdad3")))
            if any(c.startswith("**") for c in rows[r]):
                style.append(("BACKGROUND", (0, r), (-1, r), hilite))
        t.setStyle(TableStyle(style))
        return t

    story, pending_caption = [], None
    for kind, payload in parse_blocks(md):
        if kind == "title":
            story.append(Paragraph(inline(payload, "Mono"), st["title"]))
        elif kind == "subtitle":
            story.append(Paragraph(inline(payload, "Mono"), st["subtitle"]))
        elif kind == "h2":
            story.append(CondPageBreak(1.6 * cm))
            story.append(Paragraph(inline(payload, "Mono"), st["h2"]))
        elif kind == "para":
            if re.match(r"\*\*Table \d+\.\*\*", payload):
                pending_caption = Paragraph(inline(payload, "Mono"), st["caption"])
                continue
            style = st["small"] if payload.startswith("*") and not payload.startswith("**") else st["para"]
            text = payload[1:-1] if style is st["small"] else payload
            story.append(Paragraph(inline(text, "Mono"), style))
        elif kind == "bullets":
            for it in payload:
                story.append(Paragraph(inline(it, "Mono"), st["bullet"], bulletText="•"))
            story.append(Spacer(1, 1.6))
        elif kind == "table":
            # the table may split across pages (header row repeats); a conditional page break keeps the caption (and a
            # directly preceding heading) together with the header and the first two body rows
            t = table_flowable(payload)
            t.wrap(frame_w, page_h)
            need = sum(t._rowHeights[:3]) + 6
            if pending_caption is not None:
                need += pending_caption.wrap(frame_w, page_h)[1] + 4
            at = len(story)
            if story and isinstance(story[-1], Paragraph) and story[-1].style.name == "h2":
                at -= 1
                need += story[-1].wrap(frame_w, page_h)[1] + 8
                if at and isinstance(story[at - 1], CondPageBreak):
                    at -= 1
            story.insert(at, CondPageBreak(need))
            if pending_caption is not None:
                story.append(pending_caption)
            story.append(t)
            story.append(Spacer(1, 3))
            pending_caption = None
        elif kind == "image":
            cap, rel = payload
            path = (md_path.parent / rel).resolve()
            from reportlab.lib.utils import ImageReader

            iw, ih = ImageReader(str(path)).getSize()
            width = frame_w
            img = Image(str(path), width=width, height=width * ih / iw)
            story.append(KeepTogether([Spacer(1, 1), img, Paragraph(inline(cap, "Mono"), st["figcap"])]))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return doc.page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(MD_PATH))
    ap.add_argument("--pdf", default=str(PDF_PATH))
    ap.add_argument("--no-figure", action="store_true", help="reuse the existing figure PNG")
    args = ap.parse_args()
    if not args.no_figure:
        make_figure(FIG_PATH)
        print(f"figure: {FIG_PATH}")
    n = build_pdf(Path(args.md), Path(args.pdf))
    words = len(re.findall(r"[A-Za-z0-9α-ω][\w.,/%-]*", Path(args.md).read_text(encoding="utf-8")))
    print(f"pdf: {args.pdf}")
    print(f"pages: {n}")
    print(f"words (markdown source, approx.): {words}")


if __name__ == "__main__":
    main()
