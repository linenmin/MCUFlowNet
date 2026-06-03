import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)


def collect_image_refs(doc_path: Path) -> list[Path]:
    text = doc_path.read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    return [Path(ref) for ref in refs]


def parse_markdown_sections(doc_path: Path) -> list[dict]:
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    sections: list[dict] = []
    current = None

    for line in lines:
        if line.startswith("## "):
            if current:
                sections.append(current)
            current = {"title": line[3:].strip(), "content": []}
        elif current is not None:
            current["content"].append(line)

    if current:
        sections.append(current)

    return sections


def build_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#17324d"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "CustomSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#4c6272"),
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "h2": ParagraphStyle(
            "CustomH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#17324d"),
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h3": ParagraphStyle(
            "CustomH3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#234e70"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "CustomBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "CustomCaption",
            parent=styles["Italic"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#52606d"),
            spaceBefore=4,
            spaceAfter=10,
        ),
        "small": ParagraphStyle(
            "CustomSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#52606d"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
    }


def system_diagram_layout() -> dict:
    nodes = {
        "SP": {"x": 0.5, "y": 4.25, "w": 1.65, "h": 0.82, "fill": "#e6f0ff", "label": "9D Search Space"},
        "A": {"x": 2.55, "y": 4.25, "w": 1.95, "h": 0.82, "fill": "#d6f5f3", "label": "Agent A\nStrategist"},
        "B": {"x": 4.9, "y": 4.25, "w": 1.95, "h": 0.82, "fill": "#d6f5f3", "label": "Agent B\nGenerator"},
        "E": {"x": 7.25, "y": 4.25, "w": 1.95, "h": 0.82, "fill": "#fff0d9", "label": "Coordinator\nEngine"},
        "S": {"x": 9.35, "y": 2.65, "w": 1.6, "h": 0.9, "fill": "#ffe7e7", "label": "Supernet\nand Vela"},
        "C": {"x": 7.05, "y": 2.65, "w": 1.85, "h": 0.9, "fill": "#efe2ff", "label": "Agent C\nHW Distiller"},
        "H": {"x": 4.75, "y": 2.65, "w": 1.85, "h": 0.9, "fill": "#ebf7e8", "label": "History\nArchive"},
        "D": {"x": 2.45, "y": 2.65, "w": 1.85, "h": 0.9, "fill": "#efe2ff", "label": "Agent D\nScientist"},
        "AF": {"x": 0.35, "y": 0.95, "w": 2.2, "h": 0.92, "fill": "#fde68a", "label": "Assumptions\nand Findings"},
    }
    arrows = [
        {"src": "SP", "dst": "A", "label": "search prior", "mode": "straight"},
        {"src": "A", "dst": "B", "label": "strategy", "mode": "straight"},
        {"src": "B", "dst": "E", "label": "candidate batch", "mode": "straight"},
        {"src": "E", "dst": "S", "label": "evaluate", "mode": "evaluate_path"},
        {"src": "S", "dst": "C", "label": "HW report", "mode": "straight"},
        {"src": "C", "dst": "H", "label": "distilled insight", "mode": "straight"},
        {"src": "H", "dst": "D", "label": "history", "mode": "straight"},
        {"src": "D", "dst": "AF", "label": "hypotheses", "mode": "vertical_left"},
        {"src": "AF", "dst": "A", "label": "validated rules", "mode": "feedback_up"},
        {"src": "AF", "dst": "E", "label": "constraints", "mode": "feedback_across"},
    ]
    return {"nodes": nodes, "arrows": arrows}


def _node_point(node: dict, side: str) -> tuple[float, float]:
    x, y, w, h = node["x"], node["y"], node["w"], node["h"]
    points = {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
    }
    return points[side]


def _draw_labeled_arrow(ax, start, end, label, connectionstyle="arc3,rad=0.0", label_pos=None):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.6,
        color="#486581",
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    if label_pos is None:
        label_pos = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12)
    ax.text(
        label_pos[0],
        label_pos[1],
        label,
        fontsize=8.6,
        color="#334e68",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", fc="#f8fafc", ec="none", alpha=0.95),
    )


def _draw_polyline_arrow(ax, points, label=None, label_pos=None, linestyle="-", color="#486581"):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=1.6,
            linestyle=linestyle,
            solid_capstyle="round",
        )
    head = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.6,
        color=color,
        linestyle=linestyle,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(head)
    if label:
        if label_pos is None:
            start, end = points[0], points[-1]
            label_pos = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            fontsize=8.6,
            color="#334e68",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="#f8fafc", ec="none", alpha=0.95),
        )


def draw_system_diagram(target_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0.45, 6.15)
    ax.axis("off")
    fig.patch.set_facecolor("#f7fafc")
    ax.set_facecolor("#f7fafc")
    layout = system_diagram_layout()
    nodes = layout["nodes"]

    ax.text(5.7, 5.72, "EdgeFlowNAS Agentic Search Loop", ha="center", va="center", fontsize=18, color="#102a43", weight="bold")
    ax.text(5.7, 5.38, "A two-lane view: proposal and evaluation above, scientific memory and feedback below", ha="center", va="center", fontsize=10.5, color="#486581")

    for node in nodes.values():
        x, y, w, h, fill, label = node["x"], node["y"], node["w"], node["h"], node["fill"], node["label"]
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5,
            edgecolor="#243b53",
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, color="#102a43", weight="bold")

    for arrow in layout["arrows"]:
        src = nodes[arrow["src"]]
        dst = nodes[arrow["dst"]]
        mode = arrow["mode"]
        label = arrow["label"]

        if mode == "straight":
            if src["x"] <= dst["x"]:
                start = _node_point(src, "right")
                end = _node_point(dst, "left")
            else:
                start = _node_point(src, "left")
                end = _node_point(dst, "right")
            label_pos = ((start[0] + end[0]) / 2, start[1] + 0.22)
            _draw_labeled_arrow(ax, start, end, label, label_pos=label_pos)
        elif mode == "evaluate_path":
            start = _node_point(src, "bottom")
            end = _node_point(dst, "top")
            bend_y = 3.82
            points = [
                start,
                (start[0], bend_y),
                (end[0], bend_y),
                end,
            ]
            _draw_polyline_arrow(ax, points, label=label, label_pos=(8.75, 3.63))
        elif mode == "vertical_left":
            start = _node_point(src, "bottom")
            end = _node_point(dst, "top")
            bend_y = dst["y"] + dst["h"] + 0.32
            points = [
                start,
                (start[0], bend_y),
                (_node_point(dst, "top")[0], bend_y),
                end,
            ]
            _draw_polyline_arrow(ax, points, label=label, label_pos=(1.55, 2.22))
        elif mode == "feedback_up":
            start = (_node_point(src, "top")[0], src["y"] + src["h"])
            end = _node_point(dst, "bottom")
            mid_y = 3.95
            points = [
                start,
                (start[0], mid_y),
                (end[0], mid_y),
                end,
            ]
            _draw_polyline_arrow(ax, points, label=label, label_pos=(2.3, 4.0))
        elif mode == "feedback_across":
            start = _node_point(src, "right")
            end = _node_point(dst, "bottom")
            lane_y = 0.76
            points = [
                start,
                (start[0] + 0.25, lane_y),
                (end[0], lane_y),
                end,
            ]
            _draw_polyline_arrow(
                ax,
                points,
                label=label,
                label_pos=(5.55, 0.7),
                linestyle="--",
                color="#627d98",
            )

    fig.savefig(target_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return target_path


def _image_flowable(image_path: Path, max_width: float, max_height: float):
    from PIL import Image as PILImage

    with PILImage.open(image_path) as image:
        width_px, height_px = image.size

    scale = min(max_width / width_px, max_height / height_px)
    return Image(str(image_path), width=width_px * scale, height=height_px * scale)


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#52606d"))
    canvas.drawString(doc.leftMargin, A4[1] - 12 * mm, "EdgeFlowNAS Agentic Search Overview")
    canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def export_pdf(doc_path: Path, output_pdf: Path) -> Path:
    styles = build_styles()
    sections = parse_markdown_sections(doc_path)
    image_refs = collect_image_refs(doc_path)
    diagram_path = doc_path.parent / "08_System_Overview_Diagram_flow.png"
    draw_system_diagram(diagram_path)

    report = BaseDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="EdgeFlowNAS Agentic Search Overview",
        author="OpenAI Codex",
    )
    frame = Frame(report.leftMargin, report.bottomMargin, report.width, report.height, id="normal")
    report.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_header_footer)])

    story = []
    title = doc_path.read_text(encoding="utf-8").splitlines()[0].replace("# ", "").strip()
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(title, styles["title"]))
    story.append(
        Paragraph(
            "A concise system brief for advisor review, including the current Search_v5 visual evidence.",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(_image_flowable(diagram_path, max_width=160 * mm, max_height=78 * mm))
    story.append(Paragraph("Figure A. Static overview of the multi-agent search loop.", styles["caption"]))

    image_queue = list(image_refs)
    body_text = doc_path.read_text(encoding="utf-8")
    lines = body_text.splitlines()
    in_mermaid = False

    bullet_buffer = []
    numbered_buffer = []

    def flush_buffers():
        nonlocal bullet_buffer, numbered_buffer
        if bullet_buffer:
            items = [ListItem(Paragraph(item, styles["body"])) for item in bullet_buffer]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=14))
            story.append(Spacer(1, 2))
            bullet_buffer = []
        if numbered_buffer:
            items = [ListItem(Paragraph(item, styles["body"])) for item in numbered_buffer]
            story.append(ListFlowable(items, bulletType="1", leftIndent=14))
            story.append(Spacer(1, 2))
            numbered_buffer = []

    for line in lines[1:]:
        stripped = line.strip()

        if stripped.startswith("```mermaid"):
            in_mermaid = True
            continue
        if in_mermaid:
            if stripped == "```":
                in_mermaid = False
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_buffers()
            image_path = Path(image_match.group(2))
            caption = image_match.group(1) or image_path.name
            story.append(Spacer(1, 3 * mm))
            story.append(_image_flowable(image_path, max_width=165 * mm, max_height=210 * mm))
            story.append(Paragraph(caption, styles["caption"]))
            if "Figure 1" in caption or "Figure 2 GIF" in caption or "Figure 4" in caption:
                story.append(PageBreak())
            continue

        if stripped.startswith("# "):
            continue
        if stripped.startswith("## "):
            flush_buffers()
            story.append(Paragraph(stripped[3:], styles["h2"]))
            continue
        if stripped.startswith("### "):
            flush_buffers()
            story.append(Paragraph(stripped[4:], styles["h3"]))
            continue
        if re.match(r"^\d+\.\s+", stripped):
            numbered_buffer.append(re.sub(r"^\d+\.\s+", "", stripped))
            continue
        if stripped.startswith("- "):
            bullet_buffer.append(stripped[2:])
            continue
        if not stripped:
            flush_buffers()
            story.append(Spacer(1, 2))
            continue

        flush_buffers()
        text = re.sub(r"`([^`]+)`", r"<font name='Helvetica-Bold'>\1</font>", stripped)
        story.append(Paragraph(text, styles["body"]))

    flush_buffers()
    report.build(story)
    return output_pdf


def main():
    repo_root = Path(__file__).resolve().parents[1]
    doc_path = repo_root / "plan" / "search_v1" / "08_System_Overview_Diagram.md"
    output_pdf = repo_root / "plan" / "search_v1" / "08_System_Overview_Diagram.pdf"
    export_pdf(doc_path, output_pdf)
    print(output_pdf)


if __name__ == "__main__":
    main()
