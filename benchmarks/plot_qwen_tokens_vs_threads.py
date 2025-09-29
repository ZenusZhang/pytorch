#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
import xml.etree.ElementTree as ET


TOKENS_PATTERN = re.compile(r"^\s*(Eager|TorchScript|torch\.compile) details.*tokens/s: ([0-9.]+)")
THREAD_PATTERN = re.compile(r"num_cores=(\d+)")


def parse_log(log_path: Path) -> tuple[int, dict[str, float]]:
    text = log_path.read_text()
    thread_match = THREAD_PATTERN.search(text)
    if not thread_match:
        raise ValueError(f"num_cores=... not found in {log_path}")
    thread = int(thread_match.group(1))
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = TOKENS_PATTERN.match(line)
        if match:
            label, value = match.groups()
            values[label] = float(value)
    if not values:
        raise ValueError(f"No tokens/s entries found in {log_path}")
    return thread, values


def build_svg(data: dict[int, dict[str, float]]) -> ET.Element:
    threads = sorted(data)
    labels = ["Eager", "TorchScript", "torch.compile"]
    colors = {
        "Eager": "#1f77b4",
        "TorchScript": "#ff7f0e",
        "torch.compile": "#2ca02c",
    }

    width, height = 720, 480
    margin_left, margin_right = 80, 40
    margin_top, margin_bottom = 60, 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_tokens = max(data[t][label] for t in threads for label in labels)
    min_tokens = 0.0

    svg = ET.Element("svg", attrib={
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })

    ET.SubElement(svg, "rect", attrib={
        "x": "0",
        "y": "0",
        "width": str(width),
        "height": str(height),
        "fill": "#ffffff",
    })

    ET.SubElement(svg, "text", attrib={
        "x": str(width / 2),
        "y": str(margin_top / 2),
        "text-anchor": "middle",
        "font-size": "20",
        "font-family": "sans-serif",
    }).text = "Qwen3-0.6B Tokens/sec vs Threads"

    plot_x0, plot_y0 = margin_left, margin_top
    plot_x1, plot_y1 = margin_left + plot_width, margin_top + plot_height

    ET.SubElement(svg, "line", attrib={
        "x1": str(plot_x0),
        "y1": str(plot_y1),
        "x2": str(plot_x1),
        "y2": str(plot_y1),
        "stroke": "#000000",
        "stroke-width": "2",
    })
    ET.SubElement(svg, "line", attrib={
        "x1": str(plot_x0),
        "y1": str(plot_y0),
        "x2": str(plot_x0),
        "y2": str(plot_y1),
        "stroke": "#000000",
        "stroke-width": "2",
    })

    ET.SubElement(svg, "text", attrib={
        "x": str((plot_x0 + plot_x1) / 2),
        "y": str(height - margin_bottom / 3),
        "text-anchor": "middle",
        "font-size": "16",
        "font-family": "sans-serif",
    }).text = "Threads"

    ET.SubElement(svg, "text", attrib={
        "transform": f"translate({margin_left / 3},{(plot_y0 + plot_y1) / 2}) rotate(-90)",
        "text-anchor": "middle",
        "font-size": "16",
        "font-family": "sans-serif",
    }).text = "Tokens / sec"

    num_y_ticks = 5
    for i in range(num_y_ticks + 1):
        frac = i / num_y_ticks
        value = min_tokens + frac * (max_tokens - min_tokens)
        y = plot_y1 - frac * plot_height
        ET.SubElement(svg, "line", attrib={
            "x1": str(plot_x0),
            "y1": str(y),
            "x2": str(plot_x1),
            "y2": str(y),
            "stroke": "#cccccc",
            "stroke-width": "1",
            "stroke-dasharray": "4 4",
        })
        ET.SubElement(svg, "text", attrib={
            "x": str(plot_x0 - 10),
            "y": str(y + 5),
            "text-anchor": "end",
            "font-size": "14",
            "font-family": "sans-serif",
        }).text = f"{value:.1f}"

    for thread in threads:
        if len(threads) == 1:
            x_frac = 0.0
        else:
            x_frac = (thread - threads[0]) / (threads[-1] - threads[0])
        x = plot_x0 + x_frac * plot_width
        ET.SubElement(svg, "line", attrib={
            "x1": str(x),
            "y1": str(plot_y1),
            "x2": str(x),
            "y2": str(plot_y1 + 6),
            "stroke": "#000000",
            "stroke-width": "1",
        })
        ET.SubElement(svg, "text", attrib={
            "x": str(x),
            "y": str(plot_y1 + 24),
            "text-anchor": "middle",
            "font-size": "14",
            "font-family": "sans-serif",
        }).text = str(thread)

    for label in labels:
        points = []
        for thread in threads:
            tokens = data[thread][label]
            if max_tokens == min_tokens:
                y_frac = 0.0
            else:
                y_frac = (tokens - min_tokens) / (max_tokens - min_tokens)
            if len(threads) == 1:
                x_frac = 0.0
            else:
                x_frac = (thread - threads[0]) / (threads[-1] - threads[0])
            x = plot_x0 + x_frac * plot_width
            y = plot_y1 - y_frac * plot_height
            points.append((x, y))
        point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        ET.SubElement(svg, "polyline", attrib={
            "points": point_string,
            "fill": "none",
            "stroke": colors[label],
            "stroke-width": "3",
        })
        for x, y in points:
            ET.SubElement(svg, "circle", attrib={
                "cx": f"{x:.2f}",
                "cy": f"{y:.2f}",
                "r": "5",
                "fill": colors[label],
            })

    legend_x = plot_x0 + plot_width * 0.65
    legend_y = margin_top + 10
    legend_spacing = 24
    for idx, label in enumerate(labels):
        y = legend_y + idx * legend_spacing
        ET.SubElement(svg, "rect", attrib={
            "x": str(legend_x),
            "y": str(y - 12),
            "width": "18",
            "height": "18",
            "fill": colors[label],
        })
        ET.SubElement(svg, "text", attrib={
            "x": str(legend_x + 24),
            "y": str(y + 2),
            "font-size": "14",
            "font-family": "sans-serif",
        }).text = label

    return svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tokens/sec vs threads from benchmark logs")
    parser.add_argument(
        "logs",
        nargs="+",
        help="Paths to benchmark log files",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/qwen_tokens_vs_threads.svg",
        help="Output SVG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data: dict[int, dict[str, float]] = {}
    for log in args.logs:
        thread, values = parse_log(Path(log))
        data[thread] = values
    svg_root = build_svg(data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg_root).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
