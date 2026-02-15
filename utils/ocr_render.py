"""
OCR Render — Library API for single-text markdown-to-image rendering.

Usage:
    from ocr_render import generate_image, generate_image_bytes, list_templates

    img = generate_image("# Hello World\nThis is **bold** markdown.")
    img.show()

    png_data = generate_image_bytes("# Title", template="dark", width=800)

    print(list_templates())  # ['clean_light', 'dark', 'sepia', 'blue_gray']
"""

from __future__ import annotations

import io
import random
import shutil
import subprocess

import markdown
from PIL import Image

# ─────────────────────────────────────────────
# HTML Templates (variety for dataset diversity)
# ─────────────────────────────────────────────

TEMPLATES = {
    "clean_light": {
        "bg": "#ffffff",
        "text": "#1a1a1a",
        "code_bg": "#f4f4f5",
        "code_text": "#e11d48",
        "font": "'Noto Sans', 'Noto Sans CJK KR', sans-serif",
        "heading_color": "#111827",
        "border_color": "#e5e7eb",
        "link_color": "#2563eb",
    },
    "dark": {
        "bg": "#1e1e2e",
        "text": "#cdd6f4",
        "code_bg": "#313244",
        "code_text": "#f38ba8",
        "font": "'Noto Sans', 'Noto Sans CJK KR', sans-serif",
        "heading_color": "#cba6f7",
        "border_color": "#45475a",
        "link_color": "#89b4fa",
    },
    "sepia": {
        "bg": "#faf4eb",
        "text": "#433422",
        "code_bg": "#f0e6d3",
        "code_text": "#9a3412",
        "font": "'Noto Serif', 'Noto Sans CJK KR', serif",
        "heading_color": "#78350f",
        "border_color": "#d6cbbf",
        "link_color": "#b45309",
    },
    "blue_gray": {
        "bg": "#f0f4f8",
        "text": "#334155",
        "code_bg": "#e2e8f0",
        "code_text": "#7c3aed",
        "font": "'Noto Sans', 'Noto Sans CJK KR', sans-serif",
        "heading_color": "#1e293b",
        "border_color": "#cbd5e1",
        "link_color": "#2563eb",
    },
}


def build_html(md_text: str, template_name: str = "clean_light", width: int = 512) -> str:
    """Convert markdown text to fully styled HTML string."""
    t = TEMPLATES[template_name]
    extensions = ["fenced_code", "tables", "nl2br", "sane_lists"]
    html_body = markdown.markdown(md_text.strip(), extensions=extensions)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: {t['font']};
    font-size: 15px;
    line-height: 1.7;
    color: {t['text']};
    background: {t['bg']};
    padding: 32px 36px;
    max-width: {width}px;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }}
  h1 {{ font-size: 42px; font-weight: 700; color: {t['heading_color']}; margin: 0 0 12px 0; line-height: 1.27; }}
  h2 {{ font-size: 20px; font-weight: 600; color: {t['heading_color']}; margin: 16px 0 8px 0; }}
  h3 {{ font-size: 17px; font-weight: 600; color: {t['heading_color']}; margin: 12px 0 6px 0; }}
  p {{ margin: 0 0 10px 0; }}
  strong {{ font-weight: 600; }}
  em {{ font-style: italic; }}
  a {{ color: {t['link_color']}; text-decoration: none; }}
  code {{
    background: {t['code_bg']};
    color: {t['code_text']};
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Noto Sans Mono', 'Courier New', monospace;
    font-size: 13px;
  }}
  pre {{
    background: {t['code_bg']};
    padding: 14px 16px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 10px 0;
  }}
  pre code {{
    background: none;
    padding: 0;
    font-size: 13px;
    line-height: 1.5;
  }}
  blockquote {{
    border-left: 3px solid {t['border_color']};
    padding: 4px 0 4px 16px;
    margin: 10px 0;
    color: {t['text']}99;
  }}
  ul, ol {{ padding-left: 24px; margin: 6px 0 10px 0; }}
  li {{ margin: 3px 0; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 14px;
  }}
  th, td {{
    border: 1px solid {t['border_color']};
    padding: 8px 12px;
    text-align: left;
  }}
  th {{ background: {t['code_bg']}; font-weight: 600; }}
  hr {{ border: none; border-top: 1px solid {t['border_color']}; margin: 16px 0; }}
  img {{ max-width: 100%; height: auto; border-radius: 4px; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""


def list_templates() -> list[str]:
    """Return available template names."""
    return list(TEMPLATES.keys())


def resolve_template(template: str) -> str:
    """Resolve a template name. ``"random"`` picks one at random.

    Raises:
        ValueError: If *template* is not ``"random"`` and not in TEMPLATES.
    """
    if template == "random":
        return random.choice(list(TEMPLATES.keys()))
    if template not in TEMPLATES:
        raise ValueError(
            f"Unknown template {template!r}. "
            f"Available: {list(TEMPLATES.keys())}"
        )
    return template


def _render_to_bytes(
    html: str,
    width: int,
    height: int,
    quality: int,
    timeout: int,
) -> bytes:
    """Render HTML to PNG bytes via wkhtmltoimage (stdin → stdout, no temp files).

    Raises:
        RuntimeError: If wkhtmltoimage is not installed or rendering fails.
    """
    if shutil.which("wkhtmltoimage") is None:
        raise RuntimeError(
            "wkhtmltoimage not found. Install it with: "
            "sudo apt-get install wkhtmltopdf"
        )

    cmd = [
        "wkhtmltoimage",
        "--quiet",
        "--encoding", "UTF-8",
        "--width", str(width),
        "--disable-smart-width",
        "--enable-local-file-access",
        "--format", "png",
    ]
    if height > 0:
        cmd += ["--height", str(height), "--crop-h", str(height)]
    if quality < 100:
        cmd += ["--quality", str(quality)]

    # stdin → stdout: "-" for both input and output
    cmd += ["-", "-"]

    try:
        proc = subprocess.run(
            cmd,
            input=html.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"wkhtmltoimage timed out after {timeout}s") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"wkhtmltoimage failed (exit {proc.returncode}): {stderr}"
        )

    if not proc.stdout:
        raise RuntimeError("wkhtmltoimage produced empty output")

    return proc.stdout


def generate_image_bytes(
    text: str,
    template: str = "random",
    width: int = 512,
    height: int = 0,
    quality: int = 90,
    timeout: int = 30,
) -> bytes:
    """Render markdown *text* and return raw PNG bytes.

    Args:
        text:     Markdown string to render.
        template: Template name or ``"random"``.
        width:    Image width in pixels.
        height:   Image height in pixels (0 = auto-fit content).
        quality:  Image quality (1-100).
        timeout:  Subprocess timeout in seconds.

    Returns:
        PNG image as ``bytes``.

    Raises:
        ValueError:  Empty text or unknown template.
        RuntimeError: wkhtmltoimage missing or rendering failure.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    template = resolve_template(template)
    html = build_html(text, template_name=template, width=width)
    return _render_to_bytes(html, width, height, quality, timeout)


def generate_image(
    text: str,
    template: str = "random",
    width: int = 512,
    height: int = 0,
    quality: int = 90,
    save_path: str | None = None,
    timeout: int = 30,
) -> Image.Image:
    """Render markdown *text* and return a PIL Image.

    Args:
        text:      Markdown string to render.
        template:  Template name or ``"random"``.
        width:     Image width in pixels.
        height:    Image height in pixels (0 = auto-fit content).
        quality:   Image quality (1-100).
        save_path: If given, also save the image to this path.
        timeout:   Subprocess timeout in seconds.

    Returns:
        ``PIL.Image.Image`` object (RGB PNG).

    Raises:
        ValueError:  Empty text or unknown template.
        RuntimeError: wkhtmltoimage missing or rendering failure.
    """
    png_bytes = generate_image_bytes(
        text,
        template=template,
        width=width,
        height=height,
        quality=quality,
        timeout=timeout,
    )
    img = Image.open(io.BytesIO(png_bytes))

    if save_path is not None:
        img.save(save_path)

    return img
