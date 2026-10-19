#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convertir_informe_a_pdf.py
===============================================================================
Convierte cualquier informe Markdown de la tarea a un PDF estatico autocontenido
con soporte KaTeX para formulas matematicas e incrustacion de imagenes Base64.

Uso:
  python3 scripts/convertir_informe_a_pdf.py
  python3 scripts/convertir_informe_a_pdf.py resultados/computador_1/INFORME_EXPERIMENTOS.md
  python3 scripts/convertir_informe_a_pdf.py resultados/computador_2/INFORME_EXPERIMENTOS_COMPUTADOR_2.md
===============================================================================
"""

import os
import sys
import re
import base64
import shutil
import subprocess
import markdown

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def encontrar_chrome():
    """Detecta la ruta del ejecutable de Google Chrome o Chromium en el sistema."""
    candidatos = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidatos:
        if c and os.path.exists(c):
            return c
    return None

def encode_image_base64(filepath):
    """Convierte una imagen local a Data URL Base64 para incrustarla inline."""
    if not os.path.exists(filepath):
        return ""
    ext = os.path.splitext(filepath)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{encoded}"

def convertir_md_a_pdf(md_path=None):
    if md_path is None:
        if len(sys.argv) > 1:
            md_path = sys.argv[1]
        else:
            # Buscar por defecto
            c2_md = os.path.join(REPO_ROOT, "resultados", "computador_2", "INFORME_EXPERIMENTOS_COMPUTADOR_2.md")
            c1_md = os.path.join(REPO_ROOT, "resultados", "computador_1", "INFORME_EXPERIMENTOS.md")
            if os.path.exists(c2_md):
                md_path = c2_md
            elif os.path.exists(c1_md):
                md_path = c1_md
            else:
                raise FileNotFoundError("No se encontro ningun archivo de informe Markdown para compilar.")

    md_path = os.path.abspath(md_path)
    work_dir = os.path.dirname(md_path)
    base_name = os.path.splitext(os.path.basename(md_path))[0]
    html_path = os.path.join(work_dir, f"{base_name}.html")
    pdf_path = os.path.join(work_dir, f"{base_name}.pdf")

    print(f"[i] Leyendo informe: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Buscar todas las imagenes referenciadas en markdown: ![alt](ruta)
    def replacer(match):
        alt = match.group(1)
        rel_path = match.group(2)
        # Resolver ruta relativa respecto a work_dir o REPO_ROOT
        full_img_path = os.path.normpath(os.path.join(work_dir, rel_path))
        if not os.path.exists(full_img_path):
            full_img_path = os.path.normpath(os.path.join(REPO_ROOT, rel_path))
        if os.path.exists(full_img_path):
            data_url = encode_image_base64(full_img_path)
            return f"![{alt}]({data_url})"
        return match.group(0)

    md_text_inlined = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replacer, md_text)

    # Conversion de Markdown a HTML
    html_body = markdown.markdown(
        md_text_inlined,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"]
    )

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Informe de Resultados - Tarea 2 IIC3533</title>
  
  <!-- KaTeX CSS y JS para renderizado vectorial de formulas -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
      delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}}
      ],
      throwOnError: false
    }});">
  </script>

  <style>
    @page {{
      size: letter;
      margin: 20mm 18mm 20mm 18mm;
      @bottom-right {{
        content: counter(page);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 9pt;
        color: #64748b;
      }}
      @top-left {{
        content: "IIC3533 Computación de Alto Rendimiento · Tarea 2";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 8.5pt;
        color: #94a3b8;
      }}
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 10.5pt;
      line-height: 1.55;
      color: #1e293b;
      background-color: #ffffff;
      margin: 0;
      padding: 0;
    }}

    h1 {{
      font-size: 20pt;
      color: #0f172a;
      border-bottom: 2.5px solid #2563eb;
      padding-bottom: 6px;
      margin-top: 0;
      margin-bottom: 8px;
    }}

    h2 {{
      font-size: 14pt;
      color: #1e3a8a;
      border-bottom: 1.5px solid #e2e8f0;
      padding-bottom: 4px;
      margin-top: 24px;
      margin-bottom: 12px;
      page-break-after: avoid;
    }}

    h3 {{
      font-size: 11.5pt;
      color: #0f172a;
      margin-top: 16px;
      margin-bottom: 8px;
      page-break-after: avoid;
    }}

    p {{
      margin-top: 0;
      margin-bottom: 10px;
      text-align: justify;
    }}

    blockquote {{
      background-color: #f8fafc;
      border-left: 4px solid #3b82f6;
      margin: 12px 0;
      padding: 10px 14px;
      color: #334155;
      font-size: 9.5pt;
      border-radius: 0 4px 4px 0;
    }}

    blockquote p {{
      margin-bottom: 4px;
    }}
    blockquote p:last-child {{
      margin-bottom: 0;
    }}

    hr {{
      border: 0;
      height: 1px;
      background: #e2e8f0;
      margin: 20px 0;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 16px 0;
      font-size: 9pt;
      page-break-inside: avoid;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}

    th {{
      background-color: #1e3a8a;
      color: #ffffff;
      font-weight: 600;
      text-align: center;
      padding: 7px 8px;
      border: 1px solid #1e3a8a;
    }}

    td {{
      padding: 6px 8px;
      border: 1px solid #cbd5e1;
      text-align: center;
    }}

    tbody tr:nth-child(even) {{
      background-color: #f8fafc;
    }}

    tbody tr:hover {{
      background-color: #f1f5f9;
    }}

    td:first-child {{
      text-align: left;
    }}

    code {{
      font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 8.8pt;
      background-color: #f1f5f9;
      color: #0f172a;
      padding: 2px 4px;
      border-radius: 3px;
      border: 1px solid #e2e8f0;
    }}

    img {{
      max-width: 95%;
      height: auto;
      display: block;
      margin: 14px auto 8px auto;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      page-break-inside: avoid;
    }}

    ul, ol {{
      margin-top: 0;
      margin-bottom: 10px;
      padding-left: 22px;
    }}

    li {{
      margin-bottom: 4px;
    }}

    .katex-display {{
      margin: 8px 0 !important;
    }}
  </style>
</head>
<body>
  {html_body}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[✓] HTML generado: {html_path}")

    chrome_bin = encontrar_chrome()
    if not chrome_bin:
        print("[!] No se encontro un navegador Chromium/Chrome en el sistema para imprimir a PDF.")
        print(f"    El reporte HTML autocontenido esta listo en: {html_path}")
        return

    print(f"[i] Compilando PDF con {os.path.basename(chrome_bin)} headless...")
    cmd = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=4000",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        f"file://{html_path}"
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        print(f"[!] Error al generar PDF:\n{res.stderr}")
        return

    size_kb = os.path.getsize(pdf_path) / 1024.0
    print(f"\n>>> PDF CREADO CON EXITO: {pdf_path} ({size_kb:.1f} KB)\n")

if __name__ == "__main__":
    convertir_md_a_pdf()
