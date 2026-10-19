#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convertir_informe_a_pdf.py
Convierte 'INFORME_EXPERIMENTOS.md' a un PDF estatico ('INFORME_EXPERIMENTOS.pdf')
utilizando Python (markdown + KaTeX) y el motor headless de Google Chrome.
"""

import os
import sys
import base64
import subprocess
import markdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "INFORME_EXPERIMENTOS.md")
HTML_PATH = os.path.join(BASE_DIR, "INFORME_EXPERIMENTOS.html")
PDF_PATH = os.path.join(BASE_DIR, "INFORME_EXPERIMENTOS.pdf")
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

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

def generar_html():
    print(f"Leyendo: {MD_PATH}")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Incrustar imagenes como Base64 para que el HTML sea 100% autocontenido
    figuras = [
        "figura_1_apartado_a_bloques.png",
        "figura_2_apartados_b_c_d_tiempos.png",
        "figura_3_apartado_e_comparacion_speedup_eficiencia.png",
        "figura_4_apartado_e_bloques_alg2_vs_serial.png",
        "figura_5_apartado_f_escalabilidad_debil.png",
    ]
    for fig_name in figuras:
        fig_path = os.path.join(BASE_DIR, fig_name)
        if os.path.exists(fig_path):
            data_url = encode_image_base64(fig_path)
            md_text = md_text.replace(f"({fig_name})", f"({data_url})")

    # Conversion de Markdown a HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"]
    )

    # Plantilla HTML con CSS profesional y soporte de KaTeX
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Informe de Resultados Experimentales - Tarea 2 IIC3533</title>
  
  <!-- KaTeX CSS y JS -->
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

    /* Tablas elegantes y compactas */
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
      font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 8.8pt;
      background-color: #f1f5f9;
      color: #0f172a;
      padding: 2px 4px;
      border-radius: 3px;
      border: 1px solid #e2e8f0;
    }}

    /* Imagenes y Figuras */
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

    /* Control de saltos de pagina para impresion */
    .page-break {{
      page-break-before: always;
    }}
  </style>
</head>
<body>
  {html_body}
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML generado exitosamente en: {HTML_PATH}")

def generar_pdf():
    if not os.path.exists(CHROME_PATH):
        raise FileNotFoundError(f"No se encontro Google Chrome en {CHROME_PATH}")

    print("Compilando HTML a PDF con Google Chrome headless...")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=4000",
        f"--print-to-pdf={PDF_PATH}",
        "--no-pdf-header-footer",
        f"file://{HTML_PATH}"
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(PDF_PATH) or os.path.getsize(PDF_PATH) == 0:
        print(f"Error generando PDF:\n{res.stderr}")
        sys.exit(1)

    size_kb = os.path.getsize(PDF_PATH) / 1024.0
    print(f"\n>>> PDF CREADO CON EXITO: {PDF_PATH} ({size_kb:.1f} KB)\n")

if __name__ == "__main__":
    generar_html()
    generar_pdf()
