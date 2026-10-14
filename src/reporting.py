from jinja2 import Environment, FileSystemLoader
import os
from datetime import datetime

def generate_report(validation, anomalies, figures, summary, insights=None, output_path="outputs/report.html", recommendations=None):
    print("[REPORT] Generating styled HTML report with Jinja2...")

    # Load template
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report_template.html")

    # Add metadata
    footer = {
        "generated_on": datetime.now().strftime("%d %B %Y, %I:%M %p"),
        "branding": "DataPilot - AI Data Analyst & Recommendation Engine"
    }

    # Render HTML
    html_content = template.render(
        validation=validation,
        anomalies=anomalies,
        figures=figures,
        summary=summary,
        insights=insights or {},
        recommendations=recommendations or [],
        footer=footer
    )

    # Ensure output folder exists and never writes directly to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.dirname(output_path)
    if not out_dir or out_dir.strip() in [".", "", "./", ".\\"] or os.path.abspath(out_dir) == project_root:
        file_name = os.path.basename(output_path) or "report.html"
        output_path = os.path.join(project_root, "outputs", file_name)
        out_dir = os.path.dirname(output_path)

    os.makedirs(out_dir, exist_ok=True)

    # Save HTML
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[REPORT_SAVED] HTML Report saved to {output_path}")

    # Save PDF version if wkhtmltopdf is configured
    pdf_path = output_path.replace(".html", ".pdf")
    try:
        import pdfkit
        wk_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        if os.path.exists(wk_path):
            config = pdfkit.configuration(wkhtmltopdf=wk_path)
            pdfkit.from_string(html_content, pdf_path, configuration=config)
            print(f"[PDF_SAVED] PDF Report saved to {pdf_path}")
    except Exception as e:
        print(f"[PDF_INFO] PDF report skipped: {e}")

    return output_path