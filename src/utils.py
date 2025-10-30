# src/utils.py
from fpdf import FPDF
import pandas as pd

def make_pdf(csv_path='results/eval_results.csv', out_pdf='reports/llm_report.pdf'):
    df = pd.read_csv(csv_path)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "LLM Benchmark Report", ln=True)
    pdf.set_font("Arial", "", 12)

    for model, group in df.groupby("model"):
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Model: {model}", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(
            0, 6, f"Average Semantic Score: {group['semantic_score'].mean():.3f}\n"
                  f"Total Samples: {len(group)}"
        )
    pdf.output(out_pdf)
    print("📘 Report saved at:", out_pdf)
