from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_warmup_pdf():
    """Create a minimal PDF file for warming up the document processing pipeline."""
    output_path = '_processed_output/warmup.pdf'
    c = canvas.Canvas(output_path, pagesize=letter)
    
    # Add some text
    c.drawString(72, 720, "Warmup PDF")
    c.drawString(72, 700, "This is a minimal PDF file used to warm up the document processing pipeline.")
    
    # Add a simple table
    c.drawString(72, 650, "Simple Table:")
    c.rect(72, 600, 200, 40)
    c.line(172, 600, 172, 640)
    c.drawString(82, 620, "Cell 1")
    c.drawString(182, 620, "Cell 2")
    
    c.save()
    print(f"Created warmup PDF at {output_path}")

if __name__ == "__main__":
    create_warmup_pdf()
