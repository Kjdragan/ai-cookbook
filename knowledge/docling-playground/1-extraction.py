from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice, PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.datamodel.settings import settings

# Set up CUDA acceleration
accelerator_options = AcceleratorOptions(
    num_threads=8, device=AcceleratorDevice.CUDA
)

# Configure pipeline options with CUDA acceleration
pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options = accelerator_options
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True

# Create converter with CUDA-enabled options
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

# Enable profiling to measure performance
settings.debug.profile_pipeline_timings = True

# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------

result = converter.convert("https://arxiv.org/pdf/2408.09869")
doc_conversion_secs = result.timings["pipeline_total"].times
document = result.document
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()

print(markdown_output)
print(f"Conversion time (seconds): {doc_conversion_secs}")

# --------------------------------------------------------------
# Basic HTML extraction
# --------------------------------------------------------------

result = converter.convert("https://ds4sd.github.io/docling/")
document = result.document
markdown_output = document.export_to_markdown()
print(markdown_output)

# --------------------------------------------------------------
# Scrape multiple pages using the sitemap
# --------------------------------------------------------------

sitemap_urls = get_sitemap_urls("https://ds4sd.github.io/docling/")
conv_results_iter = converter.convert_all(sitemap_urls)

docs = []
for result in conv_results_iter:
    if result.document:
        document = result.document
        docs.append(document)
