Model prefetching and offline usage
By default, models are downloaded automatically upon first usage. If you would prefer to explicitly prefetch them for offline use (e.g. in air-gapped environments) you can do that as follows:

Step 1: Prefetch the models

Use the docling-tools models download utility:


$ docling-tools models download
Downloading layout model...
Downloading tableformer model...
Downloading picture classifier model...
Downloading code formula model...
Downloading easyocr models...
Models downloaded into $HOME/.cache/docling/models.
Alternatively, models can be programmatically downloaded using docling.utils.model_downloader.download_models().

Step 2: Use the prefetched models


from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

artifacts_path = "/local/path/to/models"

pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
Or using the CLI:


docling --artifacts-path="/local/path/to/models" FILE

