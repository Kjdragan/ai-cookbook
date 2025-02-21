import os
import json
import time
from pathlib import Path
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice, PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.utils.model_downloader import download_models

class DataPipeline:
    def __init__(self, artifacts_path=None):
        self.init_start_time = time.time()
        print("Initializing document processing pipeline...")
        
        if artifacts_path is None:
            artifacts_path = os.path.join(str(Path.home()), '.cache', 'docling', 'models')
        
        # Download and prefetch models if needed
        print("Checking and downloading required models...")
        self.warmup_models(artifacts_path)
        
        print("Setting up CUDA acceleration...")
        # Set up CUDA acceleration
        self.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CUDA
        )

        # Configure pipeline options with CUDA acceleration
        print("Configuring pipeline options...")
        self.pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
        self.pipeline_options.accelerator_options = self.accelerator_options
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True

        # Create converter with CUDA-enabled options
        print("Creating document converter...")
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                )
            }
        )
        
        init_time = time.time() - self.init_start_time
        print(f"Pipeline initialization time: {init_time:.2f} seconds")
        print("Pipeline initialization complete. Ready for processing.")

    def warmup_models(self, artifacts_path):
        """Download and prefetch required models."""
        try:
            # Ensure the artifacts directory exists
            os.makedirs(artifacts_path, exist_ok=True)
            
            # Download models if they're not already present
            download_models()
            print(f"Models downloaded successfully to: {artifacts_path}")
        except Exception as e:
            print(f"Warning: Model download encountered an error: {e}")
            print("Continuing with existing models if available...")

    def timing_decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Starting {func.__name__}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Processing time for {func.__name__}: {end_time - start_time:.2f} seconds")
            return result
        return wrapper

    @timing_decorator
    def process_pdf(self, url):
        result = self.converter.convert(url)
        document = result.document
        markdown_output = document.export_to_markdown()
        json_output = document.export_to_dict()

        # Save outputs to _processed_output directory
        base_name = os.path.basename(url).replace('.pdf', '')
        markdown_file = os.path.join('_processed_output', f'{base_name}.md')
        json_file = os.path.join('_processed_output', f'{base_name}.json')

        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_output)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4)

        return markdown_file, json_file

    @timing_decorator
    def process_html(self, url):
        result = self.converter.convert(url)
        document = result.document
        markdown_output = document.export_to_markdown()

        # Save output to _processed_output directory
        base_name = os.path.basename(url).replace('.html', '')
        markdown_file = os.path.join('_processed_output', f'{base_name}.md')

        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_output)

        return markdown_file

    @timing_decorator
    def process_sitemap(self, url):
        sitemap_urls = get_sitemap_urls(url)
        conv_results_iter = self.converter.convert_all(sitemap_urls)

        docs = []
        for result in conv_results_iter:
            if result.document:
                document = result.document
                docs.append(document)
        return docs

if __name__ == "__main__":
    start_time = time.time()
    pipeline = DataPipeline()
    markdown_file, json_file = pipeline.process_pdf("https://arxiv.org/pdf/2408.09869")
    total_time = time.time() - start_time
    print(f"Saved markdown to {markdown_file} and JSON to {json_file}")
    print(f"Total execution time: {total_time:.2f} seconds")
