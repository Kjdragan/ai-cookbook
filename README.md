# Introduction

This Cookbook contains examples and tutorials to help developers build AI systems, offering copy/paste code snippets that you can easily integrate into your own projects.

## Document Processing Pipeline with GPU Acceleration

This project implements a document processing pipeline using Docling, LanceDB, and OpenAI embeddings, with GPU acceleration via CUDA. The pipeline processes PDF documents, extracts text and metadata, and stores the extracted information as searchable vector embeddings.

### GPU Acceleration Features

- **CUDA Integration**: Uses PyTorch 2.6.0+cu124 for GPU-accelerated document processing
- **Hardware Support**: Compatible with NVIDIA GPUs (tested with GeForce GTX 1660 Ti)
- **Accelerated Components**:
  - Document conversion with Docling's DocumentConverter
  - OCR text extraction with EasyOCR
  - Table structure analysis and image processing
  - Document chunking and processing

### Setup and Configuration

The CUDA integration is configured in the following files:

1. **pyproject.toml**: Contains PyTorch CUDA configuration using custom package indices
   ```toml
   [tool.uv.sources]
   torch = [
       { index = "pytorch-cu124" },
   ]
   torchvision = [
       { index = "pytorch-cu124" },
   ]
   
   [[tool.uv.index]]
   name = "pytorch-cu124"
   url = "https://download.pytorch.org/whl/cu124"
   ```

2. **datapipeline.py**: Configures the Docling pipeline to use CUDA for acceleration
   ```python
   # Set up CUDA acceleration
   self.accelerator_options = AcceleratorOptions(
       num_threads=8, 
       device=AcceleratorDevice.CUDA
   )
   ```

### Performance Benefits

- Significantly faster document processing, especially for OCR and image analysis
- Improved throughput for batch processing of multiple documents
- Enhanced performance for large documents (50+ pages)

## About Me

Hi! I'm Dave, AI Engineer and founder of Datalumina®. On my [YouTube channel](https://www.youtube.com/@daveebbelaar?sub_confirmation=1), I share practical tutorials that teach developers how to build AI systems that actually work in the real world. Beyond these tutorials, I also help people start successful freelancing careers. Check out the links below to learn more!

### Explore More Resources

Whether you're a learner, a freelancer, or a business looking for AI expertise, we've got something for you:

1. **Learning Python for AI and Data Science?**  
   Join our **free community, Data Alchemy**, where you'll find resources, tutorials, and support  
   ▶︎ [Learn Python for AI](https://www.skool.com/data-alchemy)

2. **Ready to start or scale your freelancing career?**  
   Learn how to land clients and grow your business  
   ▶︎ [Start freelancing](https://www.datalumina.com/data-freelancer)

3. **Need expert help on your next project?**  
   Work with me and my team to solve your data and AI challenges  
   ▶︎ [Work with me](https://www.datalumina.com/solutions)

4. **Building AI-powered applications?**  
   Access the **GenAI Launchpad** to accelerate your AI app development  
   ▶︎ [Explore the GenAI Launchpad](https://launchpad.datalumina.com/)
