import os
import sys
from loguru import logger

# Check required packages
required_packages = [
    'magic_pdf',
    'unimernet',
    'detectron2',
    'torch',
    'torchvision',
    'paddleocr',
    'ultralytics'
]

def check_dependencies():
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.info("Please install missing packages using:")
        logger.info("pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com")
        sys.exit(1)

try:
    check_dependencies()
    
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_name = "demo1"
    pdf_path = os.path.join(current_script_dir, f"{demo_name}.pdf")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
        
    pdf_bytes = open(pdf_path, "rb").read()
    jso_useful_key = {"_pdf_type": "", "model_list": []}
    local_image_dir = os.path.join(current_script_dir, 'images')
    
    # Create images directory if it doesn't exist
    os.makedirs(local_image_dir, exist_ok=True)
    
    image_dir = str(os.path.basename(local_image_dir))
    image_writer = DiskReaderWriter(local_image_dir)
    pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()
    md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
    
    output_path = os.path.join(current_script_dir, f"{demo_name}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logger.success(f"Successfully converted PDF to Markdown: {output_path}")
    
except Exception as e:
    logger.exception(e)
    sys.exit(1)