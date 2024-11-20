# Copyright (c) Opendatalab. All rights reserved.

import warnings
import os

# Suppress the existing warnings
warnings.filterwarnings("ignore", message="import tensorrt_llm failed")
warnings.filterwarnings("ignore", message="import lmdeploy failed")

# Suppress PaddlePaddle GPU warnings
os.environ['FLAGS_call_stack_level'] = '0'  # Suppress PaddlePaddle warnings

import base64
import os
import time
import uuid
import zipfile
from pathlib import Path
import re

import pymupdf
from loguru import logger

from magic_pdf.libs.hash_utils import compute_sha256
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.tools.common import do_parse, prepare_env

import gradio as gr
from gradio_pdf import PDF


def read_fn(path):
    disk_rw = DiskReaderWriter(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)


def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, layout_mode, formula_enable, table_enable, language):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f"{str(Path(doc_path).stem)}_{time.time()}"
        pdf_data = read_fn(doc_path)

        # Determine parse method based on OCR and table settings
        if is_ocr or table_enable:
            parse_method = "ocr"
            try:
                import rapidocr_onnxruntime
                from rapidocr_onnxruntime import RapidOCR
                logger.info("Initializing RapidOCR for table recognition")
                ocr = RapidOCR()
            except ImportError:
                logger.error("rapidocr_onnxruntime not found")
                raise ImportError("Table recognition requires rapidocr_onnxruntime")
        else:
            parse_method = "auto"

        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)

        # Initialize models before parsing
        if table_enable:
            from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
            model_manager = ModelSingleton()
            model = model_manager.get_model(True, True)
            logger.info("Table recognition model initialized")

            # Patch the table recognition module to use our OCR
            from magic_pdf.model.sub_modules.table.rapidtable.rapid_table import RapidTableModel
            def patched_predict(self, image):
                try:
                    import numpy as np
                    
                    # Ensure image is valid
                    if image is None:
                        logger.warning("Received None image")
                        return None, None, 0

                    # Run OCR
                    try:
                        result = ocr(image)
                        logger.info(f"OCR result type: {type(result)}")
                        logger.debug(f"OCR result: {result}")
                        
                        # Check if OCR failed (returns (None, None))
                        if result == (None, None):
                            logger.warning("OCR failed to detect any text in the image")
                            # Try using the built-in OCR engine as fallback
                            try:
                                ocr_result, _ = self.ocr_engine(np.asarray(image))
                                if ocr_result:
                                    logger.info("Successfully used fallback OCR")
                                    return self.table_model(np.asarray(image), ocr_result)
                            except Exception as fallback_err:
                                logger.error(f"Fallback OCR also failed: {str(fallback_err)}")
                            return None, None, 0
                            
                    except Exception as ocr_err:
                        logger.error(f"OCR failed: {str(ocr_err)}")
                        return None, None, 0

                    # Validate OCR result
                    if result is None or not isinstance(result, tuple) or len(result) < 1:
                        logger.warning("Invalid OCR result format")
                        return None, None, 0

                    boxes_list = result[0]  # Get the first element which contains the OCR results
                    if not boxes_list:
                        logger.warning("No text detected in image")
                        return None, None, 0
                        
                    # Format OCR results for RapidTable
                    ocr_result = []
                    for item in boxes_list:
                        try:
                            box = item[0]  # box coordinates
                            text = item[1]  # text
                            confidence = item[2]  # confidence score
                            
                            # Convert box coordinates to integers
                            box = [[int(float(x)), int(float(y))] for x, y in box]
                            ocr_result.append([box, text, float(confidence)])
                        except (IndexError, ValueError) as e:
                            logger.debug(f"Skipping invalid OCR result item: {str(e)}")
                            continue
                    
                    if ocr_result:
                        logger.info(f"OCR found {len(ocr_result)} text regions")
                        try:
                            # Convert image to numpy array if it isn't already
                            image_array = np.asarray(image)
                            logger.debug(f"Image shape: {image_array.shape}")
                            logger.debug(f"First OCR result: {ocr_result[0] if ocr_result else None}")
                            
                            return self.table_model(image_array, ocr_result)
                        except Exception as table_err:
                            logger.error(f"Table model failed: {str(table_err)}")
                            logger.exception(table_err)
                            return None, None, 0
                    else:
                        logger.warning("No valid OCR results found")
                        return None, None, 0
                    
                except Exception as e:
                    logger.exception(f"Table recognition failed: {str(e)}")
                    return None, None, 0
                    
            RapidTableModel.predict = patched_predict

        do_parse(
            output_dir,
            file_name,
            pdf_data,
            [],
            parse_method,
            False,
            end_page_id=end_page_id,
            layout_model=layout_mode,
            formula_enable=formula_enable,
            table_enable=table_enable,
            lang=language,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)
        return None


def compress_directory_to_zip(directory_path, output_zip_path):
    """
    压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f"![{relative_path}](data:image/jpeg;base64,{base64_image})"

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


def to_markdown(file_path, end_pages, is_ocr, layout_mode, formula_enable, table_enable, language):
    try:
        # Validate input file path
        if not file_path or not os.path.exists(file_path):
            raise ValueError("Invalid or missing PDF file")

        # Convert float('inf') to -1 to indicate no page limit
        end_page_id = -1 if end_pages == float('inf') else end_pages - 1
        
        # Get the recognized md file and compressed file path
        result = parse_pdf(file_path, './output', end_page_id, is_ocr,
                          layout_mode, formula_enable, table_enable, language)
        
        if result is None:
            raise ValueError("PDF parsing failed")
            
        local_md_dir, file_name = result
        
        # Validate output directory
        if not os.path.exists(local_md_dir):
            raise ValueError("Output directory not created")
            
        archive_zip_path = os.path.join("./output", compute_sha256(local_md_dir) + ".zip")
        zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
        if zip_archive_success == 0:
            logger.info("压缩成功")
        else:
            logger.error("压缩失败")
            
        md_path = os.path.join(local_md_dir, file_name + ".md")
        if not os.path.exists(md_path):
            raise ValueError("Markdown file not generated")
            
        with open(md_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        md_content = replace_image_with_base64(txt_content, local_md_dir)
        # 返回转换后的PDF路径
        new_pdf_path = os.path.join(local_md_dir, file_name + "_layout.pdf")

        return md_content, txt_content, archive_zip_path, new_pdf_path
    except Exception as e:
        logger.exception(e)
        raise


latex_delimiters = [{"left": "$$", "right": "$$", "display": True},
                    {"left": '$', "right": '$', "display": False}]


def init_model():
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
    try:
        model_manager = ModelSingleton()
        txt_model = model_manager.get_model(False, False)
        logger.info(f"txt_model init final")
        ocr_model = model_manager.get_model(True, False)
        logger.info(f"ocr_model init final")
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


model_init = init_model()
logger.info(f"model_init: {model_init}")


with open("header.html", "r") as file:
    header = file.read()


latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
]
other_lang = ['ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']

all_lang = [""]
all_lang.extend([*other_lang, *latin_lang, *arabic_lang, *cyrillic_lang, *devanagari_lang])


def to_pdf(file_path):
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        else:
            pdf_bytes = f.convert_to_pdf()
            # 将pdfbytes 写入到uuid.pdf中
            # 生成唯一的文件名
            unique_filename = f"{uuid.uuid4()}.pdf"

            # 构建完整的文件径
            tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

            # 将字节数据写入文件
            with open(tmp_file_path, 'wb') as tmp_pdf_file:
                tmp_pdf_file.write(pdf_bytes)

            return tmp_file_path


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("./output", exist_ok=True)
    
    with gr.Blocks() as demo:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                file = gr.File(label="Please upload a PDF or image", file_types=[".pdf", ".png", ".jpeg", ".jpg"])
                max_pages = gr.Dropdown(
                    choices=["unlimited"] + [str(i) for i in range(1, 11)],
                    label="Max convert pages",
                    value="5"
                )
                with gr.Row():
                    layout_mode = gr.Dropdown(["layoutlmv3", "doclayout_yolo"], label="Layout model", value="layoutlmv3")
                    language = gr.Dropdown(all_lang, label="Language", value="")
                with gr.Row():
                    formula_enable = gr.Checkbox(label="Enable formula recognition", value=True)
                    is_ocr = gr.Checkbox(label="Force enable OCR", value=False)
                    table_enable = gr.Checkbox(label="Enable table recognition(test)", value=False)
                with gr.Row():
                    change_bu = gr.Button("Convert")
                    clear_bu = gr.ClearButton(value="Clear")
                pdf_show = PDF(label="PDF preview", interactive=True, height=800)
                with gr.Accordion("Examples:"):
                    example_root = os.path.join(os.path.dirname(__file__), "examples")
                    gr.Examples(
                        examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                  _.endswith("pdf")],
                        inputs=pdf_show
                    )

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label="convert result", interactive=False)
                with gr.Tabs():
                    with gr.Tab("Markdown rendering"):
                        md = gr.Markdown(label="Markdown rendering", height=900, show_copy_button=True,
                                         latex_delimiters=latex_delimiters, line_breaks=True)
                    with gr.Tab("Markdown text"):
                        md_text = gr.TextArea(lines=45, show_copy_button=True)
        file.upload(fn=to_pdf, inputs=file, outputs=pdf_show)
        def handle_convert(pdf_file, pages, is_ocr, layout_mode, formula_enable, table_enable, language):
            try:
                # Check if a file is selected
                if pdf_file is None:
                    gr.Warning("Please select a PDF file first")
                    return None, None, None, None

                # If table recognition is enabled, ensure OCR is available
                if table_enable:
                    try:
                        import rapidocr_onnxruntime
                        from rapidocr_onnxruntime import RapidOCR
                        import rapid_table
                        # Force OCR on when table recognition is enabled
                        is_ocr = True
                    except ImportError:
                        gr.Warning("Installing required packages for table recognition...")
                        try:
                            import subprocess
                            # Install packages separately
                            subprocess.check_call(["pip", "install", "--upgrade", "rapidocr-onnxruntime>=1.3.8"])
                            subprocess.check_call(["pip", "install", "--upgrade", "rapid-table>=0.1.6"])
                            subprocess.check_call(["pip", "install", "paddlepaddle"])
                            
                            import rapidocr_onnxruntime
                            import rapid_table
                        except Exception as e:
                            logger.exception(f"Failed to install OCR packages: {str(e)}")
                            gr.Warning("Failed to install table recognition packages. Disabling table mode.")
                            table_enable = False

                # Convert pages value
                try:
                    end_pages = float('inf') if pages == "unlimited" else int(pages)
                except (ValueError, TypeError):
                    gr.Warning("Invalid page number")
                    return None, None, None, None

                # Validate language
                if language is None:
                    language = ""

                return to_markdown(pdf_file, end_pages, is_ocr, layout_mode, formula_enable, table_enable, language)
            except Exception as e:
                logger.exception(e)
                gr.Warning(f"Error during conversion: {str(e)}")
                return None, None, None, None

        change_bu.click(
            fn=handle_convert,
            inputs=[pdf_show, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language],
            outputs=[md, md_text, output_file, pdf_show]
        )
        clear_bu.add([file, md, pdf_show, md_text, output_file, is_ocr, table_enable, language])

    demo.launch(server_name="0.0.0.0")