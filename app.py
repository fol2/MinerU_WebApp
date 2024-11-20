from flask import Flask, request, send_file, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from loguru import logger
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdf_path)

    try:
        # Convert PDF to Markdown
        pdf_bytes = open(pdf_path, "rb").read()
        jso_useful_key = {"_pdf_type": "", "model_list": []}
        local_image_dir = os.path.join(OUTPUT_FOLDER, 'images')
        os.makedirs(local_image_dir, exist_ok=True)
        image_writer = DiskReaderWriter(local_image_dir)
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
        md_content = pipe.pipe_mk_markdown(local_image_dir, drop_mode="none")

        output_md_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.md")
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return jsonify({'markdown_path': output_md_path}), 200

    except Exception as e:
        logger.exception(e)
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 