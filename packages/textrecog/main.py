import pytesseract
from PIL import Image
import os


def text_recog_windows_os(image_path):
    data_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), r"Tesseract-OCR/tesseract.exe")
    pytesseract.pytesseract.tesseract_cmd = data_file_path
    custom_config = r' --oem 3 --psm 6'
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='rus', config=custom_config)
    return text

def text_recog_macos_linux(image_path):
    os.environ['TESSDATA_PREFIX'] = ''
    custom_config = r' --oem 3 --psm 6'
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='rus', config=custom_config)
    return text
