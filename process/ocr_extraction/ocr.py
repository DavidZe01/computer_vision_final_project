# ocr.py
import numpy as np
import easyocr
from typing import List, Tuple, Any

class OcrProcess:
    def __init__(self):
        # Inicializar EasyOCR para detección de texto en español
        self.ocr_detector = easyocr.Reader(['es'], gpu=False)  # Cambiar a gpu=True si tienes una GPU compatible y configurada

    def text_detection(self, text_image: np.ndarray) -> Tuple[int, List[Tuple[Any, str, float]]]:
        # Detecta líneas de texto en la imagen usando EasyOCR
        text_line_detected = self.ocr_detector.readtext(text_image)
        return len(text_line_detected), text_line_detected

    def extractor_text_line(self, text) -> Tuple[List[int], str, float]:
        # Extrae el cuadro delimitador y el texto detectado con confianza
        bbox, text_extracted, text_confidence = text
        xi, yi, xf, yf = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])
        text_bbox = [xi, yi, xf, yf]
        return text_bbox, text_extracted, text_confidence
