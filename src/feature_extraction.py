
import torch
import requests
from PIL import Image, ImageFont, ImageDraw, ImageTransform
from transformers import AutoImageProcessor, ViTModel, AutoTokenizer, T5EncoderModel
from utils.config import Config
from src.ocr import OCRDetector


class ViT:
    def __init__(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model.to(Config.device)

    def extraction(self, image_url):
        if image_url.startswith("https://"):
            images = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        else:
            images = Image.open(image_url).convert("RGB")

        inputs = self.processor(images, return_tensors="pt").to(Config.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        attention_mask = torch.ones((last_hidden_states.shape[0], last_hidden_states.shape[1]))

        return last_hidden_states.to(Config.device), attention_mask.to(Config.device)

    def pooling_extraction(self, image):
        image_inputs = self.processor(image, return_tensors="pt").to(Config.device)

        with torch.no_grad():
            image_outputs = self.model(**image_inputs)
            image_pooler_output = image_outputs.pooler_output
            image_pooler_output = torch.unsqueeze(image_pooler_output, 0)
            image_attention_mask = torch.ones((image_pooler_output.shape[0], image_pooler_output.shape[1]))

        return image_pooler_output.to(Config.device), image_attention_mask.to(Config.device)

class OCR:
    def __init__(self) -> None:
        self.ocr_detector = OCRDetector()

    def extraction(self, image_dir):

        ocr_results = self.ocr_detector.text_detector(image_dir)
        if not ocr_results:
            print("NOT OCR1")

            return "", [], []

        ocrs = self.post_process(ocr_results)

        if not ocrs:

            return "", [], []

        ocrs.reverse()

        boxes = []
        texts = []
        for idx, ocr in enumerate(ocrs):
            boxes.append(ocr["box"])
            texts.append(ocr["text"])

        groups_box, groups_text, paragraph_boxes = OCR.group_boxes(boxes, texts)
        for temp in groups_text:
            print("OCR: ", temp)

        texts = [" ".join(group_text) for group_text in groups_text]
        ocr_content = "<extra_id_0>".join(texts)
        ocr_content = ocr_content.lower()
        ocr_content = " ".join(ocr_content.split())
        ocr_content = "<extra_id_0>" + ocr_content


        return ocr_content, groups_box, paragraph_boxes

    def post_process(self,ocr_results):
        ocrs = []
        for result in ocr_results:
            text = result["text"]
            # if len(text) <=2:
            #   continue
            # if len(set(text.replace(" ", ""))) <=2:
            #   continue
            box = result["box"]

            # (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
            # w = x2 - x1
            # h = y4 - y1
            # if h > w:
            #   continue

            # if w*h < 300:
            #   continue

            ocrs.append(
                {"text": text.lower(),
                "box": box}
            )
        return ocrs

    @staticmethod
    def cut_image_polygon(image, box):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
        w = x2 - x1
        h = y4 - y1
        scl = h//7
        new_box = [max(x1-scl,0), max(y1 - scl, 0)], [x2+scl, y2-scl], [x3+scl, y3+scl], [x4-scl, y4+scl]
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = new_box
        # Define 8-tuple with x,y coordinates of top-left, bottom-left, bottom-right and top-right corners and apply
        transform = [x1, y1, x4, y4, x3, y3, x2, y2]
        result = image.transform((w,h), ImageTransform.QuadTransform(transform))
        return result


    @staticmethod
    def check_point_in_rectangle(box, point, padding_devide):
      (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
      x_min = min(x1, x4)
      x_max = max(x2, x3)

      padding = (x_max-x_min)//padding_devide
      x_min = x_min - padding
      x_max = x_max + padding

      y_min = min(y1, y2)
      y_max = max(y3, y4)

      y_min = y_min - padding
      y_max = y_max + padding

      x, y = point

      if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
        return True

      return False

    @staticmethod
    def check_rectangle_overlap(rec1, rec2, padding_devide):
      for point in rec1:
        if OCR.check_point_in_rectangle(rec2, point, padding_devide):
          return True

      for point in rec2:
        if OCR.check_point_in_rectangle(rec1, point, padding_devide):
          return True

      return False

    @staticmethod
    def group_boxes(boxes, texts):
      groups = []
      groups_text = []
      paragraph_boxes = []
      processed = []
      boxes_cp = boxes.copy()
      for i, (box, text) in enumerate(zip(boxes_cp, texts)):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box

        if i not in processed:
          processed.append(i)
        else:
          continue

        groups.append([box])
        groups_text.append([text])
        for j, (box2, text2) in enumerate(zip(boxes_cp[i+1:], texts[i+1:])):
          if j+i+1 in processed:
            continue
          padding_devide = len(groups[-1])*4
          is_overlap = OCR.check_rectangle_overlap(box, box2, padding_devide)
          if is_overlap:
            (xx1, yy1), (xx2, yy2), (xx3, yy3), (xx4, yy4) = box2
            processed.append(j+i+1)
            groups[-1].append(box2)
            groups_text[-1].append(text2)
            new_x1 = min(x1, xx1)
            new_y1 = min(y1, yy1)
            new_x2 = max(x2, xx2)
            new_y2 = min(y2, yy2)
            new_x3 = max(x3, xx3)
            new_y3 = max(y3, yy3)
            new_x4 = min(x4, xx4)
            new_y4 = max(y4, yy4)

            box = [(new_x1, new_y1), (new_x2, new_y2), (new_x3, new_y3), (new_x4, new_y4)]

        paragraph_boxes.append(box)
      return groups, groups_text, paragraph_boxes
