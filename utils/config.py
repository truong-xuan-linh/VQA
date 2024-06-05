class Config:
    device = "cpu"
    model_url = "https://drive.google.com/uc?id=1WqyPOxgTj9vdnEQl_TJwr_U_hdQeI1iz"
    ocr_path = "storage/vlsp_transfomer_vietocr.pth"
    model_path = "storage/vit_base_vit5_base_v2_1.3197_0.4732_3.5212.pt"
    question_maxlen = 32
    vietocr_threshold = 0.5
    answer_maxlen = 56
    ocr_maxlen = 128
    ocr_maxobj = 10000
    num_ocr = 32
    num_beams = 3
    revision = "version_2_with_extra_id_0"