import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = ROOT_DIR + "/models"
DINO_CONFIG_FILE = (
    ROOT_DIR + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DEFAULT_OUTPUT_FILE = "output.csv"
DEFAULT_PROMPT_FILE = "caption_prompt.txt"

IMAGE_WARNING_THRESHOLD = 16
