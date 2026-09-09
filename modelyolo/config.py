"""
Configuration settings for Land Waste & Object Detection Model.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11_trash_detection.pt"
COCO_MODEL_PATH = BASE_DIR / "yolo11n.pt"
OUTPUT_DIR = BASE_DIR / "runs" / "detect"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Detection Parameters
DEFAULT_CONF_THRESHOLD = 0.15  # Lowered default for high sensitivity in live camera feeds
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMG_SIZE = 640

# Custom Waste Class Label Mapping (for yolo11_trash_detection.pt)
TRASH_CLASS_NAMES = {
    0: "Plastic / Bottle",
    1: "Paper / Cardboard",
    2: "Metal / Can",
    3: "Glass",
    4: "Organic / Debris",
    5: "Polythene / Bag",
    6: "E-Waste / Hazardous",
    7: "General Land Waste"
}

# Common COCO Object Classes that are relevant to litter & everyday objects
COMMON_OBJECT_MAPPING = {
    "bottle": "Plastic Bottle / Drink Container",
    "cup": "Cup / Disposable Glass",
    "wine glass": "Glass Container",
    "fork": "Plastic Utensil",
    "knife": "Plastic Utensil",
    "spoon": "Plastic Utensil",
    "bowl": "Food Bowl / Container",
    "banana": "Organic Waste (Banana peel)",
    "apple": "Organic Waste (Apple core)",
    "sandwich": "Organic Food Waste",
    "orange": "Organic Waste (Fruit peel)",
    "broccoli": "Organic Food Waste",
    "carrot": "Organic Food Waste",
    "backpack": "Bag / Litter",
    "handbag": "Bag / Litter",
    "suitcase": "Discarded Bag / Luggage",
    "cell phone": "E-Waste (Phone)",
    "laptop": "E-Waste (Laptop)",
    "keyboard": "E-Waste (Keyboard)",
    "mouse": "E-Waste (Mouse)",
    "tv": "E-Waste (Monitor / TV)",
    "chair": "Discarded Furniture",
    "couch": "Discarded Furniture",
    "bed": "Discarded Furniture",
    "book": "Paper Waste (Book / Paper)",
    "vase": "Ceramic / Glass Item",
    "scissors": "Metal Item / Tool"
}

# Color palette for bounding boxes (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 165, 255),    # Orange
    1: (255, 191, 0),    # Sky Blue
    2: (0, 255, 255),    # Yellow
    3: (255, 0, 255),    # Magenta
    4: (0, 200, 0),      # Green
    5: (0, 69, 255),     # Red-Orange
    6: (0, 0, 255),      # Red
    7: (180, 105, 255)   # Pink
}


def get_waste_severity(trash_count: int) -> tuple[str, tuple[int, int, int]]:
    """Returns severity label and BGR color based on count."""
    if trash_count == 0:
        return "CLEAN LAND", (0, 200, 0)          # Green
    elif trash_count <= 3:
        return "LOW WASTE", (0, 220, 220)        # Light Yellow
    elif trash_count <= 8:
        return "MODERATE WASTE", (0, 140, 255)   # Orange
    else:
        return "CRITICAL POLLUTION", (0, 0, 230)  # Red
