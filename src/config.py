from dataclasses import dataclass, field
from typing import List

@dataclass
class Columns:
    # Matches your CSV:
    # Image Index,Finding Labels,Patient ID,Patient Age,Patient Gender,View Position
    image: str = "Image Index"
    labels: str = "Finding Labels"
    age: str = "Patient Age"
    gender: str = "Patient Gender"
    view: str = "View Position"

@dataclass
class ProjectConfig:
    # ChestX-ray14 label set (excluding 'No Finding')
    classes: List[str] = field(default_factory=lambda: [
        "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
        "Edema", "Emphysema", "Fibrosis", "Effusion",
        "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule",
        "Mass", "Hernia"
    ])
    imgsz: int = 640
    val_split: float = 0.15
    test_split: float = 0.10
    seed: int = 42
    label_sep: str = "|"          # multi-label separator in CSV
    no_finding_token: str = "No Finding"

COLUMNS = Columns()
CFG = ProjectConfig()
