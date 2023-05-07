from typing import List, Optional

# this file stores all utility functions, such as calculating the metrics, preprocessing the datasets


class Dataset:
    def __init__(self, texts: List[str], label_names: List[str], prompt: Optional[str] = None):
        self.texts = texts
        self.label_names = label_names
        self.prompt = prompt
        
class Labels:
    def __init__(self, labels: List[int]):
        self.labels = labels
