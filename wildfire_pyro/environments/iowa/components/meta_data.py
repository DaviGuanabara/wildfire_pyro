from dataclasses import dataclass
from typing import List


@dataclass
class Metadata:
    time: str                 # "data"
    position: List[str]       # ["Latitude1", "Longitude1", "Elevation [m]"]
    id: str                   # "ID" (ou outro nome)
    target: List[str]         # ["high", "low"] ou outra lista
    exclude: List[str] = None  # colunas a excluir (opcional)
