"""Swap the ID-1 card in a real photo for a template card, keeping the photo's light (synthetic training data)."""
from .detect import ProdCardDetector, grabcut_quad, quad_from_mask, regular
from .swap import Template, load_templates, quad_from_mrz, swap_card

__all__ = ["ProdCardDetector", "Template", "grabcut_quad", "load_templates", "quad_from_mask", "quad_from_mrz",
           "regular", "swap_card"]
