# Custom Nodes Registration
from .enhanced_comfy_spritesheet_node import EnhancedSpritesheetProcessor
from .trading_card_nodes import NODE_CLASS_MAPPINGS as TC_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TC_NAMES

NODE_CLASS_MAPPINGS = {
    "EnhancedSpritesheetProcessor": EnhancedSpritesheetProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedSpritesheetProcessor": "🎮 Enhanced Spritesheet Processor",
}

# Trading Card Nodes hinzufügen
NODE_CLASS_MAPPINGS.update(TC_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(TC_NAMES)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
