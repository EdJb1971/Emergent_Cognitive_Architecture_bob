# Quick fix script for main.py indentation
import re

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the indentation issue
content = content.replace(
    """        )
            # Connect to ChromaDB for persistent profile storage
            await app.state.emotional_memory_service.connect()
            logger.info("EmotionalMemoryService initialized and connected to ChromaDB.")""",
    """        )
        # Connect to ChromaDB for persistent profile storage
        await app.state.emotional_memory_service.connect()
        logger.info("EmotionalMemoryService initialized and connected to ChromaDB.")"""
)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed indentation in main.py")
