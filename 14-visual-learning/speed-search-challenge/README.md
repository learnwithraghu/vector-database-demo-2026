# Needle in a Haystack — Speed Search Challenge

## The Lesson 
This interactive lab demonstrates why **Vector Search** is fundamentally better than traditional **Keyword Search** when finding information based on *semantic meaning*.

Often, users don't know the exact keyword for what they're looking for. If they search for "a device for taking pictures" in a traditional database, an exact text match will fail to recognize related items like "camera" or "camcorder". 

Vector Databases solve this by translating text into mathematical vectors (embeddings). Sentences with similar meanings are placed closer together in this vector space. 

## The Game
1. **Round 1:** You act as the keyword search engine. You must manually scroll through 50 cards to find the item that best matches the prompt: *"a device for taking pictures"*. A timer will track your speed.
2. **Round 2:** The AI agent acts as a Vector Database. Using cosine similarity (meaning-based matching), it instantly locates the correct card. 

At the end, a comparison reveals the dramatic difference in speed and accuracy. Keyword search breaks on semantic meaning. Vector search doesn't. 
