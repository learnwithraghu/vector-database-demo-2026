# Demo 3: Semantic Search Engine (The "Brain")

## 🎯 Level Up: From Numbers to Meaning
In Demo 2, we manually typed numbers like `[0.9, 0.1]`. That's tedious!
Now, we introduce the **Neural Network**. We will use a model to *automatically* turn English text into vectors. This allows us to search by **Meaning**, not keyword.

*   **Previous Step**: Understanding Vectors manually.
*   **Next Step**: Adding filters to our search.

## 🛠️ Pre-flight Check
```bash
# 1. Check if Qdrant is running
docker ps 

# 2. Setup Python Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements (New: sentence-transformers & streamlit!)
pip install qdrant-client sentence-transformers streamlit watchdog
```

### 1. The Analogy: "The Intuitive Librarian"
*   **Keyword Search** (Old Way) is like a bad librarian. You ask for "stories about extraterrestrials", and he says *"Sorry, I don't have any books with the word 'extraterrestrials' in the title."* He ignores the book named "Aliens".
*   **Semantic Search** (New Way) is like a genius librarian. He reads your mind, not just your words. He knows that *Extraterrestrial* = *Alien* = *Martian*.
*   In this demo, we replace the "Keyword Match" with an **"Idea Match"**.

### 2. The Encoder (Sentence Transformers)
Explain: "Computers don't understand text. They understand numbers. We use a model to translate 'Star Wars' into a list of 384 numbers."
*   We use `all-MiniLM-L6-v2`. It's small, fast, and good for English.

### 2. Prepare Data (Shared Dataset)
We will load a shared dataset `datasets/movies.json` containing diverse movies like **Arrival**, **The Matrix**, and **Finding Nemo**.
1.  Read the JSON file.
2.  Loop through movies.
3.  Turn specific descriptions into vectors.

### 3. The Search
We search for "**aliens attacking earth**".

*   **Observation**: Qdrant finds "**Arrival**" even though the word "Aliens" is NOT in the description (it says "Spaceships" and "Heptapods").
*   **Success**: This proves the model understands that *Spaceships landing* ≈ *Aliens attacking*.
*   **💡 Teacher's Note (The Magic Explained)**:
    *   Ask students: *"How did it know?"*
    *   **Answer**: The Model (`all-MiniLM-L6-v2`) has read billions of sentences during training.
    *   It learned that words like **"Attack"**, **"War"**, **"Invasion"**, and **"Conflict"** often appear in similar contexts.
    *   It learned that **"Earth"** and **"World"** are related.
    *   Therefore, it places the vector for *"Aliens attacking Earth"* very close to *"Extraterrestrials invading the World"* in the mathematical space.
    *   It's not matching words; it's matching **concepts**.

## 💻 Code Walkthrough: `semantic_search.py`

### Phase 1: The "Brain" (Lines 6-10)
*   **Line 7**: `model = SentenceTransformer('all-MiniLM-L6-v2')`
    *   This loads a small pre-trained Neural Network 🧠 onto your CPU.
    *   It knows how to read English and output 384 numbers.
*   **Line 19**: `VectorParams(size=384, ...)`
    *   We MUST match the collection size to the model's output. If the model outputs 384 numbers, the database bucket must be width 384.

### Phase 2: Learning the Data (Lines 23-45)
*   Instead of a hardcoded list, we now load `../datasets/movies.json`.
*   **Lines 35**: `vector = model.encode(doc["description"])`
    *   This is the translation step. Text -> Numbers.

### Phase 3: The Search (Lines 50-60)
*   **Line 50**: `query = "Aliens attacking earth"`
*   **Line 53**: `model.encode(query)` transforms our question.
*   **Result**: The code prints the top matches. We expect "Arrival" to be at the top because its description ("gigantic spaceships touch down") is semantically closest to "aliens attacking".

## 🤖 Bonus Information: The "Chatbot" Interface
To make this more interactive, we have added a Streamlit App.

1.  **Run the App**:
    ```bash
    streamlit run 03-semantic-search/chatbot_ui_03.py
    ```
2.  **Interact**:
    *   Type a query like "I want to fly with my dog".
    *   See the results dynamically!

