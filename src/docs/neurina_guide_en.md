# The Comprehensive Guide to Neurina App (English)

Welcome to the official documentation for the Neurina smart application. This document covers technical details, user guides, FAQs, and team structure, serving as the primary knowledge base for our RAG (Retrieval-Augmented Generation) system to answer user queries in English.

---

## 1. The Development Team

The Neurina app was brought to life by an exceptional team of engineers:
- **Osama Hamada Mohamed Mohamed Elhabashy:** The AI Engineer and Backend Developer. Osama is the mastermind behind the agentic workflows, the dynamic state machines, the RAG implementation, and the database architecture.
- **Adel:** The Frontend Developer. Adel is responsible for translating backend intelligence into an interactive, sleek, and user-friendly web interface.
- **Mohamed Eltokhy:** The Flutter Developer. Mohamed built the mobile application, ensuring that Neurina's AI capabilities are accessible seamlessly on smartphones.

---

## 2. What is Neurina?

Neurina is a revolutionary Style Transfer application powered by Generative AI and Agentic Workflows. Unlike traditional apps with static filters, Neurina acts as your personal AI graphic designer.
You chat with the AI, request any style you can imagine, and the AI will search the web for the best reference images, ask you to pick your favorite, and then apply that exact style to your personal photo.

**Key Features:**
- **Interactive Chat Interface:** Natural language conversations instead of rigid menus.
- **Self-Correction:** If the AI fails to find a good reference image, it automatically rewrites your prompt and searches again.
- **Intelligent RAG System:** You can ask the AI general questions about the app, and it will answer accurately based on this documentation.

---

## 3. How to Use the App (Step by Step)

Using Neurina is as easy as having a conversation:

1. **Initial Request:** Start by describing the style you want. (e.g., "I want my photo to look like a Da Vinci painting" or "Make me look like Tom Cruise").
2. **Search & Candidates:** The AI's Query Agent will extract keywords, search the internet, and present you with up to 3 candidate images that match your description.
3. **Selection or Refinement:** 
   - If you like an image: Reply with "I choose the first one" or "Option 2".
   - If you don't like any: Tell the AI "These aren't good, try something else." The AI will automatically try a different search query.
4. **Upload Source Image:** Once a style is selected, the AI will ask you to upload your personal photo.
5. **Processing:** The system will apply the style to your photo and deliver the stunning result in seconds!

---

## 4. Technical Architecture (Under the Hood)

For users interested in how Neurina works:
- The backend is designed as a **State Machine** managed by a `SupervisorAgent`.
- **IntentAgent:** Analyzes every message to determine your intent (e.g., asking a question, selecting an image, requesting a style transfer).
- **Databases:** We use `MongoDB` to store chat sessions (so you can resume later) and `ChromaDB` as a vector database for the RAG system to answer questions.
- **RAG (Retrieval-Augmented Generation):** Allows the AI to read this documentation to answer user questions without hallucinating or making up facts.

---

## 5. Frequently Asked Questions (FAQ)

**Q: Does the app keep or publish my personal photos?**
A: No. We respect your privacy completely. Your photos are only used during the Style Transfer process and are not shared or published anywhere.

**Q: What should I do if the AI says it couldn't find suitable images?**
A: Your description might be too specific or vague. Try rephrasing your request using famous styles, names, or eras. The AI actually tries 3 times in the background before giving up and asking you to try again!

**Q: Does the app support Arabic and English?**
A: Yes! The AI Agents, including the chat interface and the RAG question-answering system, fully support both Arabic and English. 

**Q: Who should I contact for support or joining the team?**
A: You can reach out to our AI Engineer Osama Hamada Mohamed Mohamed Elhabashy, our Frontend Developer Adel, or our Flutter Developer Mohamed Eltokhy via the app's support channels.

**Q: What exactly is Style Transfer?**
A: Style Transfer is an AI technique that takes the artistic style, lighting, or theme from one image (the reference) and applies it to another image (your photo) while keeping your original facial features intact.
