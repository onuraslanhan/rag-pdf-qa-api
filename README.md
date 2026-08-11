# Multi-Source RAG API

A FastAPI-based Retrieval-Augmented Generation (RAG) system that lets you upload documents from multiple formats (PDF, DOCX, web URLs) and ask questions about their content. Built with LangChain, ChromaDB, HuggingFace embeddings, and Google's Gemini model.

## What it does

- Ingests content from **PDF**, **DOCX**, and **web pages** (via `trafilatura` for clean text extraction)
- Indexes everything into a single persistent ChromaDB vector store
- Answers questions using only the retrieved context — with **source attribution** showing which document the answer came from
- Refuses to answer when no relevant content exists, instead of hallucinating

## Architecture

```
Upload (PDF / DOCX / URL) 
    → Loader (PyPDFLoader / Docx2txtLoader / trafilatura)
    → Text splitter (chunk_size=1000, overlap=150)
    → HuggingFace embeddings (all-MiniLM-L6-v2)
    → ChromaDB (persistent, single shared collection)

Question 
    → Similarity search with relevance scores (k=5)
    → Score threshold filter
    → Context assembly + source metadata
    → Gemini (via LangChain) 
    → Answer + sources
```

## Engineering decisions

**Why a single shared vector store instead of one per document?**
Retrieval needs to work across all uploaded sources at once — the user shouldn't have to specify which document their question is about. Chroma's metadata (`source`) tags each chunk, so retrieval finds the right content regardless of which document it came from, and the response reports which source(s) were actually used.

**Why `add_documents()` instead of `from_documents()`?**
`from_documents()` creates a brand-new collection every call, silently discarding everything indexed before it. Early versions of this project used it and lost previously uploaded documents on every new upload. Switching to `add_documents()` on a persistent store fixed this — new uploads are appended, not overwritten.

**Why `trafilatura` for web pages instead of a raw HTML fetch?**
A naive scrape pulls in navbars, footers, cookie banners, and ads alongside the actual content, polluting chunks with noise that hurts retrieval quality. `trafilatura` is purpose-built for main-content extraction and produces much cleaner text. `WebBaseLoader` is kept as a fallback if `trafilatura` fails to extract content.

**Why a relevance-score threshold on retrieval?**
Without it, the retriever always returns its top-k chunks regardless of how relevant they actually are — so a completely unrelated question still gets 5 chunks of context, and the LLM may hallucinate an answer from irrelevant material. Filtering by score lets the system say "this isn't in the documents" instead of guessing.

## A real limitation found during testing

The relevance scores returned by Chroma are **not absolute** — they shift depending on what else is in the collection. The same question/chunk pair scored ~0.30 with one document loaded, and ~0.06 once three unrelated-topic documents (statistics, numerical methods, medical) were added to the same store. A fixed threshold that works for a small, single-topic collection can silently break as the collection grows and diversifies.

Current threshold: `0.05` (tuned empirically against manually verified true/false cases from the test set below). This is a known fragility — a more robust approach would use a relative/adaptive threshold instead of a fixed value. Left as a documented next step rather than solved, in the interest of being upfront about it.

A second issue surfaced during testing: re-uploading the same URL multiple times creates duplicate chunks, since there's no check for "this source is already indexed." This inflates the retrieval pool and can subtly shift which chunks rank in the top-k. Not yet fixed — noted here as a known gap.

## Test results

Tested with 4 sources of different types and topics, to verify both correct retrieval/attribution and that the threshold actually blocks hallucination on irrelevant questions.

| # | Question | Expected source | Result | Correct? |
|---|---|---|---|---|
| 1 | What are the symptoms of Erdheim–Chester disease? | Wikipedia (ECD) | Correct, detailed answer, correct source | ✅ |
| 2 | What is the histology of the disease? | Wikipedia (ECD) | Correct, correct source | ✅ |
| 3 | What is the treatment of the disease? | Wikipedia (ECD) | Correct, correct source | ✅ |
| 4 | How do you generate a Bernoulli random variable using a uniform random number? | CENG 222 PDF | Correct formula, correct source | ✅ |
| 5 | What is Chebyshev's inequality? | CENG 222 PDF | Correct formula, correct source | ✅ |
| 6 | How does the rejection method work? | CENG 222 PDF | Correct explanation, correct source | ✅ |
| 7 | Standard deviation formula for estimated probability p̂? | CENG 222 PDF | Correct formula, correct source | ✅ |
| 8 | When was the Bastille stormed? | French Revolution doc | Correct (July 14, 1789), correct source | ✅ |
| 9 | Main causes of the French Revolution? | French Revolution doc | Correct, 6-point summary, correct source | ✅ |
| 10 | Who was executed in January 1793? | French Revolution doc | Correct (Louis XVI), correct source | ✅ |
| 11 | What was the Reign of Terror? | French Revolution doc | Correct, correct source | ✅ |
| 12 | Who overthrew the Directory in 1799? | French Revolution doc | Correct (Bonaparte), correct source | ✅ |

**12/12 correct retrieval and source attribution** across 3 topically unrelated sources (medical, statistics, history) mixed in the same vector store — no cross-contamination between sources.

**Negative tests (question unrelated to any uploaded document):**

| # | Question | Expected | Result | Correct? |
|---|---|---|---|---|
| 13 | What does it taste like? | Refuse to answer | "This information is not available in the uploaded document.", sources: [] | ✅ |
| 14 | Who is the president of the USA? | Refuse to answer | "This information is not available in the uploaded document.", sources: [] | ✅ |
| 15 | How many genes does the rabies virus have? | Refuse to answer | "This information is not available in the uploaded document.", sources: [] | ✅ |

**15/15 total** — the threshold correctly separates in-domain from out-of-domain questions, even with an unrelated but plausible-sounding science question (#15).

## Known gaps / next steps

- Adaptive relevance threshold instead of a fixed value (see limitation above)
- Deduplication check before indexing (avoid re-indexing an already-uploaded source)
- No support for DOCX/PDF formats beyond plain text extraction (e.g. tables, images not handled)

## Stack

FastAPI · LangChain · ChromaDB · HuggingFace Embeddings (`all-MiniLM-L6-v2`) · Google Gemini · `trafilatura` · `python-docx2txt`
