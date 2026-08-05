# Text Generation with the Groq API

A hands-on notebook exploring LLM text generation through Groq's inference API. Covers PDF text extraction, basic prompting, few-shot prompt construction, and a side-by-side comparison of three open-weight models on an identical prompt.

## What it does

**PDF extraction.** Pulls raw text out of a lecture PDF with PyMuPDF, page by page, as a source of prompt material.

**Basic generation.** A `generateContent()` helper wraps the Groq chat completions endpoint with a fixed system message, `temperature=0.7`, and `max_tokens=300`, then runs it over a list of prompts.

**Few-shot prompting.** Builds a sentiment classification prompt with worked examples. This cell is worth reading for the result rather than the technique: one of the examples is deliberately mislabelled ("Trip was amazing" tagged as negative), and the model refuses to follow the pattern, flags the error, and returns corrected labels instead. A neat demonstration that instruction-tuned models weight their own judgement against inconsistent few-shot examples.

**Model comparison.** The same prompt is sent to three models to compare output style:

| Model | Character |
|---|---|
| `llama-3.1-8b-instant` | Fast, concise, straightforward prose |
| `llama-3.3-70b-versatile` | Fuller answers with concrete examples |
| `qwen/qwen3-32b` | Emits a visible `<think>` reasoning block before the answer |

The Qwen output is the most instructive difference: reasoning models return their scratchpad inline, so any downstream parsing has to strip it.

## Running it

```bash
pip install groq pymupdf
```

The notebook reads the API key from Colab's secrets manager via `userdata.get("Groq_API_Key")`, so no key is stored in the file. Running outside Colab means swapping that for an environment variable:

```python
import os
client = Groq(api_key=os.environ["GROQ_API_KEY"])
```

You will also need to point `pdf_file` at a PDF of your own.

## Scope

This is a learning exercise, not a library or an application. It is a record of working through the Groq API and prompt patterns, and the model comparison is a single prompt with one sample per model, so it shows stylistic differences rather than anything measurable about quality.
