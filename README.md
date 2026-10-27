# Cross-Lingual-LLM-Bias

[![Status](https://img.shields.io/badge/Status-Active_Research-blue.svg)]()


**Abstract:** This repository investigates how Large Language Models (LLMs) change their behavior depending on the language of the prompt. The same 50 questions were asked in English, Russian, Kazakh, and Chinese to Llama 3.1 8B, Llama 3.3 70B, and Jais 30B, and the 600 resulting responses were compared on hedging phrases, answer length, assertiveness, and cross-lingual semantic similarity.

<div align="center">
  <img src="similarity_heatmap.png" alt="Cross-lingual semantic similarity heatmap" width="800">
  <br>
  <em>Kazakh pairs shown for completeness; not reliable, see Limitations.</em>
</div>

## 🌱 Personal Motivation
I speak Kazakh and Russian natively, study in English and am learning Chinese. I noticed that AI systems don't just "translate" a question; the answer itself can change with the language. I started this project to measure how much, with a focus on Kazakh, a language that most models see very little of during training.

## 🎯 Research Questions
- Does the "safety curtain" shift when the same opinion question is asked in English and in Russian?
- Is there an "underrepresented tax": do models give shorter, less detailed answers in Kazakh than in English?
- Do models explain culturally tied concepts (e.g., "democracy", "traditional medicine") differently across languages?

## 🧠 Methodology
- **Benchmark:** 50 questions (15 factual, 15 opinion, 20 commonsense), each written in English, Russian, Kazakh, and Chinese: 200 prompts per model, 600 responses in total.
- **Models:**
  - Llama 3.1 8B and Llama 3.3 70B (Meta), queried through the Groq API with the same neutral system prompt: *"You are a helpful assistant. Answer the question directly and concisely."*
  - Jais 30B (Inception / MBZUAI), queried manually through [jaischat.ai](https://jaischat.ai/) and parsed with `scripts/parse_manual.py`.
- **Metrics:**
  - Answer length in words (Chinese characters for Chinese). Kazakh and English are also compared in characters, normalized by how long the same question is in each language.
  - Hedging phrases (e.g., "however", "it depends on") and assertiveness markers, detected with language-specific regex patterns. English and Russian hedging patterns are mirrored one to one in `scripts/analyze_responses.py`; `scripts/reanalysis.py` compares them with the original, asymmetric patterns.
  - Cross-lingual semantic similarity: cosine similarity of `paraphrase-multilingual-MiniLM-L12-v2` embeddings for all 6 language pairs.
- **Tech stack:** Python, Groq API, sentence-transformers, pandas, NumPy, matplotlib, seaborn, Streamlit, Plotly.

## 📊 Key Findings
- **Jais 30B cannot answer in Kazakh:** 49 of 50 Kazakh replies repeat the question, and **39 of 50 end with the same meaningless phrase**. The rest are nonsense too (one says water boils at 30 to 40 degrees).
- **The "Underrepresented Tax" depends on the model:** after adjusting for question length, Llama 3.1 8B answered more briefly in Kazakh than in English on **34 of 50 questions** (sign test p = 0.015). Llama 3.3 70B showed no consistent gap (23 of 50, p = 0.67).
- **No "Safety Curtain" after a fix:** an earlier version of this README reported hedging in 12 of 15 Russian vs. 4 of 15 English opinion answers for Llama 3.1 8B. That gap came from the detector: the Russian list matched every "однако" ("however"), the English list only "however, it's/there". With mirrored patterns the counts are 12 vs. 9 (McNemar p = 0.45), so there is no clear difference.

All numbers can be reproduced with `python scripts/reanalysis.py .` from the repository root.

## ⚠️ Limitations
- **Small sample:** 15 opinion and 50 total questions per language, so the findings are exploratory even where a test is significant.
- **Keyword-based detection:** hedging is counted with regex patterns, which can miss paraphrases or produce false positives. Patterns must have the same breadth in every language, or the comparison measures the detector instead of the model.
- **Different collection method for Jais:** Jais responses were collected through a web chat rather than an API, so the system prompt and generation settings differ from the Llama runs.
- **Kazakh similarity scores:** Kazakh is not among the languages the embedding model was trained to compare (its paraphrase training covers about 50 languages), so similarity scores for Kazakh language pairs are unreliable and are not reported as findings.
- **Gemini:** a Gemini 2.0 Flash Lite run was attempted but returned no responses because of API quota limits, so it is excluded from the analysis.

## 🔁 Reproduce
```bash
cp .env.example .env      # add your GROQ_API_KEY
bash run_pipeline.sh      # query models, import Jais responses, analyze, visualize
streamlit run app.py      # interactive dashboard
```

## 📬 Contact & Research Interests
I want to continue this work on evaluating language models in low-resource languages, starting with Kazakh, and eventually help build models that handle Kazakh well.

If you are a professor, researcher, or engineer working on multilingual evaluation or alignment, I would be glad to connect.

**Arsen Bakhitbekov**: [bakhitbekovv@gmail.com](mailto:bakhitbekovv@gmail.com)

## License
MIT
