# Cross-Lingual-LLM-Bias

[![Status](https://img.shields.io/badge/Status-Active_Research-blue.svg)]()
[![Paper](https://img.shields.io/badge/Paper-Pending_Publication-green.svg)]()

**Abstract:** This repository investigates the behavioral divergence and cultural alignment of open-source Large Language Models (LLMs) such as Llama 3 and Jais 30B when prompted in different languages, specifically examining variations in safety filters, personality, and knowledge depth between English, Russian, Kazakh, and Chinese.

<div align="center">
  <img src="similarity_heatmap.png" alt="Project Architecture" width="800">
</div>

# Cross-Lingual-LLM-Bias

[![Status](https://img.shields.io/badge/Status-Active_Research-blue.svg)]()
[![Report](https://img.shields.io/badge/Report-paper%2Freport.md-green.svg)](paper/report.md)

**Abstract:** This repository investigates how Large Language Models (LLMs) change their behavior depending on the language of the prompt. The same 50 questions were asked in English, Russian, Kazakh, and Chinese to Llama 3.1 8B, Llama 3.3 70B, and Jais 30B, and the 600 resulting responses were compared on safety disclaimers, answer length, assertiveness, and cross-lingual semantic similarity.

<div align="center">
  <img src="similarity_heatmap.png" alt="Cross-lingual semantic similarity heatmap" width="800">
</div>

## 🌱 Personal Motivation
As a multilingual speaker (Kazakh, Russian, English, Chinese) and an active language learner, I noticed early on that AI systems don't just "translate" thought; they change it depending on the language used. I started this research because I want to make sure that non-Western languages aren't penalized by the architectures of modern LLMs. Working with **Jais 30B** during this project opened my eyes to the potential of region-specific, culturally aligned models. My goal is to deepen my research into MENA-region NLP and work more extensively with the Jais ecosystem to bridge the linguistic divide in AI.

## 🎯 Research Questions
- Does the "safety curtain" shift when the same opinion question is asked in English and in Russian?
- Is there an "underrepresented tax": do models give shorter, less detailed answers in Kazakh than in English?
- Do models explain culturally tied concepts (e.g., "democracy", "traditional medicine") differently across languages?

## 🧠 Methodology
- **Benchmark:** 50 questions (15 factual, 15 opinion, 20 commonsense), each written in English, Russian, Kazakh, and Chinese: 200 prompts per model, 600 responses in total.
- **Models:**
  - Llama 3.1 8B and Llama 3.3 70B (Meta), queried through the Groq API with the same neutral system prompt: *"You are a helpful assistant. Answer the question directly and concisely."*
  - Jais 30B (Inception / MBZUAI), queried manually through [jaischat.ai](https://jaischat.ai/) and parsed with scripts/parse_manual.py.
- **Metrics:**
  - Answer length in words (Chinese characters for Chinese).
  - Safety disclaimers and assertiveness markers, detected with language-specific regex patterns.
  - Cross-lingual semantic similarity: cosine similarity of paraphrase-multilingual-MiniLM-L12-v2 embeddings for all 6 language pairs.
- **Tech stack:** Python, Groq API, sentence-transformers, pandas, NumPy, matplotlib, seaborn, Streamlit, Plotly.

## 📊 Key Findings
- **The "Safety Curtain":** Llama 3.1 8B added at least one safety disclaimer to **80% of opinion answers in Russian (12 of 15) vs. 27% in English (4 of 15)**. Llama 3.3 70B shows the same direction on a smaller scale (33% vs. 13%).
- **The "Underrepresented Tax":** Jais 30B answers in Kazakh averaged **22 words vs. 48 words in English**, about 50% shorter.

All numbers can be reproduced from results/analysis_summary.csv.

## ⚠️ Limitations
- **Small sample:** 15 opinion questions per language and no statistical significance testing, so the findings are exploratory.
- **Keyword-based detection:** disclaimers are counted with regex patterns, which can miss paraphrased disclaimers or produce false positives.
- **Different collection method for Jais:** Jais responses were collected through a web chat rather than an API, so the system prompt and generation settings differ from the Llama runs.
- **Kazakh similarity scores:** the embedding model was not trained on Kazakh, so similarity scores for Kazakh language pairs are unreliable and are not reported as findings.
- **Gemini:** a Gemini 2.0 Flash Lite run was attempted but returned no responses because of API quota limits, so it is excluded from the analysis.
## 🔁 Reproduce
```bash
cp .env.example .env      # add your GROQ_API_KEY
bash run_pipeline.sh      # query models, import Jais responses, analyze, visualize
streamlit run app.py      # interactive dashboard
```

## 📬 Contact & Research Interests
I am actively seeking to expand this research into MENA-region NLP and culturally aligned foundation models. My primary academic aspiration is to join the research community and contribute to the development of the **Jais** ecosystem and similar initiatives.

If you are a professor, researcher, or engineer working on multilingual LLM alignment, I would be glad to connect.

**Arsen Bakhitbekov**: [bakhitbekovv@gmail.com](mailto:bakhitbekovv@gmail.com)

## License
MIT
