#  Prompt Improver

**Prompt Improver** is a lightweight Streamlit application that helps you transform raw prompts into **professional, structured, and high-quality prompts**, then compare model outputs **before and after improvement**.

---

##  Features

- Guided 3-step workflow:
  1. Choose a prompt (dataset example or manual input)
  2. Improve the prompt (role, goal, tone, constraints)
  3. Compare LLM outputs (before / after)
- Professional prompt upgrade engine
- Prompt quality heuristic score
- Supports Groq & OpenAI providers
- Automatic fallback if a Groq model is deprecated
- Clean UI optimized for demos and presentations

---

##  Use Cases

- Prompt engineering demonstrations
- LLM evaluation (before / after prompt quality)
- Training & workshops
- Portfolio / PFE / DevOps & AI demos

---

## 🖥️ Run Locally

### 1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API keys

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY="your_groq_key"
OPENAI_API_KEY="your_openai_key"
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push the project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. New App → select this repo
4. Set secrets in the web UI
5. Deploy 🚀

---

## 👤 Author

**Dhafer Bouthelja**
Cloud & DevOps Engineer • Software Quality
🔗 LinkedIn: [https://www.linkedin.com/in/bouthelja-dhafer-116681a0/](https://www.linkedin.com/in/bouthelja-dhafer-116681a0/)

---

## 📄 License

This project is for educational and demonstration purposes.

````


