# Emergency Text Classifier

A machine learning model that classifies text as **emergency** or **non-emergency** in real-time. Built to demonstrate how AI can help triage 911-style reports and reduce the burden of non-emergency calls on dispatch systems.

**[Try the Live Demo](https://parthavfulzele-emergency-classifier-app-hdp7k1.streamlit.app/)**

---

## The Problem

### 911 systems are overwhelmed by non-emergency calls

Every year, approximately **240 million calls** are made to 911 in the United States. A significant portion of these are not actual emergencies.

| Statistic | Detail |
|-----------|--------|
| **35-50%** of 911 calls | Are estimated to be non-emergencies across U.S. jurisdictions |
| **62.6%** of calls | Are noncriminal in nature (Vera Institute study across 9 cities) |
| **82%** of 911 centers | Are understaffed (NENA, 2023) |
| **$3.4 billion** | Annual operating cost of U.S. 911 systems |

### What counts as a non-emergency call?

People routinely call 911 for issues that don't require an immediate emergency response:
- Noise complaints and neighbor disputes
- Parking violations and minor traffic issues
- Lost pets or stray animals
- Internet/cable service outages
- Property damage reports (no injuries)
- General city service questions
- Minor fender benders with no injuries

### The real-world impact

**Delayed response times:** In New York City, average response time for life-threatening calls increased from **5 minutes 53 seconds to 7 minutes 23 seconds** over a roughly 3-year period, partly due to call volume strain.

**Dispatcher burnout:** Between **18-33% of dispatchers** show signs of clinical PTSD, compared to **1.4%** in the general population. High volumes of non-critical calls contribute to fatigue, compassion fatigue, and turnover.

**Staffing crisis:** NENA's 2023 survey found that **82% of 911 centers are understaffed**, with many operating at 10-20% below minimum staffing levels. Non-emergency calls amplify this strain.

**Financial burden:** Each 911 call costs the system resources regardless of severity. With over 80 million non-emergency calls annually (at a conservative 35% estimate), the waste is substantial.

### What's being done about it

- **311 systems:** Cities like New York, Chicago, and Los Angeles operate 311 lines for non-emergency municipal services, but many callers still default to 911.
- **CAHOOTS (Eugene, OR):** A community-based crisis response program that diverts non-emergency calls to trained responders instead of police. It handles ~20% of call volume and saves an estimated **$8.5 million/year** while operating at **2.3% of the police budget**.
- **AI triage tools:** Companies like Carbyne and Prepared911 are deploying AI-assisted dispatch tools that help classify and prioritize calls, reporting time savings of **15-20 minutes per incident** in early deployments.

---

## This Project

This classifier is a proof-of-concept demonstrating that **text-based emergency classification can be done locally, without APIs, with high accuracy** using traditional ML techniques.

### How it works

1. **Input:** A text description of a situation (e.g., "someone collapsed and isn't breathing" or "my neighbor's dog keeps barking")
2. **Preprocessing:** Lowercase, strip punctuation, normalize whitespace
3. **Feature extraction:** TF-IDF vectorization with word n-grams (1-2) and character n-grams (3-5) for typo robustness
4. **Classification:** SGDClassifier (Stochastic Gradient Descent) selected via GridSearchCV across multiple classifiers
5. **Output:** Emergency / Non-Emergency label with confidence score

### Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 94.2% |
| **Cross-Validation Accuracy** | 92.9% |
| **Emergency F1** | 0.94 |
| **Non-Emergency F1** | 0.94 |

Trained on a **601-example dataset** (300 emergency, 301 non-emergency) covering:
- Fire, medical, crime, natural disasters, industrial accidents, water rescues
- Noise complaints, parking, permits, animal control, city services, HOA issues
- Edge cases: typos, slang, short inputs, past-tense non-emergencies, ambiguous phrasing

### Model selection

GridSearchCV evaluated three classifiers with 5-fold stratified cross-validation:
- Logistic Regression (varied C)
- SGDClassifier (varied alpha) — **winner**
- LinearSVC (varied C)

Character n-grams (3-5) were added alongside word n-grams to handle misspelled or informal text like "thers a fir in the bilding" or "somone is not bretahing."

---

## Run Locally

```bash
# Clone
git clone https://github.com/parthavfulzele/emergency-classifier.git
cd emergency-classifier

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train the model
python model/train.py

# Launch the dashboard
streamlit run app.py
```

Opens at `http://localhost:8501`. The model auto-trains on first launch if `classifier.pkl` doesn't exist.

## Project Structure

```
emergency-classifier/
├── app.py                # Streamlit dashboard
├── requirements.txt      # Python dependencies
├── data/
│   └── dataset.csv       # 601 labeled examples
└── model/
    ├── train.py           # Training script (GridSearchCV, TF-IDF, SGD)
    └── metrics.json       # Saved performance metrics
```

## Tech Stack

- **scikit-learn** — TF-IDF vectorization, SGDClassifier, GridSearchCV
- **pandas** — data handling
- **Streamlit** — interactive web dashboard
- **Python 3.10+** — no external APIs, fully local

---

## Sources

- NENA (National Emergency Number Association) — 911 staffing and call volume statistics
- Vera Institute of Justice — analysis of 911 call composition across 9 U.S. cities
- CAHOOTS (White Bird Clinic, Eugene OR) — alternative response program data
- NYC Mayor's Management Report — response time trends
