# 🎵 Spotify Music Analysis & Interactive Dashboard

An interactive, Spotify-themed web application built with **Streamlit** and **Python** that visualizes music trends, track audio features, and success-tier predictions using custom data analytics, machine learning models, and a polished Spotify dark-mode aesthetic.

---

## 📊 Dashboard Preview

![KPI Summary](assets/spotify_kpi.png)
![Popularity Score Distribution](assets/popularity_score_distribution.png)
![Success Tier Distribution](assets/success_tier_contribution.png)
![Track Scorer](assets/track_score.png)

> Screenshots are stored in the `assets/` folder. If you add or rename any images, update the paths above to match exactly (including file extension and case).

---

## ✨ Features

- **Spotify Dark Aesthetic:** Customized UI with Spotify's signature black-to-dark-grey gradient, green glowing accents (`#1DB954`), pill-shaped buttons, and custom metric cards.
- **Track Scorer:** Interactively score a new or hypothetical track's popularity and success tier from its audio features and artist context, with a probability breakdown and an audio-feature radar chart vs. genre average.
- **Genre Trends & Forecast:** Explore average popularity by genre over time, a one-year-ahead trend forecast, and an energy-vs-valence scatter landscape.
- **Driver Importance:** See which features drive the regression and classification models, plus an audio-feature correlation heatmap.
- **Catalog & Export:** Browse the filtered catalog, view top artists by track count, score the entire filtered catalog, and download predictions as CSV.
- **Dynamic Filtering:** Sidebar filters for genre, artist country, and audio indicator, applied live across every tab.

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Frontend / Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization:** [Plotly](https://plotly.com/)
- **Modeling:** scikit-learn (regression + classification pipelines, trained via `src.training`)
- **Styling:** Custom CSS injection via `st.markdown()`

---

## 📁 Project Structure

```
Spotify-Music-Analysis/
├── assets/                     # README screenshots
├── dashboard/
│   └── app.py                  # Streamlit dashboard entry point
├── src/
│   ├── prediction/              # Scoring logic (predict.py)
│   ├── training/                 # Model training (mlflow_tracking.py, etc.)
│   └── utils/                    # Config/helper/model-loading utilities
├── data/
│   └── processed/                # Model-ready CSVs
├── reports/                      # Leaderboards, driver importance CSVs
├── models/                       # Trained model artifacts
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

Follow these steps to run the project locally on your machine.

### 1. Prerequisites

Ensure you have **Python 3.9+** and **Git** installed on your system.

### 2. Clone the Repository

```bash
git clone https://github.com/Shubham03-hub/Spotify-Music-Analysis.git
cd Spotify-Music-Analysis
```

### 3. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Models (first run only)

The dashboard is a pure consumption layer — it reads pre-trained models and pre-computed CSVs rather than training anything itself. If `models/` is empty, train first:

```bash
python -m src.training.mlflow_tracking
```

### 6. Run the Dashboard

```bash
streamlit run dashboard/app.

## 📊 Dashboard Preview

![KPI Summary](assets/kpi_summary.png)
![Popularity Score Distribution](assets/popularity.png)
```

Streamlit will start a local server and open the dashboard automatically at:

```
http://localhost:8501
```

If it doesn't open automatically, paste that URL into your browser.

---

## 📸 Updating Screenshots

To refresh the images shown in this README:

1. Take a new screenshot of the running dashboard.
2. Save it into the `assets/` folder (create the folder if it doesn't exist).
3. Reference it in this README with a relative path, e.g. `![Alt text](assets/your-image.png)`.
4. Commit and push:

```bash
git add assets/ README.md
git commit -m "Update dashboard screenshots"
git push origin main
```

---

## 🤝 Contributing

Issues and pull requests are welcome. If you spot a bug or have an idea for a new chart or feature, feel free to open an issue.

---

## 📄 License

This project is available for personal and educational use. Add a `LICENSE` file to this repository if you'd like to specify formal usage terms (e.g. MIT, Apache 2.0).