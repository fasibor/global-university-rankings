# 🎓 Global University Rankings Dashboard

[![Hugging Face Space](https://img.shields.io/badge/Deploy-HuggingFace-blue?style=flat-square)](https://huggingface.co/spaces/fasibor/global-university-rankings)

Interactive dashboard to explore **world university rankings** across multiple dimensions, built with **Streamlit** and deployed on **Hugging Face Spaces**.



## Overview

This dashboard allows you to:

- Visualize universities by **world rank buckets**.
- Compare **national vs global rankings**.
- Analyze **ranking dimensions coverage**.
- Explore **top countries by university count** and **dominance in top 100**.
- Interact with charts dynamically, including **scatter plots, bar charts, and lollipop charts**.
- Gain **auto-generated insights** from the data.


## Features

- **Streamlit** interactive charts
- **Plotly**-based visualizations
- **Power BI–style spacing and layout**
- **Dark theme support**
- **Auto-generated insights and stats**
- **Docker-ready for Hugging Face Spaces**



## How to Run Locally

1. Clone this repository:

```bash
git clone https://github.com/fasibor/global-university-rankings.git
cd global-university-rankings
```
2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

3. Install dependencies:
pip install -r requirements.txt

4. Run the app:
streamlit run app.py

## Docker Deployment (Hugging Face)
This app is Docker-ready. Hugging Face will automatically build the Docker container with your Dockerfile and requirements.txt.
Make sure runtime.txt specifies python-3.10 and README.md includes the HF Space configuration.

## File Structure
```text
.
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker setup for Hugging Face
├── README.md               # Project description
├── data/                   # Dataset (optional)
├── .streamlit/             # Streamlit theme configuration
└── screenshots/            # Dashboard screenshots
```

## Key Insights from the Dashboard

1. **World Rank Distribution:**  
   The majority of universities are concentrated beyond the 500+ rank bucket, highlighting a long-tail structure in global rankings. Marginal improvements at the top end lead to disproportionately large changes in position, emphasizing the extreme competitiveness of elite rankings.

2. **Top Countries – Quantity vs Quality:**  
   While **China** has the highest number of ranked universities overall, the **United States** dominates the top 100 segment, accounting for nearly half of elite institutions. This contrast underscores that volume alone does not guarantee top-tier performance, revealing a structural gap between national higher-education scale and global excellence.

3. **Composite Score as a Predictor:**  
   The **composite rank score** exhibits a strong inverse relationship with world rank (correlation ≈ 0.80), confirming its effectiveness as a meaningful aggregate metric. Institutions with higher composite scores consistently occupy better global positions, validating the composite score as a robust indicator of overall performance.

4. **Ranking Dimension Strength and Drivers:**  
   **Research** and **Education** emerge as the most influential and consistent dimensions, while **Employability** shows greater variability and **Faculty** metrics contribute less to global differentiation. Correlation heatmaps reinforce Research performance as the primary driver of elite status.

5. **Top 100 vs Other Universities:**  
   Top 100 universities consistently outperform others in Research and Employability, with differences most pronounced in these dimensions. Strong performance in these areas is critical for achieving elite global ranking.

6. **Rank Gap Anomalies:**  
   About **19.8% of universities** show a rank inconsistency (gap > 1658.5). These anomalies may reflect specialization, unusual performance patterns, or evaluation inconsistencies, highlighting the need for careful interpretation of rankings.

7. **Ranking Coverage and Performance:**  
   Universities evaluated across more ranking dimensions generally achieve better global rankings, suggesting that comprehensive assessment reflects higher institutional maturity and balanced performance.

8. **National vs Global Rankings:**  
   Strong domestic performance does not always translate to elite global rank. Some universities excel nationally but occupy mid-tier global positions, emphasizing the competitive international context of higher education.

**Overall Conclusion:**  
Elite global status is driven primarily by **research excellence** and **comprehensive dimension performance**, while country-level representation and sheer volume of institutions alone are insufficient to secure top-tier positioning.


## Links

Hugging Face Space: [Click here](https://huggingface.co/spaces/fasibor/global-university-rankings)

## Tech Stack

- Python 3.10

- Streamlit

- Plotly

- Pandas

- Docker

- Hugging Face Spaces

## License
This project is licensed under the MIT License. See the LICENSE file for details.
