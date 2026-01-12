
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Global University Rankings EDA",
    layout="wide"
)

st.title("🌍 Global University Rankings – Interactive EDA Dashboard")
st.caption("Exploratory Data Analysis with Automated Insights")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cwur_2025_rankings_cleaned.csv")

df = load_data()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("Filters")

country_options = ["All"] + sorted(df['Location'].dropna().unique())
selected_country = st.sidebar.selectbox("Country", country_options)

bucket_options = ["All"] + sorted(df['World Rank Bucket'].dropna().unique())
selected_bucket = st.sidebar.selectbox("World Rank Bucket", bucket_options)

rank_gap_options = ["All"] + sorted(df['Rank Gap Flag'].dropna().unique())
selected_gap = st.sidebar.selectbox("Rank Gap Flag", rank_gap_options)

max_rank = st.sidebar.slider(
    "Maximum World Rank",
    int(df['World Rank'].min()),
    int(df['World Rank'].max()),
    int(df['World Rank'].max())
)

# -------------------------------
# APPLY FILTERS
# -------------------------------
@st.cache_data
def filter_data(df, country, bucket, gap, max_rank):
    filtered = df.copy()
    if country != "All":
        filtered = filtered[filtered['Location'] == country]
    if bucket != "All":
        filtered = filtered[filtered['World Rank Bucket'] == bucket]
    if gap != "All":
        filtered = filtered[filtered['Rank Gap Flag'] == gap]
    filtered = filtered[filtered['World Rank'] <= max_rank]
    return filtered

filtered_df = filter_data(df, selected_country, selected_bucket, selected_gap, max_rank)

# -------------------------------
# KPI CALCULATIONS
# -------------------------------
top_100_count = (filtered_df['World Rank'] <= 100).sum()
top_10_percent_cutoff = filtered_df['World Rank'].quantile(0.10)
top_10_percent_count = (filtered_df['World Rank'] <= top_10_percent_cutoff).sum()
best_inst = filtered_df.loc[filtered_df['World Rank'].idxmin(), 'Institution']
worst_inst = filtered_df.loc[filtered_df['World Rank'].idxmax(), 'Institution']
full_dim_pct = round(
    (filtered_df['Ranking Dimensions Count'] == filtered_df['Ranking Dimensions Count'].max()).mean() * 100, 1
)

# -------------------------------
# UNIVERSITY NAME ABBREVIATION
# -------------------------------
short_names = {
    "Harvard University": "Harvard Uni",
    "Massachusetts Institute of Technology": "MIT",
    "University of Oxford": "Oxford Uni",
    "California Institute of Technology": "Caltech",
    "Stanford University": "Stanford Uni",
    "University of Cambridge": "Cambridge Uni",
    "Imperial College London": "Imperial College",
    "ETH Zurich": "ETH Zurich",
    "University of Chicago": "Chicago Uni",
    "University College London": "UCL",
    "Obafemi Awolowo University": "OAU",
    "Başkent University": "Başkent Uni",
    "University of Ibadan": "UI"
}

def get_short_name(name):
    return short_names.get(name, name)

# -------------------------------
# KPI DISPLAY
# -------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Top 100 Institutions", top_100_count)
c2.metric("Top 10% Universities", top_10_percent_count)
c3.metric("Best Ranked Institution", get_short_name(best_inst))
c4.metric("Worst Ranked Institution", get_short_name(worst_inst))
c5.metric("Full Dimension Coverage (%)", f"{full_dim_pct}%")

st.divider()

# -------------------------------
# CHART 1: WORLD RANK DISTRIBUTION
# -------------------------------
rank_dist = filtered_df["World Rank Bucket"].value_counts().sort_index().reset_index()
rank_dist.columns = ["World Rank Bucket", "Count"]

fig = px.scatter(rank_dist, x="World Rank Bucket", y="Count", size="Count",
                 title="Distribution of Universities by World Rank Bucket")
for _, row in rank_dist.iterrows():
    fig.add_shape(type="line", x0=row["World Rank Bucket"], y0=0, x1=row["World Rank Bucket"], y1=row["Count"],
                  line=dict(width=2))
fig.update_traces(marker=dict(opacity=0.85),
                  hovertemplate="<b>%{x}</b><br>Universities: %{y}<extra></extra>")
fig.update_layout(showlegend=False, xaxis_title="World Rank Bucket", yaxis_title="Number of Universities")
st.plotly_chart(fig, use_container_width=True)

top_bucket = rank_dist.loc[rank_dist["Count"].idxmax(), "World Rank Bucket"]
top_count = rank_dist["Count"].max()
lowest_bucket = rank_dist.loc[rank_dist["Count"].idxmin(), "World Rank Bucket"]
lowest_count = rank_dist["Count"].min()
total_universities = rank_dist["Count"].sum()
top_pct = round((top_count / total_universities) * 100, 1)

st.info(
    f"🔍 **Insight**\n\n"
    f" The **{top_bucket}** bucket has the highest concentration with **{top_count} universities**, "
    f"representing **{top_pct}%** of all ranked institutions.\n"
    f" The **{lowest_bucket}** bucket is the most exclusive, with only **{lowest_count} universities**."
)

# -------------------------------
# CHART 2: TOP COUNTRIES BY UNIVERSITY COUNT
# -------------------------------
country_counts = filtered_df['Location'].value_counts().head(10)
fig = px.bar(country_counts, title="Top 10 Countries by Number of Universities")
st.plotly_chart(fig, use_container_width=True)

st.info(f"🌍 Insight: **{country_counts.index[0]}** hosts the most universities.")

# -------------------------------
# CHART 3: COUNTRIES DOMINATING TOP 100
# -------------------------------
@st.cache_data
def top100_countries(df):
    return (df[df['World Rank'] <= 100].groupby('Location')['Institution']
            .count().sort_values(ascending=False).head(10))

top100_country = top100_countries(filtered_df)
fig = px.bar(top100_country, title="Top 10 Countries by Top 100 Universities")
st.plotly_chart(fig, use_container_width=True)
if not top100_country.empty:
    st.info(f"🏆 Insight: **{top100_country.index[0]}** dominates with **{top100_country.iloc[0]}** top 100 universities.")

# -------------------------------
# CHART 4: COMPOSITE SCORE VS WORLD RANK
# -------------------------------
fig = px.scatter(
    filtered_df,
    x="Composite Rank Score",
    y="World Rank",
    trendline="ols",
    title="Composite Rank Score vs World Rank"
)
st.plotly_chart(fig, use_container_width=True)

corr = filtered_df['Composite Rank Score'].corr(filtered_df['World Rank'])
st.info(f"📉 Insight: Correlation of **{corr:.2f}** confirms that higher composite scores align with better world rankings.")

# -------------------------------
# CHART 5A: RANKING DIMENSION STRENGTH
# -------------------------------
dim_cols = ['Education Rank Status', 'Research Rank Status', 'Employability Rank Status']

@st.cache_data
def compute_dim_strength(df, dim_cols):
    dim_strength = df[dim_cols].apply(pd.Series.value_counts).fillna(0)
    dim_strength = dim_strength.reset_index().rename(columns={'index': 'Status'})
    dim_long = dim_strength.melt(id_vars='Status', var_name='Dimension', value_name='Count')
    variation = dim_strength[dim_cols].std()
    return dim_strength, dim_long, variation

dim_strength, dim_long, variation = compute_dim_strength(filtered_df, dim_cols)

fig = px.bar(
    dim_long,
    x='Status',
    y='Count',
    color='Dimension',
    barmode='stack',
    title="Strength Distribution Across Ranking Dimensions",
    color_discrete_map={
        'Education Rank Status': '#636EFA',  # blue
        'Research Rank Status': '#EF553B',   # red
        'Employability Rank Status': '#00CC96'  # green
    }
)

st.plotly_chart(fig, use_container_width=True)

most_consistent = variation.idxmin().replace(' Rank Status','')
most_variable = variation.idxmax().replace(' Rank Status','')

st.info(f"🎓Insight: {most_consistent} is the most consistent dimension across universities, while {most_variable} shows the most variation in performance.")

# -------------------------------
# CHART 5B: CORRELATION HEATMAP OF RANKINGS AND SCORES
# -------------------------------
numeric_cols = [
    'World Rank', 'Education Rank', 'Research Rank', 
    'Faculty Rank', 'Employability Rank', 'Composite Rank Score', 'Score'
]

@st.cache_data
def compute_corr_matrix(df, numeric_cols):
    return df[numeric_cols].corr()

corr_matrix = compute_corr_matrix(filtered_df, numeric_cols)

fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    origin='upper',
    aspect="auto"
)
fig.update_layout(title='Correlation Heatmap of Rankings and Scores')
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# INSIGHT
# -------------------------------
strong_corrs = [(c1, c2, v) 
                for i, c1 in enumerate(corr_matrix.columns) 
                for j, c2 in enumerate(corr_matrix.columns) 
                if j > i and abs((v := corr_matrix.iloc[i, j])) >= 0.7]

if strong_corrs:
    c1, c2, val = max(strong_corrs, key=lambda x: abs(x[2]))
    if val > 0:
        insight = f"🔬 Insight: **{c1}** and **{c2}** move hand-in-hand, showing strong {c2.lower()} when {c1.lower()} is high."
    else:
        insight = f"⚠️ Insight: **{c1}** and **{c2}** move in opposite directions, suggesting high {c1.lower()} may coincide with lower {c2.lower()}."
else:
    insight = "No strong correlations (>|0.7|) found among ranking dimensions."

st.info(insight)

# -------------------------------
# CHART 6: TOP 100 VS OTHERS (RESEARCH)
# -------------------------------

# -----------------------------
# 1️⃣ Create a descriptive group column
# -----------------------------
filtered_df['University Group'] = filtered_df['World Rank'].apply(
    lambda x: "Top 100 Universities" if x <= 100 else "Other Universities"
)

# -----------------------------
# 2️⃣ Calculate median Research Rank for Top 100 universities
# -----------------------------
median_top100 = filtered_df.loc[filtered_df['University Group']=="Top 100 Universities", 'Research Rank'].median()

# -----------------------------
# 3️⃣ Create the box plot
# -----------------------------
fig = px.box(
    filtered_df,
    x="University Group",
    y="Research Rank",
    color="University Group",
    color_discrete_map={
        "Top 100 Universities": "#4C6EF5",
        "Other Universities": "tomato"
    },
    title="Research Performance: Top 100 vs Other Universities",
    labels={
        "University Group": "University Group",
        "Research Rank": "Research Rank (lower is better)"
    }
)

# Clean layout
fig.update_traces(showlegend=False)
fig.update_layout(
    yaxis_title='Research Rank (lower is better)',
    xaxis_title='University Group',
)


# -----------------------------
# 5️⃣ Display chart in Streamlit
# -----------------------------
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 6️⃣ Optional info box
# -----------------------------
st.info(
    "🔬 Top 100 universities consistently outperform others in Research, "
    "highlighting research as a key factor for elite ranking."
)



# -------------------------------
# CHART 7: RANK GAP ANALYSIS
# -------------------------------

# -----------------------------
# 1️⃣ Ranking columns
# -----------------------------
rank_cols = ['World Rank', 'Education Rank', 'Research Rank', 'Employability Rank']

# -----------------------------
# 2️⃣ Compute rank range and dynamic threshold
# -----------------------------
filtered_df['Rank Range'] = filtered_df[rank_cols].max(axis=1) - filtered_df[rank_cols].min(axis=1)
threshold = filtered_df['Rank Range'].mean() + filtered_df['Rank Range'].std()

# -----------------------------
# 3️⃣ Flag inconsistent universities
# -----------------------------
filtered_df['Rank Inconsistency Flag'] = (filtered_df['Rank Range'] > threshold).astype(int)

# -----------------------------
# 4️⃣ Prepare data for bar chart
# -----------------------------
gap_counts = filtered_df['Rank Inconsistency Flag'].value_counts()
gap_df = gap_counts.reset_index()
gap_df.columns = ['Flag', 'Count']
gap_df['Status'] = gap_df['Flag'].map({0: 'Consistent', 1: 'Inconsistent'})

# -----------------------------
# 5️⃣ Create bar chart
# -----------------------------
fig = px.bar(
    gap_df,
    x='Status',
    y='Count',
    color='Status',
    color_discrete_map={'Consistent': '#4C6EF5', 'Inconsistent': 'tomato'},
    title=f"Universities by Rank Consistency (Gap > {round(threshold,1)} ranks considered inconsistent)"
)

fig.update_layout(
    yaxis_title='Number of Universities',
    xaxis_title='',
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 6️⃣ Display insight
# -----------------------------
gap_pct = round((gap_counts.get(1,0)/len(filtered_df))*100,1)
st.info(
    f"⚠️ About **{gap_pct}%** of universities show a **rank inconsistency**: "
    f"their ranking across dimensions differs significantly from overall performance "
    f"(gap > {round(threshold,1)} ranks automatically detected as significant)."
)


# -------------------------------
# CHART 8: COVERAGE VS PERFORMANCE
# -------------------------------
fig = px.scatter(filtered_df, x="Ranking Dimensions Count", y="World Rank", trendline="ols",
                 title="Ranking Dimension Coverage vs World Rank")
st.plotly_chart(fig, use_container_width=True)

st.info("📊 Insight: More ranking dimensions evaluated generally correlate with better global rank.")

# -------------------------------
# CHART 9: AVERAGE DIMENSION RANKS (TOP 100 VS OTHERS)
# -------------------------------

dims = ['Education Rank', 'Research Rank', 'Faculty Rank', 'Employability Rank']

# Use descriptive group column instead of Top100
comparison = filtered_df.groupby('University Group')[dims].mean().T.reset_index()
comparison = comparison.rename(columns={'index': 'Dimension'})

comparison_long = comparison.melt(
    id_vars='Dimension',
    var_name='University Group',
    value_name='Average Rank'
)

# Color mapping will now use 'University Group' values
color_map = {
    'Top 100 Universities': '#4C6EF5',   # blue (elite)
    'Other Universities': 'tomato'      # tomato (comparison)
}

# Overall average rank reference
overall_avg = comparison_long['Average Rank'].mean()

# Plot
fig = px.bar(
    comparison_long,
    x='Dimension',
    y='Average Rank',
    color='University Group',
    barmode='group',
    title="Average Dimension Ranks: Top 100 vs Other Universities",
    color_discrete_map=color_map,
    hover_data={
        'Average Rank': ':.1f'
    }
)

# Axis labels
fig.update_yaxes(title="Average Rank (Lower = Better)")
fig.update_xaxes(title="Ranking Dimension")

# Reference line
fig.add_hline(
    y=overall_avg,
    line_dash="dot",
    line_color="gray",
    annotation_text="Overall Average Rank",
    annotation_position="top right"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Insight
# -------------------------------
gap = comparison.set_index('Dimension')
gap_size = (gap['Top 100 Universities'] - gap['Other Universities']).abs().sort_values(ascending=False)
top_dims = gap_size.head(2).index.tolist()

st.info(
    f"🚀Insight: The widest ranking gaps between Top 100 and other universities appear in "
    f"{top_dims[0]} and {top_dims[1]}, showing these dimensions most clearly separate elite institutions."
)


# -------------------------------
# Explanatory Note
# -------------------------------
st.caption("ℹ️ Note: Ranking metrics are inverse — lower rank values indicate better performance.")


# -------------------------------
# CHART 10: NATIONAL RANK VS WORLD RANK
# -------------------------------
fig = px.scatter(filtered_df, x="National Rank", y="World Rank", title="National Rank vs World Rank")
st.plotly_chart(fig, use_container_width=True)

st.info("🌐 Insight: Strong national performance does not always equal top global performance, highlighting international competition.")

# -------------------------------
# INSTITUTIONS TABLE WITH FILTERS & AUTO INSIGHTS
# -------------------------------
st.subheader("Institutions Table")
if not filtered_df.empty:
    best_inst_row = filtered_df.loc[filtered_df['World Rank'].idxmin()]
    worst_inst_row = filtered_df.loc[filtered_df['World Rank'].idxmax()]

    
    st.info(
        f"🔹 **Best Institution:** {get_short_name(best_inst_row['Institution'])} (World Rank: {best_inst_row['World Rank']})\n"
        f"🔹 **Worst Institution:** {get_short_name(worst_inst_row['Institution'])} (World Rank: {worst_inst_row['World Rank']})"
        
    )

table_cols = [
    'World Rank', 'World Rank Bucket', 'Institution', 'Location', 'National Rank',
    'Education Rank', 'Employability Rank', 
    'Faculty Rank', 'Research Rank'
    
]

st.dataframe(filtered_df[table_cols].sort_values('World Rank').reset_index(drop=True), height=500)


# -------------------------------
# FOOTER
# -------------------------------
st.markdown(
    f"""
    <hr>
    <p style="text-align:center; font-size:12px; color:#6c757d;">
        🌐 Global University Rankings Dashboard | Last updated: {datetime.date.today().strftime('%B %d, %Y')}<br>
        Data Source: <a href="https://cwur.org/2025.php" target="_blank">CWUR 2025</a> |
        Developed by Felix Asibor | Academic & Portfolio Use
    </p>
    """,
    unsafe_allow_html=True
)

