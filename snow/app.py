import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

# --- APP CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="Accelerator Agent",
    page_icon="🤖"
)

# --- CUSTOM CSS ---
def local_css(file_name):
    """ Loads a local CSS file. """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def style_app():
    """ Injects custom CSS for styling. """
    css = """
    /* Main app body */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    [data-testid="stSidebar"] [data-testid="stHeader"] {
        font-size: 24px;
        color: #111;
    }

    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    }
    
    .metric-card h3 {
        font-size: 18px;
        color: #4a4a4a;
        margin-bottom: 5px;
    }
    
    .metric-card p {
        font-size: 24px;
        font-weight: 600;
        color: #0072c6; /* A nice blue */
    }
    
    .metric-card .similarity-low {
        color: #d9534f; /* Red for low match */
    }
    
    .metric-card .similarity-high {
        color: #5cb85c; /* Green for high match */
    }

    /* Buttons */
    [data-testid="stButton"] button {
        background-color: #0072c6;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    [data-testid="stButton"] button:hover {
        background-color: #005a9e;
        color: white;
    }
    
    /* Headers */
    h1 {
        color: #111;
    }
    h2 {
        color: #333;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 5px;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# --- LOAD MODELS & DATA (with caching) ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_artifacts():
    try:
        reducer = joblib.load('umap_reducer_clust.joblib')
        clusterer = joblib.load('hdbscan_clusterer.joblib')
        acc_vectors = np.load('accelerator_vectors.npy')
        df_acc = pd.read_csv('processed_accelerators.csv')
        df_hack = pd.read_csv('processed_requests.csv')
        df_reco = pd.read_csv('recommendations.csv')
        return reducer, clusterer, acc_vectors, df_acc, df_hack, df_reco
    except FileNotFoundError:
        st.error("Error: Could not find model files. Please run `train_and_save_models.py` first.")
        return None, None, None, None, None, None

model = load_embedding_model()
reducer, clusterer, acc_vectors, df_acc, df_hack, df_reco = load_artifacts()

if df_hack is None:
    st.stop()

# Apply the custom styles
style_app()

# --- SIDEBAR: REQUEST ANALYZER AGENT ---
with st.sidebar:
    st.header("🤖 Request Analyzer Agent")
    st.markdown("Paste in a new request to see which emerging need it matches and how well our current catalog covers it.")
    
    user_text = st.text_area("Enter new request text:", height=150, 
                             placeholder="e.g., 'We need help implementing security guardrails and compliance for our new Azure OpenAI service...'")
    
    if st.button("Analyze Request"):
        if not user_text:
            st.warning("Please enter some request text.")
        else:
            with st.spinner("Analyzing..."):
                # 1. Embed
                new_vector = model.encode([user_text])
                
                # 2. Find Closest Accelerator
                sims = cosine_similarity(new_vector, acc_vectors)
                best_match_idx = sims.argmax()
                best_match_score = sims.max()
                best_match_name = df_acc.iloc[best_match_idx]['name']
                
                # 3. Classify Emerging Need
                reduced_vec = reducer.transform(new_vector)
                cluster_label, strength = hdbscan.approximate_predict(clusterer, reduced_vec)
                cluster_id = cluster_label[0]
                
                st.subheader("Analysis Results")
                
                # Display results in styled cards
                similarity_class = "similarity-high" if best_match_score > 0.5 else "similarity-low"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Closest Existing Accelerator</h3>
                    <p>{best_match_name}</p>
                    <h3 style="margin-top: 15px;">Similarity Score</h3>
                    <p class="{similarity_class}">{best_match_score:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Identified Emerging Need</h3>
                </div>
                """, unsafe_allow_html=True)

                if cluster_id == -1:
                    st.info("This request appears to be unique (classified as 'noise') and does not match a known emerging trend.")
                    st.subheader("🤖 Agent Recommendation")
                    st.info("**ACTION: Review Manually.** This is a unique request. It could be an early signal for a *new* trend, but it is not yet part of a larger pattern.")
                
                else:
                    cluster_info = df_reco[df_reco['cluster_id'] == cluster_id].iloc[0]
                    st.metric(label="Matches Cluster", value=f"Cluster {cluster_id}")
                    st.markdown(f"""
                        - **Keywords:** `{cluster_info['keywords']}`
                        - **Demand:** `{cluster_info['demand']} similar requests`
                        - **Gap Score:** `{cluster_info['gap_score']:.2f}`
                    """)
                    
                    st.subheader("🤖 Agent Recommendation")
                    if cluster_info['gap_score'] > 0.4:
                        st.success(f"**ACTION: Prioritize for Development.** This request belongs to a **high-demand, high-gap** area (Cluster {cluster_id}). It is a strong candidate for a new accelerator.")
                    else:
                        st.info(f"**ACTION: Address with Existing.** This request is well-covered by our portfolio (Best Match: '{cluster_info['best_fit_accelerator']}'). No new accelerator is needed.")

# --- MAIN PAGE: DASHBOARD ---
st.title("💡 Technical Accelerator Insights")
st.markdown("This dashboard analyzes **Access to Experts (A2E)** requests to find emerging needs and identify gaps in our accelerator catalog.")

st.header("Top Recommended New Accelerators 🚀")
st.markdown("These are the **largest (High Demand)** and **most un-met (High Gap Score)** opportunities identified from A2E requests. These should be prioritized for development.")

# Style the recommendations DataFrame
st.dataframe(
    df_reco.style
        .format({'gap_score': '{:.2f}', 'demand': '{:,.0f}'})
        .highlight_max(subset=['demand', 'gap_score'], color='#dff0d8')
        .background_gradient(subset=['gap_score'], cmap='Reds')
        .background_gradient(subset=['demand'], cmap='Blues'),
    use_container_width=True
)

# Put the plot in an expander
with st.expander("Explore the 2D Cluster Map 🗺️", expanded=False):
    st.markdown("This plot shows all incoming requests, colored by the emerging need (cluster) they belong to. (Grey dots are 'noise' - unique, one-off requests).")

    plot_df = df_hack[df_hack['cluster'] != -1].copy()
    plot_df['cluster'] = plot_df['cluster'].astype(str)
    
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['number', 'initiative_title', 'primary_category'],
        title="Clusters of A2E Requests",
        height=700
    )
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)