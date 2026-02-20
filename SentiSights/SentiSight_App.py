import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import re
import time
import os
import base64
import json
import textwrap
import google.generativeai as genai
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="SentiSights | Decision Support System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ** YOUR API KEY **
API_KEY = "Insert Your Gemini API Key Here"
MODEL_ID = 'gemma-3-27b-it'

# --- 2. ASSETS ---
def set_bg_hack(main_bg):
    try:
        with open(main_bg, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{bin_str}");
                background-attachment: fixed;
                background-size: cover;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass 

set_bg_hack('bg1.jpg') 

# --- 3. STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .main { background-color: transparent; }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, span {
        color: #f0f2f6 !important;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5);
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stDataFrame"] { 
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px; 
    }
    [data-testid="stDataFrame"] div, [data-testid="stDataFrame"] span, [data-testid="stDataFrame"] p {
        color: #2c3e50 !important;
        text-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. CORE DATA FUNCTIONS ---
FINAL_COLUMNS_ORDER = [
    'Original_Review_ID', 'Date', 'Branch_Name', 'Reviewer_Name', 'Rating',
    'Full_Review', 'Category', 'Aspect_Term', 'Opinion_Term', 
    'Sentiment_Label', 'Sentiment_Text', 'Recommendation', 
    'Cleaned_Review', 'Model_Input'
]

@st.cache_data
def load_master_database():
    file_path = "Ready_For_Modelling.csv"
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_csv(file_path)
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if 'Sentiment_Text' not in df.columns and 'Sentiment_Label' in df.columns:
            df['Sentiment_Text'] = df['Sentiment_Label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
        
        valid_cols = [c for c in FINAL_COLUMNS_ORDER if c in df.columns]
        return df[valid_cols]
    except: return None

@st.cache_resource
def load_ai_models():
    try:
        if not os.path.exists("svm_model.pkl"): return None, None
        model = joblib.load("svm_model.pkl")
        vectorizer = joblib.load("svm_tfidf_vectorizer.pkl")
        return model, vectorizer
    except: return None, None

def sidebar_logic(df):
    st.sidebar.header("🔍 Filter Controls")
    
    # 1. Branch
    branches = ["All Branches"] + sorted([str(x) for x in df['Branch_Name'].unique()])
    sel_branch = st.sidebar.selectbox("Select Branch", branches)
    
    # 2. Date
    min_d, max_d = df['Date'].min().date(), df['Date'].max().date()
    sel_date = st.sidebar.date_input("Date Range", [min_d, max_d])
    
    # 3. Sentiment
    sel_sent = st.sidebar.multiselect("Sentiment", ['Positive', 'Neutral', 'Negative'], default=['Positive', 'Neutral', 'Negative'])
    
    # 4. Category
    cats = ["All Categories"] + sorted([str(x) for x in df['Category'].unique()])
    sel_cat = st.sidebar.selectbox("Category", cats)
    
    # 5. Aspect Term (Dependent on Category) --- RESTORED ---
    if sel_cat != "All Categories":
        # Only show aspects belonging to selected category
        valid_aspects = sorted(df[df['Category'] == sel_cat]['Aspect_Term'].unique().astype(str))
        aspect_options = ["All Aspects"] + valid_aspects
    else:
        # Show all aspects if no category selected
        valid_aspects = sorted(df['Aspect_Term'].unique().astype(str))
        aspect_options = ["All Aspects"] + valid_aspects
        
    sel_asp = st.sidebar.selectbox("Aspect Term", aspect_options)
    
    # Apply Filters
    filtered = df.copy()
    if sel_branch != "All Branches": filtered = filtered[filtered['Branch_Name'] == sel_branch]
    if len(sel_date) == 2: filtered = filtered[(filtered['Date'].dt.date >= sel_date[0]) & (filtered['Date'].dt.date <= sel_date[1])]
    if sel_sent: filtered = filtered[filtered['Sentiment_Text'].isin(sel_sent)]
    if sel_cat != "All Categories": filtered = filtered[filtered['Category'] == sel_cat]
    if sel_asp != "All Aspects": filtered = filtered[filtered['Aspect_Term'] == sel_asp]
    
    return filtered

# --- 5. AI ENGINE ---

def clean_json_response(text):
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

def extract_aspects_with_gemma(model, review_text):
    prompt = f"""
    Analyze this hotel review. Extract distinct aspects.
    Return ONLY a JSON LIST of objects.
    Format: [{{"Category": "Room", "Aspect_Term": "bed", "Opinion_Term": "comfy"}}]
    Categories: Service, Room, Food, Location, Amenities, Price, Other.
    
    Review: "{review_text}"
    """
    try:
        response = model.generate_content(prompt)
        cleaned = clean_json_response(response.text)
        return json.loads(cleaned)
    except:
        return []

def generate_recommendation(model, category, aspect, opinion):
    prompt = f"Act as Hotel Director. Give a 3-6 word action plan for: {category} - {aspect} was {opinion}."
    try:
        return model.generate_content(prompt).text.strip()
    except:
        return "Investigate."

# --- 6. MODULE: SENTIGEST ---
def render_sentigest(df):
    st.title("📋 SentiGest Dashboard")
    st.markdown("---")
    filtered_df = sidebar_logic(df)
    
    neg_df = filtered_df[filtered_df['Sentiment_Text'] == 'Negative']
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Reviews", f"{len(filtered_df):,}")
    k2.metric("Negative Alerts 🔴", f"{len(neg_df)}", delta="Action Required", delta_color="inverse")
    k3.metric("Top Complaint 🔥", neg_df['Aspect_Term'].value_counts().idxmax() if not neg_df.empty else "N/A")
    k4.metric("Complaint Vol", neg_df['Aspect_Term'].value_counts().max() if not neg_df.empty else 0)
    
    st.divider()
    
    table_df = filtered_df.copy()
    table_df['Date_Str'] = table_df['Date'].dt.strftime('%Y-%m-%d')
    table_df['Sentiment_Display'] = table_df['Sentiment_Text'].map({'Positive':'🟢 Positive','Neutral':'⚪ Neutral','Negative':'🔴 Negative'})
    table_df['Action_Plan'] = table_df.apply(lambda x: f"⚠️ {x['Recommendation']}" if x['Sentiment_Text'] == 'Negative' else "", axis=1)

    selection = st.dataframe(
        table_df[['Date_Str', 'Branch_Name', 'Reviewer_Name', 'Category', 'Aspect_Term', 'Sentiment_Display', 'Action_Plan']],
        use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row",
        column_config={"Date_Str": "Date", "Action_Plan": st.column_config.TextColumn("Recommendation", width="large")}
    )

    if selection.selection["rows"]:
        row = table_df.iloc[selection.selection["rows"][0]]
        glass_bg = "rgba(255, 0, 50, 0.15)" if row['Sentiment_Text'] == 'Negative' else "rgba(255, 255, 255, 0.1)"
        
        st.markdown(f"""
        <div style="margin-top:20px; background:{glass_bg}; backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.2); border-radius:16px; padding:25px; color:white;">
            <h3>👤 {row['Reviewer_Name']}</h3>
            <p>{row['Date_Str']} • {row['Branch_Name']}</p>
            <hr>
            <p><strong>Concern:</strong> {row['Category']} &rarr; {row['Aspect_Term']}</p>
            <p><strong>Opinion:</strong> "{row.get('Opinion_Term', 'N/A')}"</p>
            <div style="background:rgba(0,0,0,0.3); padding:10px; border-radius:8px; margin:10px 0;">
                <em>"{row.get('Full_Review', 'N/A')}"</em>
            </div>
            <p><strong>Recommendation:</strong> {row['Action_Plan']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.download_button("📥 Download CSV", table_df.to_csv().encode('utf-8'), "SentiGest_Report.csv")

# --- 7. MODULE: SENTITREND ---
def render_sentitrend(df):
    st.title("📈 SentiTrend Analytics")
    st.markdown("---")
    filtered_df = sidebar_logic(df)
    
    if filtered_df.empty:
        st.warning("No data found.")
        return

    # 1. Volume Chart
    st.subheader("📊 Aspect Sentiment Volume")
    aspect_grp = filtered_df.groupby(['Aspect_Term', 'Sentiment_Text']).size().reset_index(name='Count')
    top = filtered_df['Aspect_Term'].value_counts().head(20).index
    fig1 = px.bar(aspect_grp[aspect_grp['Aspect_Term'].isin(top)], y='Aspect_Term', x='Count', color='Sentiment_Text', orientation='h',
                  color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}, template="plotly_dark", title="Volume vs Sentiment")
    fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Timeline & Stars (DEDUPLICATED)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 Sentiment Timeline")
        time_grp = filtered_df.groupby([pd.Grouper(key='Date', freq='W'), 'Sentiment_Text']).size().reset_index(name='Count')
        fig2 = px.line(time_grp, x='Date', y='Count', color='Sentiment_Text', color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}, template="plotly_dark")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
        
    with c2:
        st.subheader("⭐ Star Rating (Unique)")
        unique_reviews = filtered_df.drop_duplicates(subset=['Original_Review_ID'])
        if 'Rating' in unique_reviews.columns:
            fig_star = px.histogram(unique_reviews, x='Rating', color='Rating', template="plotly_dark", title="True Star Distribution")
            fig_star.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.2)
            st.plotly_chart(fig_star, use_container_width=True)
        else:
            st.info("Rating data not available.")

    # 3. Branch Benchmarking
    st.subheader("🏨 Branch Benchmarking")
    branch_grp = filtered_df.groupby(['Branch_Name', 'Sentiment_Text']).size().reset_index(name='Reviews')
    fig3 = px.bar(branch_grp, x='Branch_Name', y='Reviews', color='Sentiment_Text', barmode='group', color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}, template="plotly_dark")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Word Cloud
    st.subheader("☁️ Word Clouds")
    c1, c2 = st.columns(2)
    def plot_wc(txt, cmap):
        if txt.empty: return
        wc = WordCloud(width=400, height=200, background_color='rgba(0,0,0,0)', mode="RGBA", colormap=cmap).generate(" ".join(txt.astype(str)))
        fig, ax = plt.subplots(facecolor='none'); ax.imshow(wc); ax.axis('off')
        return fig

    with c1:
        st.markdown("**Positive**")
        st.pyplot(plot_wc(filtered_df[filtered_df['Sentiment_Text']=='Positive']['Opinion_Term'], "Greens"), transparent=True)
    with c2:
        st.markdown("**Negative**")
        st.pyplot(plot_wc(filtered_df[filtered_df['Sentiment_Text']=='Negative']['Opinion_Term'], "Reds"), transparent=True)

# --- 8. MODULE: SENTILEARN ---
def render_sentilearn():
    st.title("🤖 SentiLearn: Hybrid AI Pipeline")
    st.markdown("---")
    
    # REPAIR LOGIC
    master_df = load_master_database()
    failed_rows = master_df[master_df['Aspect_Term'] == 'General'] if master_df is not None else pd.DataFrame()
    
    if not failed_rows.empty:
        st.error(f"⚠️ **Found {len(failed_rows)} rows with 'General' labels.**")
        
        if st.button("🛠️ Repair Failed Rows Now"):
            status = st.status(f"Repairing with {MODEL_ID}...", expanded=True)
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel(MODEL_ID)
            svm_model, vectorizer = load_ai_models()
            
            fixed_rows_list = []
            total = len(failed_rows)
            prog = st.progress(0)
            
            for i, (idx, row) in enumerate(failed_rows.iterrows()):
                aspects = extract_aspects_with_gemma(model, row['Full_Review'])
                
                if aspects:
                    for asp in aspects:
                        new_row = row.copy()
                        new_row['Category'] = asp.get('Category', 'Other')
                        new_row['Aspect_Term'] = asp.get('Aspect_Term', 'General')
                        new_row['Opinion_Term'] = asp.get('Opinion_Term', 'General')
                        
                        # Re-Score
                        inp = f"{new_row['Aspect_Term']} {row.get('Cleaned_Review', '')}"
                        try:
                            pred = svm_model.predict(vectorizer.transform([str(inp)]))[0]
                            new_row['Sentiment_Label'] = pred
                            new_row['Sentiment_Text'] = {0:'Negative',1:'Neutral',2:'Positive'}.get(pred, 'Neutral')
                            new_row['Recommendation'] = generate_recommendation(model, new_row['Category'], new_row['Aspect_Term'], new_row['Opinion_Term']) if pred == 0 else ""
                        except: pass
                        
                        for col in ['state', 'Review_ID']: 
                            if col in new_row: del new_row[col]
                            
                        fixed_rows_list.append(new_row)
                else:
                    fixed_rows_list.append(row)
                
                prog.progress((i+1)/total)
                time.sleep(2.0)
            
            good_rows = master_df[master_df['Aspect_Term'] != 'General']
            final_df = pd.concat([good_rows, pd.DataFrame(fixed_rows_list)], ignore_index=True)
            final_df = final_df[[c for c in FINAL_COLUMNS_ORDER if c in final_df.columns]]
            final_df.to_csv("Ready_For_Modelling.csv", index=False)
            
            status.update(label="✅ Repair Complete! Database Cleaned.", state="complete")
            time.sleep(2)
            st.rerun()
            
    # UPLOAD LOGIC
    uploaded_file = st.file_uploader("Upload New Data", type=['csv'])
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        if st.button("🚀 Process File"):
            status = st.status(f"Processing with {MODEL_ID}...", expanded=True)
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel(MODEL_ID)
            svm, vec = load_ai_models()
            
            df_proc = raw_df.rename(columns={'text': 'Full_Review', 'stars': 'Rating', 'name': 'Reviewer_Name', 'publishedAtDate': 'Date', 'title': 'Branch_Name'})
            df_proc['Date'] = pd.to_datetime(df_proc['Date'], errors='coerce').dt.date
            df_proc['Cleaned_Review'] = df_proc['Full_Review'].astype(str).apply(lambda x: re.sub(r'<.*?>', '', x))
            
            start_id = 10000
            if master_df is not None and 'Original_Review_ID' in master_df.columns:
                 start_id = int(pd.to_numeric(master_df['Original_Review_ID'], errors='coerce').max()) + 1
            df_proc['Review_ID'] = range(start_id, start_id + len(df_proc))

            results = []
            prog = st.progress(0)
            
            for i, row in df_proc.iterrows():
                aspects = extract_aspects_with_gemma(model, row['Cleaned_Review'])
                if not aspects: aspects = [{'Category':'General', 'Aspect_Term':'General', 'Opinion_Term':'General'}]
                
                for asp in aspects:
                    res = {
                        'Original_Review_ID': row['Review_ID'],
                        'Date': row['Date'],
                        'Branch_Name': row['Branch_Name'],
                        'Reviewer_Name': row['Reviewer_Name'],
                        'Rating': row['Rating'],
                        'Full_Review': row['Full_Review'],
                        'Cleaned_Review': row['Cleaned_Review'],
                        'Category': asp.get('Category', 'General'),
                        'Aspect_Term': asp.get('Aspect_Term', 'General'),
                        'Opinion_Term': asp.get('Opinion_Term', 'General')
                    }
                    
                    inp = f"{res['Aspect_Term']} {res['Cleaned_Review']}"
                    res['Model_Input'] = inp
                    try:
                        pred = svm.predict(vec.transform([str(inp)]))[0]
                        res['Sentiment_Label'] = pred
                        res['Sentiment_Text'] = {0:'Negative',1:'Neutral',2:'Positive'}.get(pred)
                        res['Recommendation'] = generate_recommendation(model, res['Category'], res['Aspect_Term'], res['Opinion_Term']) if pred == 0 else ""
                    except: pass
                    
                    results.append(res)
                
                prog.progress((i+1)/len(df_proc))
                time.sleep(2.0)
            
            final_new = pd.DataFrame(results)
            valid_cols = [c for c in FINAL_COLUMNS_ORDER if c in final_new.columns]
            st.session_state.sl_processed_data = final_new[valid_cols]
            status.update(label="Done!", state="complete")

    if st.session_state.get('sl_processed_data') is not None:
        st.divider()
        new_df = st.session_state.sl_processed_data
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### Sentiment Distribution")
            fig_pie = px.pie(new_df, names='Sentiment_Text', 
                             color='Sentiment_Text', 
                             color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'})
            fig_pie.update_traces(textinfo='percent+value')
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.markdown("#### Processed Data Preview")
            st.dataframe(new_df.head(), height=300)
        
        if st.button("💾 Append to Master"):
            current = load_master_database()
            combined = pd.concat([current, st.session_state.sl_processed_data], ignore_index=True)
            combined = combined[[c for c in FINAL_COLUMNS_ORDER if c in combined.columns]]
            combined.to_csv("Ready_For_Modelling.csv", index=False)
            st.success("Saved & Cleaned!")
            st.session_state.sl_processed_data = None
            time.sleep(1)
            st.rerun()

# --- 9. MAIN ---
def main():
    df = load_master_database()
    st.sidebar.title("SentiSights 🏨")
    page = st.sidebar.radio("Navigation", ["SentiGest (Ops)", "SentiTrend (Analytics)", "SentiLearn (AI Pipeline)"])
    if df is not None:
        if "Ops" in page: render_sentigest(df)
        elif "Analytics" in page: render_sentitrend(df)
        elif "Pipeline" in page: render_sentilearn()
    else: st.warning("Please upload 'Ready_For_Modelling.csv'")

if __name__ == "__main__":
    main()