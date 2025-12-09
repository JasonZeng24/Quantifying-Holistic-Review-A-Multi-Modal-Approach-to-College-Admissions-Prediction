import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import re
from math import exp, isnan
import os
import plotly.graph_objects as go
import logging
import streamlit.components.v1 as components

# --- 1. Page Config & Styling ---
st.set_page_config(
    page_title="CAPS: College Application Prediction System",
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Header Styling */
.stApp header {
    background-color: transparent;
}
.stApp [data-testid="stTitle"] {
    color: #4A0099; /* Theme Purple */
    font-weight: 800;
}
/* Main Container Styling */
.block-container {
    padding-top: 2rem;
    padding-bottom: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
/* Custom Metric Card */
div[data-testid="stMetricValue"] {
    font-size: 28px;
    color: #6A0DAD; /* Accent Color */
}
div[data-testid="stMetricLabel"] {
    font-size: 14px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# --- Sidebar Config ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    OPENAI_API_KEY = st.text_input("OpenAI API Key (Required)", "sk-...", type="password", key="sidebar_api_key_input")

# --- 2. Init & Model Loading ---

OPENAI_BASE_URL = "https://api.gptsapi.net/v1" 

# Init Client
try:
    client = OpenAI(api_key=st.session_state.get("sidebar_api_key_input"), base_url=OPENAI_BASE_URL)
except Exception as e:
    logging.warning(f"OpenAI Client Init Failed: {e}") 

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

@st.cache_resource
def load_models():
    """Load all models and data files"""
    try:
        scaler = joblib.load("sas_scaler.pkl")
        with open("sas_fused_weights.json", "r") as f:
            fused_weights_dict = json.load(f)
        
        eqi_model = xgb.XGBRegressor()
        eqi_model.load_model("xgb_eqi_regressor_tuned.json")
        
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        return scaler, fused_weights_dict, eqi_model, embed_model
    except FileNotFoundError as e:
        st.error(f"Error: Missing model file -> {e.filename}. Please ensure all .pkl and .json files are in the same directory as app.py.")
        return None, None, None, None

models = load_models()
if any(model is None for model in models):
    st.stop()
scaler, fused_weights_dict, eqi_model, embed_model = models

# --- ⭐️ AI Parser for EC Text ---
def parse_ec_text_with_ai(raw_text: str):
    tier_definitions = """
    T1: National or international-level leadership or achievement (e.g., Olympiad medalist, startup founder).
    T2: Major leadership roles or achievements at state or regional level (e.g., state champion, conference organizer).
    T3: Sustained participation with moderate leadership in school-level activities (e.g., club president, team captain).
    T4: General involvement without leadership (e.g., active club member, consistent volunteer).
    T5: Short-term or minimal involvement (e.g., one-time participation, casual hobby).
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "structure_activities",
                "description": "Parses raw text about extracurriculars into a structured list of activities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "activities": {
                            "type": "array",
                            "description": "A list of all the parsed extracurricular activities.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "The concise title of the activity."},
                                    "desc": {"type": "string", "description": "A detailed description of the activity."},
                                    "tier": {"type": "string", "enum": ["T1", "T2", "T3", "T4", "T5"], "description": f"The inferred tier based on these definitions: {tier_definitions}"}
                                },
                                "required": ["name", "desc", "tier"]
                            }
                        }
                    },
                    "required": ["activities"]
                }
            }
        }
    ]

    messages = [{
        "role": "user",
        "content": f"Please parse the following text containing extracurricular activities and structure them using the available tool. Infer the tier for each activity based on the provided tier definitions.\n\nText to parse:\n'''\n{raw_text}\n'''"
    }]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "structure_activities"}},
            temperature=0.1,
        )

        tool_call = response.choices[0].message.tool_calls[0]
        if tool_call.function.name == "structure_activities":
            function_args = json.loads(tool_call.function.arguments)
            return function_args.get("activities", [])

        st.warning("AI did not use the expected tool.")
        return []

    except Exception as e:
        st.error(f"AI parsing with Tools failed: {e}")
        return []


# --- 2. SAS Module ---
feature_order = ["GPA", "SAT", "TOEFL", "AP_5_Count", "Course_Difficulty"]
fused_weights = np.array([fused_weights_dict.get(f, 0) for f in feature_order])

def convert_act_to_sat(act):
    mapping = {36: 1600, 35: 1560, 34: 1500, 33: 1460, 32: 1430, 31: 1400, 30: 1370, 29: 1340, 28: 1310, 27: 1280, 26: 1240, 25: 1210, 24: 1180, 23: 1140, 22: 1110, 21: 1080}
    return mapping.get(act, 1080)

def convert_ielts_to_toefl(ielts):
    mapping = {9.0: 120, 8.50: 115, 8.0: 110, 7.5: 103, 7.0: 98, 6.5: 86, 6.0: 70}
    return mapping.get(ielts, 70)

def compute_sas_score(user_input: dict) -> float:
    user_vec_df = pd.DataFrame([user_input], columns=feature_order)
    user_vec_scaled = scaler.transform(user_vec_df)
    sas_raw = np.dot(user_vec_scaled, fused_weights).item()
    scaled_score = (sas_raw - 0.2) / (1.0 - 0.2)
    return round(min(max(scaled_score, 0.0), 1.0), 4)

# --- 3. EQI Module ---
GPT_MODEL = "gpt-4o"

def get_gpt_scores(essay_text: str) -> dict:
    prompt = f"""
    You are an admissions officer at a top U.S. university. Evaluate the following college essay based on these three criteria:
    1. Content: Is the theme original and does it address the prompt?
    2. Language: Is the word choice precise and natural?
    3. Structure: Does it have a compelling introduction, smooth transitions, and a clear conclusion?
    Give a score from 1 to 5 for each criterion (you may use decimals). Use this exact format, no explanations and no asterisks:
    Content: X  
    Language: X  
    Structure: X
    Essay:
    \"\"\"{essay_text}\"\"\"
    """
    try:
        response = client.chat.completions.create(model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
        reply = response.choices[0].message.content.strip()
        score_dict = {}
        for line in reply.splitlines():
            if ":" in line:
                key, value = line.strip().split(":")
                key = key.strip().lower()
                value = float(value.strip())
                if key == "content": score_dict["EssayContentScore"] = value
                elif key == "language": score_dict["EssayLanguageScore"] = value
                elif key == "structure": score_dict["EssayStructureScore"] = value
        if len(score_dict) != 3: return {"EssayContentScore": 3.0, "EssayLanguageScore": 3.0, "EssayStructureScore": 3.0}
        return score_dict
    except Exception as e:
        st.warning(f"EQI GPT Scoring Failed: {e}")
        return {"EssayContentScore": 3.0, "EssayLanguageScore": 3.0, "EssayStructureScore": 3.0}

def get_prompt_alignment(essay_text: str, prompt_text: str) -> tuple:
    prompt = f"""
    You're a college admissions reviewer. Analyze whether the following college essay answers this Common App prompt:
    \"{prompt_text}\"
    Use this **exact format**, do not change it:
    Alignment Score: [a number between 0 and 1, like 0.83]  
    Explanation: [a short paragraph here]
    Essay:
    \"\"\"{essay_text}\"\"\"
    """
    try:
        response_content = client.chat.completions.create(model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0).choices[0].message.content.strip()
        match = re.search(r"Alignment\s*Score\s*[:：]?\s*\**\s*(0\.\d+|1\.0)", response_content, re.IGNORECASE)
        if not match: raise ValueError("No valid alignment score found in GPT response.")
        alignment_score = float(match.group(1))
        explanation_match = re.search(r"Explanation\s*[:：]\s*(.*)", response_content, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else "Unable to extract explanation."
        return alignment_score, explanation
    except Exception as e:
        st.warning(f"EQI Prompt Alignment Failed: {e}")
        return 0.75, "GPT evaluation issue. Using default value."

def apply_sigmoid_curve(eqi_raw, alignment_score, k=4, x0=0.3, min_val=0.6):
    base_sigmoid = 1 / (1 + exp(-k * (alignment_score - x0)))
    penalty = min_val + (1 - min_val) * base_sigmoid
    return round(eqi_raw * penalty, 4)

def get_eqi_feedback(essay_text, essay_prompt, gpt_scores, alignment_score):
    prompt = f"""
    You are an essay coach.
    ESSAY TEXT: \"\"\"{essay_text}\"\"\"
    PROMPT: {essay_prompt}
    Based on:
    - Content: {gpt_scores['EssayContentScore']}, Language: {gpt_scores['EssayLanguageScore']}, Structure: {gpt_scores['EssayStructureScore']}
    - Prompt alignment score: {alignment_score}
    Give integrated feedback that is direct, helpful, and NOT overly positive. Be honest if the essay is weak.
    IMPORTANT: 
    - Reference specific parts of THIS essay.
    - Give concrete steps to improve.
    - Format your response clearly using Markdown.
    """
    try:
        response = client.chat.completions.create(model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3)
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"EQI Feedback Generation Failed: {e}")
        return "Unable to generate feedback."
        
def evaluate_essay_full(essay_text: str, essay_prompt: str):
    gpt_scores = get_gpt_scores(essay_text)
    embedding = embed_model.encode([essay_text])
    df = pd.DataFrame(embedding, columns=[f"EssayEmbedding_{i}" for i in range(embedding.shape[1])])
    for col, score in gpt_scores.items():
        df[col] = score
    
    # Ensure column order matches training
    feature_names = eqi_model.get_booster().feature_names
    df = df[feature_names]

    eqi_raw = round(eqi_model.predict(df)[0], 4)
    alignment_score, alignment_feedback = get_prompt_alignment(essay_text, essay_prompt)
    eqi_final = apply_sigmoid_curve(eqi_raw, alignment_score)
    suggestions = get_eqi_feedback(essay_text, essay_prompt, gpt_scores, alignment_score)
    
    return {
        "eqi_final": eqi_final,
        "gpt_scores": gpt_scores,
        "alignment_score": alignment_score,
        "alignment_feedback": alignment_feedback,
        "suggestions": suggestions
    }

# --- 4. EIS Module ---
tier_mapping = {"T1": 1.0, "T2": 0.8, "T3": 0.6, "T4": 0.4, "T5": 0.2}

def get_eis_activity_gpt_score(activity_description: str) -> float:
    prompt = f"""
    You are a college admissions officer. Please score the following extracurricular activity on a scale of 0.00 to 1.00 based on its impact, uniqueness, and leadership. Be concise and return only a number.
    Activity: {activity_description}
    Score:"""
    try:
        response = client.chat.completions.create(model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2)
        score_str = response.choices[0].message.content.strip()
        score = float(score_str)
        return round(min(max(score, 0.0), 1.0), 4)
    except Exception as e:
        st.warning(f"EIS Single Activity Scoring Failed: {e}")
        return 0.5

def evaluate_coherence_gpt(descriptions: list) -> float:
    formatted = "\\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])
    prompt = f"""
    You're an admissions reviewer. Evaluate the following set of extracurricular activities and judge how thematically connected they are.
    Rate the coherence of this applicant's activities from 0.0 (completely scattered) to 1.0 (highly focused around a core theme). Return ONLY the number.
    Activities:\n{formatted}
    """
    try:
        response = client.chat.completions.create(model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2)
        score = float(response.choices[0].message.content.strip())
        return round(min(max(score, 0.0), 1.0), 4)
    except Exception as e:
        st.warning(f"EIS Coherence Evaluation Failed: {e}")
        return 0.7

def evaluate_activities_weighted(activity_list: list, alpha: float = 0.5):
    if not activity_list:
        return pd.DataFrame(columns=["Activity", "Tier", "GPT_Score", "EIS_Score", "Weight"]), 0.0

    results = []
    for act in activity_list:
        gpt_score = get_eis_activity_gpt_score(act.get("desc", ""))
        tier_score = tier_mapping.get(act.get("tier", "T3").upper(), 0.0)
        eis_score = round(alpha * gpt_score + (1 - alpha) * tier_score, 4)
        results.append({
            "Name": act.get("name", ""),
            "Tier": act.get("tier", "T3"),
            "Description": act.get("desc", ""),
            "EIS_Score": eis_score
        })

    # 1. Sort by score high to low
    sorted_activities = sorted(results, key=lambda x: x["EIS_Score"], reverse=True)
    
    # 2. Generate decay weights
    decay_weights = np.array([1 / (i + 1) for i in range(len(sorted_activities))])
    
    # 3. Normalize weights
    normalized_weights = decay_weights / np.sum(decay_weights)
    
    # 4. Calculate weighted avg
    weighted_avg_eis = 0
    display_results = []
    for i, activity in enumerate(sorted_activities):
        weight = normalized_weights[i]
        weighted_avg_eis += activity["EIS_Score"] * weight
        display_results.append({
            "Activity": activity["Name"],
            "Tier": activity["Tier"],
            "EIS_Score": activity["EIS_Score"],
            "Weight": f"{weight:.2%}" 
        })
    
    df_display = pd.DataFrame(display_results)

    # 5. Apply coherence curve
    all_descriptions = [act["Description"] for act in sorted_activities]
    coherence = evaluate_coherence_gpt(all_descriptions)
    final_eis_weighted = round(weighted_avg_eis * (0.85 + 0.15 * coherence), 4)

    return df_display, final_eis_weighted

# --- 4. Streamlit UI ---
st.title("🎓 CAPS: College Application Prediction System")
st.markdown("---")

if 'activities' not in st.session_state:
    st.session_state.activities = [
        {"name": "Sample Activity 1", "desc": "Description for sample activity 1.", "tier": "T2"},
        {"name": "Sample Activity 2", "desc": "Description for sample activity 2.", "tier": "T3"},
    ]

# --- Main Input Form ---
with st.form("caps_form"):
    
    # === 1. Academic Background (SAS) ===
    st.header("1. 👤 Academic Background (SAS)")
    st.markdown("Please quantify your **standardized** academic performance:")
    with st.expander("💡 What is SAS? (Click for details)"):
        st.markdown("""
                    **SAS (Standardized Academic Score)** is a quantitative evaluation of your academic competitiveness. Based on our research, it synthesizes:
                    * **GPA**: High school Grade Point Average.
                    * **Standardized Tests**: SAT or ACT scores.
                    * **Language Proficiency**: TOEFL or IELTS scores.
                    * **AP/Advanced Course Count**: Number of AP exams with a score of 5 (or equivalent high achievement).
                    * **Course Difficulty**: The overall rigor of your high school curriculum.
                    """)
    
    col_gpa, col_ap, col_diff = st.columns(3)
    gpa = col_gpa.number_input("**GPA (0.0-4.0)**", 0.0, 4.0, 3.9, 0.01, key="input_gpa")
    ap_count = col_ap.number_input("**Count of 5s in AP/Advanced Courses**", 0, 20, 5, 1, key="input_ap_count")
    course_diff = col_diff.slider("**Course Rigor**", 1, 5, 4, help="5 is maximum rigor, indicating 10+ AP/IB/A-Level courses.", key="input_course_diff")
    st.markdown("---")
    
    col_sat_act, col_sat, col_lang_test, col_lang = st.columns(4)
    test_type = col_sat_act.radio("**Test Type**", ("SAT", "ACT"), horizontal=True, key="test_type")
    
    if test_type == "SAT":
        sat_score = col_sat.number_input("SAT Total", 1000, 1600, 1520, 10, key="input_sat")
        act_score = None
    else:
        act_score = col_sat.number_input("ACT Total", 1, 36, 34, 1, key="input_act")
        sat_score = None
        
    lang_test_type = col_lang_test.radio("**Language Test**", ("TOEFL", "IELTS"), horizontal=True, key="lang_test_type")
    
    if lang_test_type == "TOEFL":
        toefl_score = col_lang.number_input("TOEFL Score", 0.0, 120.0, 110.0, 1.0, key="input_toefl")
        ielts_score = None
    else:
        ielts_score = col_lang.number_input("IELTS Score", 0.0, 9.0, 8.0, 0.5, key="input_ielts")
        toefl_score = None
    
    st.markdown("---")

    # === 2. Essays (EQI) ===
    st.header("2. 📝 Essays (EQI)")
    st.markdown("Essays are critical for demonstrating **personal qualities**.")
    with st.expander("💡 What is EQI? (Click for details)"):
        st.markdown("""**EQI (Essay Quality Index)** evaluates the quality of your writing. It uses NLP and LLMs to analyze your **theme, structure, language, and alignment with the prompt**, generating a quantitative score.""")

    essay_prompt = st.text_input("Essay Prompt (Common App or other)",
                                 value="Discuss an accomplishment, event, or realization that sparked a period of personal growth and a new understanding of yourself or others.", 
                                 key="essay_prompt")
    essay_text = st.text_area("Essay Content (Paste full text here)", 
                              height=300, 
                              value="Start your essay here...", 
                              key="essay_text")
    st.markdown("---")
    
    # === 3. Extracurriculars (EIS) ===
    st.header("3. 🌍 Extracurriculars (EIS)")
    st.markdown("Manage and input your activity list below:")
    
    with st.expander("💡 Tier Definitions (Click to expand)", expanded=True):
        st.markdown("""
        Tiers measure the impact and achievement level of an activity. Select the most appropriate tier based on these definitions:
        | Tier | Achievement Level | Examples |
        | :--- | :--- | :--- |
        | **T1** | **Top National/International** | Olympiad medalist, founder of impactful startup, research published in top journal. |
        | **T2** | **Major State/Regional** | State champion, large conference organizer, head of regional non-profit. |
        | **T3** | **School Level/Sustained Leadership** | Student Body President, Varsity Captain, major school awards. |
        | **T4** | **General Involvement** | Active club member, consistent community volunteer. |
        | **T5** | **Limited/Short-term** | One-time participation, casual hobbies. |
        """)

    # Table Header
    cols_header = st.columns([3, 5, 2])
    cols_header[0].markdown("**Activity Name/Role**")
    cols_header[1].markdown("**Description/Achievements**")
    cols_header[2].markdown("**Tier**")
    
    # Input Loop
    for i, activity in enumerate(st.session_state.activities):
        cols = st.columns([3, 5, 2])
        st.session_state.activities[i]["name"] = cols[0].text_input(f"Activity {i+1} Name", value=activity.get("name", ""), key=f"name_{i}", label_visibility="collapsed")
        st.session_state.activities[i]["desc"] = cols[1].text_area(f"Activity {i+1} Desc", value=activity.get("desc", ""), key=f"desc_{i}", height=100, label_visibility="collapsed")
        st.session_state.activities[i]["tier"] = cols[2].selectbox(f"Activity {i+1} Tier", options=["T1", "T2", "T3", "T4", "T5"], index=["T1", "T2", "T3", "T4", "T5"].index(activity.get("tier", "T3")), key=f"tier_{i}", label_visibility="collapsed")

    submitted = st.form_submit_button("🚀 Calculate CAPS Score", use_container_width=True, type="primary")

# --- Activity List Tools (Must be outside form) ---
with st.expander("🛠️ Activity Tools (AI Import, Add/Delete)"):
    st.subheader("🤖 AI Quick Import")
    raw_ec_text = st.text_area("Paste all activity descriptions here at once...", height=150)
    if st.button("AI Auto-Fill List"):
        if raw_ec_text and OPENAI_API_KEY and "sk-..." not in OPENAI_API_KEY:
            with st.spinner("🤖 AI Parsing..."):
                parsed_activities = parse_ec_text_with_ai(raw_ec_text)
                if parsed_activities:
                    st.session_state.activities = parsed_activities
                    st.success("AI Import Successful! List updated.")
                    st.rerun()
        else:
            st.warning("Please enter your API Key in the sidebar and paste text here.")
    
    st.markdown("---")
    st.subheader("Manual Management")
    col1, col2, _ = st.columns([1, 1, 4])
    if col1.button("➕ Add Activity"):
        st.session_state.activities.append({"name": "", "desc": "", "tier": "T3"})
        st.rerun()
    if col2.button("➖ Remove Last"):
        if st.session_state.activities:
            st.session_state.activities.pop()
            st.rerun()


# --- 5. Calculation & Results ---
if submitted:
    # Check API Key
    current_api_key = st.session_state.get("sidebar_api_key_input")
    if not current_api_key or "sk-..." in current_api_key:
        st.error("Please enter your OpenAI API Key in the **Sidebar**.")
        st.stop()
    
    try:
        # --- Calculations ---
        with st.spinner('⏳ Analyzing Academic Background (SAS)...'):
            sas_input = {"GPA": gpa, "AP_5_Count": ap_count, "Course_Difficulty": course_diff}
            sas_input["SAT"] = sat_score if sat_score is not None else convert_act_to_sat(act_score)
            sas_input["TOEFL"] = toefl_score if toefl_score is not None else convert_ielts_to_toefl(ielts_score)
            sas_final_score = compute_sas_score(sas_input)

        with st.spinner('Evaluating Extracurriculars (EIS)...'):
            eis_activities = [{"name": a.get("name", ""), "desc": a.get("desc", ""), "tier": a.get("tier", "T3")} for a in st.session_state.activities if a.get("name")]
            eis_df, eis_final_score = evaluate_activities_weighted(eis_activities)

        with st.spinner('⏳ Deeply Analyzing Essays (EQI)...'):
            eqi_results = evaluate_essay_full(essay_text, essay_prompt)
            eqi_final_score = eqi_results['eqi_final']
            
        st.success("🎉 Assessment Complete!")

        # --- Results Display ---
        weights = {'SAS': 0.40, 'EQI': 0.31, 'EIS': 0.29}
        caps_score = (sas_final_score * weights['SAS'] + eqi_final_score * weights['EQI'] + eis_final_score * weights['EIS'])
        
        # 1. Total Score
        st.header("📈 Comprehensive Assessment Result (CAPS)")
        st.markdown(f"""
        <div style="text-align: center; padding: 25px; border-radius: 12px; background-color: #f7f0ff; border: 2px solid #6A0DAD;">
            <p style="font-size: 26px; font-weight: bold; color: #4B0082; margin: 0;">Your Comprehensive Applicant Profile Score (CAPS)</p>
            <p style="font-size: 72px; font-weight: extra-bold; color: #6A0DAD; margin: 0; line-height: 1.2;">{caps_score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(caps_score)
        st.markdown("---")
        
        # 2. Module Details
        st.subheader("📊 Module Scores")
        col1, col2, col3 = st.columns(3)
        col1.metric("Academics (SAS)", f"{sas_final_score:.2%}")
        col2.metric("Essays (EQI)", f"{eqi_final_score:.2%}")
        col3.metric("Activities (EIS)", f"{eis_final_score:.2%}")

        # 3. Radar Chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[sas_final_score*100, eqi_final_score*100, eis_final_score*100],
            theta=['Academics (SAS)', 'Essays (EQI)', 'Activities (EIS)'], fill='toself', name='Your Score',
            marker=dict(color="#6A0DAD"), line=dict(color="#6A0DAD")
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")), 
            showlegend=False, 
            title=dict(text="Core Competency Radar", font=dict(size=18)),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Detailed Feedback
        st.subheader("📝 Detailed Feedback & Suggestions")
        
        col_eqi, col_eis = st.columns(2)
        
        with col_eqi.expander("Essay (EQI) Deep Dive"):
            st.markdown(f"**Prompt Alignment**: **{eqi_results['alignment_score']:.2f}**")
            st.markdown(eqi_results['alignment_feedback'])
            st.markdown("---")
            st.markdown("**AI Suggestions:**"); st.markdown(eqi_results['suggestions'])
            
        with col_eis.expander("Activity (EIS) Details"):
            st.dataframe(eis_df, hide_index=True, use_container_width=True)
            st.markdown(f"**Total Activity Score (EIS):** `{eis_final_score:.2%}`")


        # 5. CAPS Interpretation
        st.markdown("---")
        st.header("💡 How to Interpret Your CAPS Score?")
        st.info("⚠️ **Important Disclaimer**: The following interpretation is for high-level reference only and allows you to understand the relative competitiveness of your profile. It does NOT constitute a guarantee of admission.")

        score_percent = caps_score * 100
        
        if score_percent >= 96:
            st.subheader(f"✅ Your Range: {score_percent:.1f}% (A+)")
            st.markdown("""
            ### **🥇 Competitiveness Analysis**
            **Extremely Competitive.** Your profile demonstrates **top-tier** performance across Academics, Essays, and Activities, placing you among the most competitive applicants globally.

            ### **🌍 Reference School Tier**
            You are a **strong contender** for the world's **most elite universities** (Ivy League, MIT, Stanford, Caltech, with acceptance rates <10%).

            ### **🛠️ Next Steps**
            Maintain your edge. Focus on **polishing every detail** of your application. Ensure your essays present a **unique, profound, and deep** personal story. Avoid any unforced errors.
            """)

        elif score_percent >= 90:
            st.subheader(f"⭐ Your Range: {score_percent:.1f}% (A)")
            st.markdown("""
            ### **🌟 Competitiveness Analysis**
            **Very Strong.** Your profile is excellent in all aspects. You have the strength to **challenge top-tier schools** and are a strong candidate for Top 20 institutions.

            ### **🇺🇸 Reference School Tier**
            You have a high chance of admission to **Top Universities** (Top 20, e.g., UCLA, UCB, Rice, Vandy, Cornell, JHU, UChi), and are a qualified candidate for even higher-ranked schools.

            ### **🛠️ Next Steps**
            **Identify and bridge gaps.** Check for any slight weaknesses. Consider how to use your **essays** to connect your various highlights into a **persuasive and consistent narrative**.
            """)

        elif score_percent >= 85:
            st.subheader(f"⚡ Your Range: {score_percent:.1f}% (A / A-)")
            st.markdown("""
            ### **🚀 Competitiveness Analysis**
            **Highly Competitive.** Your profile is solid and excellent, giving you a distinct advantage in the competitive Top 25 applicant pool.

            ### **🏫 Reference School Tier**
            You have a high chance of admission to **High-Ranking Universities** (Top 25, e.g., CMU (non-CS), UMich, WashU, UVA, Emory). Key factors will be **school selection strategy** and **essay differentiation**.

            ### **🛠️ Next Steps**
            **Emphasize Differentiation.** Your score is very close to the Top 20 threshold. Focus on highlighting unique values and depth of interest in your **essays and activity descriptions** to avoid being "excellent but generic."
            """)

        elif score_percent >= 80:
            st.subheader(f"📈 Your Range: {score_percent:.1f}% (A- / B+)")
            st.markdown("""
            ### **💪 Competitiveness Analysis**
            **Strong Competitiveness.** Your profile is generally excellent, earning you a place in the pool for Top 30 or higher-ranked schools.

            ### **🎯 Reference School Tier**
            You have a strong chance of admission to **Excellent Universities** (Top 30, e.g., NYU, USC, UCSD). A smart school list can secure good results.

            ### **🛠️ Next Steps**
            **Focus on Conversion.** Your scores are sufficient; the key is **efficiently using essays and recommendation letters** to convert quantitative excellence (scores) into a vivid, high-potential persona (story).
            """)

        elif score_percent >= 70:
            st.subheader(f"👍 Your Range: {score_percent:.1f}% (B+ / B)")
            st.markdown("""
            ### **📊 Competitiveness Analysis**
            **Good Competitiveness.** Your profile is solid and reliable, making you a **dependable applicant** for many excellent universities.

            ### **✅ Reference School Tier**
            You are a reliable candidate for **Excellent Universities** (Top 50, e.g., UIUC (non-Eng), Wisconsin Madison, UCD, UCI, UCSB, BU), with chances to **reach higher** through outstanding essays and activities.

            ### **🛠️ Next Steps**
            **Highlight Core Strengths.** Analyze your score composition. If one dimension (e.g., ECs or a specific contest) is particularly strong, **emphasize it** in your essays to compensate for other areas.
            """)

        else:
            st.subheader(f"⚠️ Your Range: {score_percent:.1f}% (C+ or below)")
            st.markdown("""
            ### **🚨 Competitiveness Analysis**
            **Good Foundation, Needs Strengthening.** You may face significant challenges targeting elite schools, but you still have opportunities at many excellent universities.

            ### **📝 Reference School Tier**
            Be cautious with Top 50 applications. However, you have good chances at many **Strong Universities** (e.g., OSU, Penn State, Rutgers).

            ### **🛠️ Next Steps**
            **Analyze and Improve Immediately.** Identify your **weakest links**. Consider if you can achieve **substantive improvement** before deadlines via competitions, retaking tests, or deepening activities.
            """)

    except Exception as e:
        st.error(f"An unexpected error occurred during calculation: {e}")

# --- Footer Map ---
st.title("Visitor Map")

map_html = """
<!DOCTYPE html>
<html>
<head>
  <title>ClustrMaps</title>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
    }
  </style>
</head>
<body>
  <a href='https://clustrmaps.com/site/1c88k'  title='Visit tracker'><img src='//clustrmaps.com/map_v2.png?cl=ffffff&w=500&t=tt&d=b6TIoDWDQalUCD6D23hrraFkyPQky1HEmQtA-iNIP7w'/></a>
</html>
"""

components.html(map_html, height=500)

st.markdown("<br>" * 5, unsafe_allow_html=True)
