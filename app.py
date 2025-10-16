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

# --- 1. 页面配置和样式 (只保留一次，并使用更完整的配置) ---
st.set_page_config(
    page_title="CAPS: 大学申请整体评估系统",
    layout="wide", 
    initial_sidebar_state="expanded" # 默认展开侧边栏，方便用户输入 API Key
)

st.markdown("""
<style>
/* 标题美化 */
.stApp header {
    background-color: transparent;
}
.stApp [data-testid="stTitle"] {
    color: #4A0099; /* 主题紫色 */
    font-weight: 800;
}
/* 提升主容器美观度 */
.block-container {
    padding-top: 2rem;
    padding-bottom: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
/* 自定义Metric卡片 */
div[data-testid="stMetricValue"] {
    font-size: 28px;
    color: #6A0DAD; /* 强调色 */
}
div[data-testid="stMetricLabel"] {
    font-size: 14px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# --- 侧边栏配置 (修复 Key 冲突) ---
with st.sidebar:
    st.title("⚙️ 系统配置与工具")
    
    # 修复点: 确保 key 唯一
    OPENAI_API_KEY = st.text_input("OpenAI API Key (必需)", "sk-...", type="password", key="sidebar_api_key_input")

# --- 2. 初始化和模型加载 (使用侧边栏的输入值，如果可用) ---

# --- OpenAI API Key ---
OPENAI_BASE_URL = "https://api.gptsapi.net/v1" 

# 初始化客户端和日志
try:
    client = OpenAI(api_key=st.session_state.get("sidebar_api_key_input"), base_url=OPENAI_BASE_URL)
except Exception as e:
    logging.warning(f"OpenAI客户端初始化失败，功能受限: {e}") 

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

@st.cache_resource
def load_models():
    """加载所有模型和数据文件"""
    try:
        scaler = joblib.load("sas_scaler.pkl")
        with open("sas_fused_weights.json", "r") as f:
            fused_weights_dict = json.load(f)
        
        eqi_model = xgb.XGBRegressor()
        eqi_model.load_model("xgb_eqi_regressor_tuned.json")
        
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        return scaler, fused_weights_dict, eqi_model, embed_model
    except FileNotFoundError as e:
        st.error(f"错误：缺少必要的模型文件 -> {e.filename}。请确保所有 .pkl 和 .json 文件与 app.py 在同一个文件夹中。")
        return None, None, None, None

models = load_models()
if any(model is None for model in models):
    st.stop()
scaler, fused_weights_dict, eqi_model, embed_model = models

# --- ⭐️ AI解析EC文本 (函数部分) ---
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


# --- 2. SAS 模块函数 (保持不变) ---
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

# --- 3. EQI 模块函数 (保持不变) ---
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
        st.warning(f"EQI GPT评分失败: {e}")
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
        explanation = explanation_match.group(1).strip() if explanation_match else "未能提取解释。"
        return alignment_score, explanation
    except Exception as e:
        st.warning(f"EQI Prompt对齐度评估失败: {e}")
        return 0.75, "GPT评估出现问题，使用默认值。"

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
        st.warning(f"EQI反馈生成失败: {e}")
        return "未能生成反馈。"
        
def evaluate_essay_full(essay_text: str, essay_prompt: str):
    gpt_scores = get_gpt_scores(essay_text)
    embedding = embed_model.encode([essay_text])
    df = pd.DataFrame(embedding, columns=[f"EssayEmbedding_{i}" for i in range(embedding.shape[1])])
    for col, score in gpt_scores.items():
        df[col] = score
    
    # 确保列顺序与训练时一致
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

# --- 4. EIS 模块函数 (保持不变) ---
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
        st.warning(f"EIS单项活动评分失败: {e}")
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
        st.warning(f"EIS连贯性评估失败: {e}")
        return 0.7

# 新的、更优的EIS计算函数
# 新的、更优的EIS计算函数
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

    # 1. 按分数从高到低排序
    sorted_activities = sorted(results, key=lambda x: x["EIS_Score"], reverse=True)
    
    # 2. 生成衰减权重 (1, 1/2, 1/3, ...)
    decay_weights = np.array([1 / (i + 1) for i in range(len(sorted_activities))])
    
    # 3. 归一化权重，使总和为1
    normalized_weights = decay_weights / np.sum(decay_weights)
    
    # 4. 计算加权平均分
    weighted_avg_eis = 0
    display_results = []
    for i, activity in enumerate(sorted_activities):
        weight = normalized_weights[i]
        weighted_avg_eis += activity["EIS_Score"] * weight
        display_results.append({
            "Activity": activity["Name"],
            "Tier": activity["Tier"],
            "EIS_Score": activity["EIS_Score"],
            "Weight": f"{weight:.2%}" # 以百分比形式显示权重
        })
    
    df_display = pd.DataFrame(display_results)

    # 5. 应用连贯性曲线
    all_descriptions = [act["Description"] for act in sorted_activities]
    coherence = evaluate_coherence_gpt(all_descriptions)
    final_eis_weighted = round(weighted_avg_eis * (0.85 + 0.15 * coherence), 4)

    return df_display, final_eis_weighted

# --- 4. Streamlit UI ---
st.title("🎓 CAPS: 大学申请整体评估系统")
st.markdown("---") # 分割线

if 'activities' not in st.session_state:
    st.session_state.activities = [
        {"name": "Sample Activity 1", "desc": "Description for sample activity 1.", "tier": "T2"},
        {"name": "Sample Activity 2", "desc": "Description for sample activity 2.", "tier": "T3"},
    ]

# --- 主输入表单 ---
with st.form("caps_form"):
    
    # === 1. 学术背景 (SAS) ===
    st.header("1. 👤 学术背景 (SAS)")
    st.markdown("请量化您的**标准化**学术表现：")
    with st.expander("💡 什么是SAS分数？（点击查看详情）"):
        st.markdown("""
                    **SAS (Standardized Academic Score)** 是对您学术竞争力的量化评估。根据您的论文，它综合了以下几个方面：
                    * **GPA**: 在校平均成绩
                    * **标化成绩**: SAT 或 ACT 分数
                    * **语言成绩**: 托福 (TOEFL) 或 雅思 (IELTS) 分数
                    * **AP/高阶课程数量**: 获得5分的AP考试数量
                    * **课程难度**: 您高中课程的整体挑战性
                    """)
    # 使用列来优化布局
    col_gpa, col_ap, col_diff = st.columns(3)
    gpa = col_gpa.number_input("**GPA (0.0-4.0)**", 0.0, 4.0, 3.9, 0.01, key="input_gpa")
    ap_count = col_ap.number_input("**AP AP/高阶课程数量**", 0, 20, 5, 1, key="input_ap_count")
    course_diff = col_diff.slider("**高中课程难度**", 1, 5, 4, help="5为最高难度，代表您选修了大量AP/IB/A-Level等高阶课程(10+)。", key="input_course_diff")
    st.markdown("---")
    
    col_sat_act, col_sat, col_lang_test, col_lang = st.columns(4)
    test_type = col_sat_act.radio("**标化考试类型**", ("SAT", "ACT"), horizontal=True, key="test_type")
    
    if test_type == "SAT":
        sat_score = col_sat.number_input("SAT 总分", 1000, 1600, 1520, 10, key="input_sat")
        act_score = None
    else:
        act_score = col_sat.number_input("ACT 总分", 1, 36, 34, 1, key="input_act")
        sat_score = None
        
    lang_test_type = col_lang_test.radio("**语言考试类型**", ("TOEFL", "IELTS"), horizontal=True, key="lang_test_type")
    
    if lang_test_type == "TOEFL":
        toefl_score = col_lang.number_input("TOEFL 分数", 0.0, 120.0, 110.0, 1.0, key="input_toefl")
        ielts_score = None
    else:
        ielts_score = col_lang.number_input("IELTS 分数", 0.0, 9.0, 8.0, 0.5, key="input_ielts")
        toefl_score = None
    
    st.markdown("---")

    # === 2. 申请文书 (EQI) ===
    st.header("2. 📝 申请文书 (EQI)")
    st.markdown("文书是展示**个人特质**的关键环节。")
    with st.expander("💡 什么是EQI分数？（点击查看详情）"):
        st.markdown("""**EQI (Essay Quality Index)** 是对文书质量的综合评估。它利用自然语言处理（NLP）和大型语言模型（LLM）来分析您文书的**主题内容、语言结构、以及与题目（Prompt）的契合度**，最终生成一个量化分数。""")

    essay_prompt = st.text_input("文书题目 (Common App 或其他)",
                                 value="Discuss an accomplishment, event, or realization that sparked a period of personal growth and a new understanding of yourself or others.", 
                                 key="essay_prompt")
    essay_text = st.text_area("文书内容（建议粘贴主文书全文）", 
                                  height=300, 
                                  value="\"Start your essay here", 
                                  key="essay_text")
    st.markdown("---")
    
    # === 3. 课外活动 (EIS) ===
    st.header("3. 🌍 课外活动 (EIS)")
    st.markdown("请在下方管理和输入您的课外活动列表：")
    
    with st.expander("💡 活动等级 (Tier) 定义详解（点击展开）", expanded=True):
        st.markdown("""
        Tier（等级）是衡量课外活动影响力和成就水平的一种方式。请根据以下定义为你的每项活动选择最合适的等级：
        | 等级 | 成就水平 | 示例 |
        | :--- | :--- | :--- |
        | **T1** | **国家/国际级顶尖** | 国际奥赛奖牌、拥有一定影响力的创业公司创始人、在知名期刊发表研究  |
        | **T2** | **州/区域级重要** | 州级竞赛冠军、大型会议组织者、非盈利组织主管  |
        | **T3** | **校级/持续领导力** | 学生会主席、校队队长、校级重要奖项  |
        | **T4** | **持续参与/无领导** | 俱乐部活跃成员、持续社区志愿者  |
        | **T5** | **参与度有限或短期** | 单次活动参与者、普通兴趣爱好  |
        """)

    # 表格头部
    cols_header = st.columns([3, 5, 2])
    cols_header[0].markdown("**活动名称/角色**")
    cols_header[1].markdown("**活动描述/成就**")
    cols_header[2].markdown("**等级 (Tier)**")
    
    # 循环输入
    for i, activity in enumerate(st.session_state.activities):
        cols = st.columns([3, 5, 2])
        st.session_state.activities[i]["name"] = cols[0].text_input(f"Activity {i+1} Name", value=activity.get("name", ""), key=f"name_{i}", label_visibility="collapsed")
        st.session_state.activities[i]["desc"] = cols[1].text_area(f"Activity {i+1} Desc", value=activity.get("desc", ""), key=f"desc_{i}", height=100, label_visibility="collapsed")
        st.session_state.activities[i]["tier"] = cols[2].selectbox(f"Activity {i+1} Tier", options=["T1", "T2", "T3", "T4", "T5"], index=["T1", "T2", "T3", "T4", "T5"].index(activity.get("tier", "T3")), key=f"tier_{i}", label_visibility="collapsed")

    submitted = st.form_submit_button("🚀 计算 CAPS 评估分数", use_container_width=True, type="primary")

# --- 活动列表管理工具 (必须放在表单外部) ---
with st.expander("🛠️ 活动列表工具 (AI填充, 添加/删除)"):
    st.subheader("🤖 AI 快速导入")
    raw_ec_text = st.text_area("将所有活动描述一次性粘贴到这里...", height=150)
    if st.button("AI 自动填充列表"):
        if raw_ec_text and OPENAI_API_KEY and "sk-..." not in OPENAI_API_KEY:
            with st.spinner("🤖 AI 正在解析..."):
                parsed_activities = parse_ec_text_with_ai(raw_ec_text)
                if parsed_activities:
                    st.session_state.activities = parsed_activities
                    st.success("AI 填充成功！列表已更新。")
                    st.rerun()
        else:
            st.warning("请在侧边栏输入API Key并在此处粘贴活动文本。")
    
    st.markdown("---")
    st.subheader("手动管理")
    col1, col2, _ = st.columns([1, 1, 4])
    if col1.button("➕ 添加新活动"):
        st.session_state.activities.append({"name": "", "desc": "", "tier": "T3"})
        st.rerun()
    if col2.button("➖ 删除最后活动"):
        if st.session_state.activities:
            st.session_state.activities.pop()
            st.rerun()


# --- 5. 提交后的计算和结果展示 ---
if submitted:
    # 检查实际使用的 API Key（侧边栏输入的值）
    current_api_key = st.session_state.get("sidebar_api_key_input")
    if not current_api_key or "sk-..." in current_api_key:
        st.error("请在左侧**侧边栏**输入您的 OpenAI API Key。")
        st.stop()
    
    try:
        # --- 计算部分 (调用您的真实函数) ---
        with st.spinner('⏳ 正在分析您的学术背景 (SAS)...'):
            sas_input = {"GPA": gpa, "AP_5_Count": ap_count, "Course_Difficulty": course_diff}
            sas_input["SAT"] = sat_score if sat_score is not None else convert_act_to_sat(act_score)
            sas_input["TOEFL"] = toefl_score if toefl_score is not None else convert_ielts_to_toefl(ielts_score)
            sas_final_score = compute_sas_score(sas_input)

        # 修正后的代码 ✅
        with st.spinner('正在评估您的课外活动 (EIS)...'):
        # 将 "Name", "Description", "Tier" 改为 "name", "desc", "tier"
            eis_activities = [{"name": a.get("name", ""), "desc": a.get("desc", ""), "tier": a.get("tier", "T3")} for a in st.session_state.activities if a.get("name")]
            eis_df, eis_final_score = evaluate_activities_weighted(eis_activities)

        with st.spinner('⏳ 正在深度解析您的申请文书 (EQI)...'):
            eqi_results = evaluate_essay_full(essay_text, essay_prompt)
            eqi_final_score = eqi_results['eqi_final']
            
        st.success("🎉 评估完成！")

        # --- 结果展示部分 (保持不变) ---
        weights = {'SAS': 0.40, 'EQI': 0.31, 'EIS': 0.29}
        caps_score = (sas_final_score * weights['SAS'] + eqi_final_score * weights['EQI'] + eis_final_score * weights['EIS'])
        
        # 1. 总分展示
        st.header("📈 综合评估结果 (CAPS)")
        st.markdown(f"""
        <div style="text-align: center; padding: 25px; border-radius: 12px; background-color: #f7f0ff; border: 2px solid #6A0DAD;">
            <p style="font-size: 26px; font-weight: bold; color: #4B0082; margin: 0;">您的综合申请人档案分数 (CAPS)</p>
            <p style="font-size: 72px; font-weight: extra-bold; color: #6A0DAD; margin: 0; line-height: 1.2;">{caps_score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(caps_score)
        st.markdown("---")
        
        # 2. 各模块分数详情
        st.subheader("📊 各模块分数详情")
        col1, col2, col3 = st.columns(3)
        col1.metric("学术分数 (SAS)", f"{sas_final_score:.2%}")
        col2.metric("文书分数 (EQI)", f"{eqi_final_score:.2%}")
        col3.metric("活动分数 (EIS)", f"{eis_final_score:.2%}")

        # 3. 雷达图
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[sas_final_score*100, eqi_final_score*100, eis_final_score*100],
            theta=['学术 (SAS)', '文书 (EQI)', '活动 (EIS)'], fill='toself', name='您的分数',
            marker=dict(color="#6A0DAD"), line=dict(color="#6A0DAD")
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")), 
            showlegend=False, 
            title=dict(text="三大核心能力雷达图", font=dict(size=18)),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. 详细反馈 (精简与分组)
        st.subheader("📝 详细反馈与建议")
        
        col_eqi, col_eis = st.columns(2)
        
        with col_eqi.expander("文书 (EQI) 深度解析"):
            st.markdown(f"**Prompt 对齐度**: **{eqi_results['alignment_score']:.2f}**")
            st.markdown(eqi_results['alignment_feedback'])
            st.markdown("---")
            st.markdown("**AI 综合建议:**"); st.markdown(eqi_results['suggestions'])
            
        with col_eis.expander("活动 (EIS) 详细列表"):
            st.dataframe(eis_df, hide_index=True, use_container_width=True)
            st.markdown(f"**活动总分 (EIS):** `{eis_final_score:.2%}`")


        # 5. CAPS 解读 (保持不变)
        st.markdown("---")
        st.header("💡 如何解读你的CAPS分数？")
        st.info("⚠️ **重要声明**: 以下解读仅为宏观层面的大致参考，旨在帮助你了解个人档案的相对竞争力，绝不构成任何形式的录取保证。")

        score_percent = caps_score * 100
        # 此处使用上次优化后的 if/elif 结构进行分数解读
        if score_percent >= 96:
            st.subheader(f"✅ 你的分数段: {score_percent:.1f}% (A+)")
            st.markdown("""
            ### **🥇 竞争力分析**
            **极具竞争力。** 你的档案在学术、文书和活动三个维度上均表现出**顶尖水平**，属于全球最优秀申请人之列。

            ### **🌍 参考学校层级**
            你是全球**最顶尖大学**（Ivy, MIT、Stanford、Caltech，录取率低于10%）的**有力竞争者**。

            ### **🛠️ 下一步建议**
            保持优势。重点是**精心打磨申请材料的每一个细节**，确保文书能呈现一个**独特、深刻且有深度**的个人故事。避免任何低级错误。
            """)

        elif score_percent >= 90:
            st.subheader(f"⭐ 你的分数段: {score_percent:.1f}% (A)")
            st.markdown("""
            ### **🌟 竞争力分析**
            **非常强劲。** 你的档案在各方面都表现出色，具备**冲击最顶尖名校**的实力，是 Top 20 院校的有力候选人。

            ### **🇺🇸 参考学校层级**
            你有很大机会被**顶尖大学**（如 Top 20，例如UCLA, UCB, Rice, Vandy, Cornell, JHU, UChi）录取，同时也是更顶尖学校的**合格候选人**。

            ### **🛠️ 下一步建议**
            **找出短板并弥补。** 检查是否存在某个维度的突出不足。思考如何通过**文书**将你的多方面亮点串联成一个**有说服力的、一致的故事**。
            """)

        elif score_percent >= 85:
            st.subheader(f"⚡ 你的分数段: {score_percent:.1f}% (A / A-)")
            st.markdown("""
            ### **🚀 竞争力分析**
            **高度竞争力。** 档案基础扎实且优秀，在竞争激烈的 Top 25 申请池中具有明显优势。

            ### **🏫 参考学校层级**
            你有很大机会被**高排位大学**（如 Top 25，例如 CMU(非CS相关), UMich, WashU, UVA, Emory）录取。关键在于**选校策略**和**文书的区分度**。

            ### **🛠️ 下一步建议**
            **强调差异化。** 你的分数很接近 Top 20 门槛，需重点在**文书和活动细节**上突出你的独特价值和兴趣深度，避免被“分数优秀但无特色”的申请者淹没。
            """)

        elif score_percent >= 80:
            st.subheader(f"📈 你的分数段: {score_percent:.1f}% (A- / B+)")
            st.markdown("""
            ### **💪 竞争力分析**
            **较强竞争力。** 档案整体优秀，在 Top 30 甚至更高排位学校的申请池中有一席之地。

            ### **🎯 参考学校层级**
            你有很大机会被**优秀大学**（如 Top 30，例如 NYU, USC, UCSD）录取。通过合理的选校可以确保录取结果。

            ### **🛠️ 下一步建议**
            **聚焦转化。** 分数已达标，重点在于**高效利用文书和推荐信**，将量化的优秀（分数）转化为招生官眼中有血有肉、未来可期的潜力（故事）。
            """)

        elif score_percent >= 70:
            st.subheader(f"👍 你的分数段: {score_percent:.1f}% (B+ / B)")
            st.markdown("""
            ### **📊 竞争力分析**
            **良好竞争力。** 你的档案整体稳健，是大多数优秀大学申请池中的**可靠申请者**。

            ### **✅ 参考学校层级**
            你是**优秀大学**（如 Top 50，例如 UIUC(非工院), Wisconsin Madison, UCD, UCI, UCSB, BU）的可靠申请者，并有机会通过突出的文书和活动**冲刺更高排位**。

            ### **🛠️ 下一步建议**
            **突出核心优势，形成差异化。** 重点分析你的分数构成，如果某一维度（如课外活动、特定竞赛）特别突出，应在文书中**重点强调**，以弥补分数上的不足。
            """)

        else:
            st.subheader(f"⚠️ 你的分数段: {score_percent:.1f}% (C+ 及以下)")
            st.markdown("""
            ### **🚨 竞争力分析**
            **基础良好，但需强化。** 你的档案在冲击顶尖名校时会面临较大挑战，但仍有机会被众多优秀大学录取。

            ### **📝 参考学校层级**
            对于 Top 50 院校的申请可能需要更加审慎，但在众多**优秀大学**（例如 OSU, Penn State, Rutgers 等）中，你仍有机会获得录取。

            ### **🛠️ 下一步建议**
            **深入分析，立即提升。** 深入分析你的分数构成，**找到最薄弱的环节**。思考是否有机会在截止日期前通过参加竞赛、重考标化、或深化活动等方式进行**实质性提升**。
            """)

    except Exception as e:
        st.error(f"在计算过程中发生了一个意料之外的错误: {e}")
st.set_page_config(layout="wide")

st.title("我的访客地图")


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

# Adjust the height to better fit the new width
components.html(map_html, height=500)

# Your markdown for spacing is perfectly fine
st.markdown("<br>" * 5, unsafe_allow_html=True)