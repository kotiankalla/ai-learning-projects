from dotenv import load_dotenv

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
st.set_page_config(page_title="Blood Work Analyzer", layout="wide")

llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it")
st.markdown(""" 

<style>

.scroll-box {

    height: 230px;
    overflow-y: auto;
    padding: 12px 16px;
    border: 1px solid #333;
    border-radius: 8px;
    background-color: #lelele;
    font-size: 0.9rem;
    line-height: 1.6;
}

.scroll-box p, .scroll-box li {
    color: #e0e@eO;
}

.section-label{
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 6px;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.title("Blood Work Analyzer")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Blood Work Report")

    blood_report = st.text_area(
        label="Paste your report below",
        height=500,
        placeholder="Paste your blood work report here...",
        label_visibility="collapsed"
    )
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

with right_col:
    st.subheader("Health Summary")
    health_box = st.empty()
    health_box.markdown( '<div class="scroll-box"></div>', unsafe_allow_html=True)

    st.subheader("Suggested Diet Plan")
    diet_box = st.empty ()
    diet_box.markdown('<div class="scroll-box"></div>' , unsafe_allow_html=True)

if analyze_clicked:
    if not blood_report.strip():
        with left_col:
            st.warning("Please paste a blood work report before analyzing.")
    else:
        with st.spinner("Analyzing your blood work..."):
            # Stage1: extract and flag  abnormal vaues.
            extraction_prompt = f"""
            You are a medical data extaction assistant.
            from the blood report below, extract all test values and classify each one as HIGH, LOW, NORMAL based on the reference range provided in the report.

            Format your response as:
            - Test Name : value | status: HIGH/LOW/NORMAL | Reference: range

            Blood Report:
            {blood_report}
            """
            extraction_response = llm.invoke(extraction_prompt)
            extracted_values = extraction_response.text


            # Stage 2: Health summary and Indian diet plan
            diet_prompt = f"""
            You are a clinical nutritionist specializing in Indian dietery habits.

            Based on the bood work analysis below, write:
            1. A short health summary in 3 lines explaining the patient's condition in simple language.
            2. A short, practical Indian dient plan having only two sections (1) food to avoid (2) food to eat more of
                Do not include any other sections in diet plan.

            Blood Work Anaysis: {extracted_values}
            """

            diet_response = llm.invoke(diet_prompt)
            full_response = diet_response.text

        # split response into two sections
        if"SECTION 2" in full_response:
            parts = full_response.split("SECTION 2")
            health_summary = parts[0].replace("SECTION 1 - HEALTH SUMMARY:","").replace("SECTION 1","").strip()
            diet_plan = ("SECTION 2" + parts[1]).replace("SECTION 2 - INDIAN DIET PLAN:", "").replace("SECTION 2","")

        else:
            health_summary = full_response
            diet_plan = ""

        #Render into fixed-height scrollable box
        health_box.markdown(
            f'<div class="scroll-box"> {health_summary} </div>',
            unsafe_allow_html=True
        )
        diet_box.markdown(
            f'<div class="scroll-box">{diet_plan if diet_plan else full_response} </div>',
            unsafe_allow_html=True
        )


