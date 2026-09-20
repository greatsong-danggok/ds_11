import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 1. 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 데이터 탐색")
st.write("뇌졸중 데이터를 여러 각도에서 살펴보는 페이지입니다.")

st.divider()

# -----------------------------
# 2. 데이터 불러오기 (첫 화면과 동일)
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="UTF-8")
    return df

df = load_data()

# -----------------------------
# 3. 나이와 평균 혈당 분포 - 히스토그램 두 개 나란히
# -----------------------------
st.subheader("1️⃣ 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age_hist = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_age_hist, use_container_width=True)

with col2:
    fig_glucose_hist = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.divider()

# -----------------------------
# 4. 뇌졸중 여부에 따른 나이 / 평균 혈당 상자그림 비교
# -----------------------------
st.subheader("2️⃣ 뇌졸중 여부에 따른 나이·평균 혈당 비교")

# stroke 열은 0과 1이라서, 보기 편하도록 문자열 이름을 새로 만들어줌
df["stroke_label"] = df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

col3, col4 = st.columns(2)

with col3:
    fig_age_box = px.box(
        df,
        x="stroke_label",
        y="age",
        title="뇌졸중 여부별 나이 상자그림",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col4:
    fig_glucose_box = px.box(
        df,
        x="stroke_label",
        y="avg_glucose_level",
        title="뇌졸중 여부별 평균 혈당 상자그림",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 두 그룹의 평균값을 표로 정리
mean_table = df.groupby("stroke_label")[["age", "avg_glucose_level"]].mean().reset_index()
mean_table.columns = ["뇌졸중 여부", "평균 나이", "평균 혈당"]
mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.write("**그룹별 평균값**")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# 5. 고혈압/심장병 유무에 따른 뇌졸중 비율 - 막대그래프
# -----------------------------
st.subheader("3️⃣ 고혈압·심장병 유무에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

with col5:
    # 고혈압 유무(0/1)별로 뇌졸중 비율(평균값 = 1의 비율)을 계산
    hypertension_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hypertension_ratio["hypertension"] = hypertension_ratio["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})
    hypertension_ratio["stroke"] = hypertension_ratio["stroke"] * 100  # 백분율로 변환

    fig_hyper = px.bar(
        hypertension_ratio,
        x="hypertension",
        y="stroke",
        title="고혈압 유무별 뇌졸중 비율(%)",
        labels={"hypertension": "고혈압 유무", "stroke": "뇌졸중 비율(%)"},
        text_auto=".2f"
    )
    st.plotly_chart(fig_hyper, use_container_width=True)

with col6:
    # 심장병 유무(0/1)별로 뇌졸중 비율 계산
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["heart_disease"] = heart_ratio["heart_disease"].map({0: "심장병 없음", 1: "심장병 있음"})
    heart_ratio["stroke"] = heart_ratio["stroke"] * 100

    fig_heart = px.bar(
        heart_ratio,
        x="heart_disease",
        y="stroke",
        title="심장병 유무별 뇌졸중 비율(%)",
        labels={"heart_disease": "심장병 유무", "stroke": "뇌졸중 비율(%)"},
        text_auto=".2f"
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# -----------------------------
# 6. bmi 결측치가 있는 사람들의 뇌졸중 비율 vs 전체 뇌졸중 비율
# -----------------------------
st.subheader("4️⃣ bmi 결측치와 뇌졸중 비율")

# bmi가 비어있는(결측인) 행만 골라냄
bmi_missing_df = df[df["bmi"].isnull()]

missing_count = len(bmi_missing_df)
missing_stroke_ratio = bmi_missing_df["stroke"].mean() * 100 if missing_count > 0 else 0
overall_stroke_ratio = df["stroke"].mean() * 100

bmi_compare_table = pd.DataFrame({
    "구분": ["bmi 결측인 사람", "전체 사람"],
    "인원 수": [missing_count, len(df)],
    "뇌졸중 비율(%)": [round(missing_stroke_ratio, 2), round(overall_stroke_ratio, 2)]
})

st.dataframe(bmi_compare_table, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# 7. 흡연 상태별 인원 수 표
# -----------------------------
st.subheader("5️⃣ 흡연 상태별 인원 수")

smoking_count_table = df["smoking_status"].value_counts().reset_index()
smoking_count_table.columns = ["흡연 상태", "인원 수"]

st.dataframe(smoking_count_table, use_container_width=True, hide_index=True)
