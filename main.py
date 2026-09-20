import streamlit as st
import pandas as pd

# -----------------------------
# 1. 페이지 기본 설정
#    - page_title: 브라우저 탭에 표시되는 제목
#    - page_icon: 브라우저 탭 아이콘 (이모지 사용 가능)
#    - layout: 화면을 넓게 쓰도록 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# 2. 화면 맨 위 제목
# -----------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 데이터를 살펴보고 분석해보는 실습 공간입니다.")

st.divider()

# -----------------------------
# 3. 데이터 불러오기
#    - @st.cache_data: 데이터를 한 번만 불러오고 재사용해서 앱 속도를 빠르게 함
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="UTF-8")
    return df

df = load_data()

# -----------------------------
# 4. 큰 숫자 카드 네 개 (st.metric 사용)
#    - 전체 사람 수
#    - 열 개수
#    - stroke가 1인 사람 수
#    - stroke 비율
# -----------------------------
st.subheader("📊 데이터 한눈에 보기")

total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())  # stroke 열의 값이 0 또는 1이므로 합을 구하면 1인 사람 수가 됨
stroke_ratio = stroke_count / total_people * 100  # 백분율로 계산

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,}명")

with col2:
    st.metric(label="열 개수", value=f"{total_columns}개")

with col3:
    st.metric(label="뇌졸중 발생자 수", value=f"{stroke_count:,}명")

with col4:
    st.metric(label="뇌졸중 발생 비율", value=f"{stroke_ratio:.2f}%")

st.divider()

# -----------------------------
# 5. 열 이름 / 우리말 뜻 / 값의 종류 / 빈 값 개수 표
#    - 우리말 뜻은 학생이 직접 채워 넣을 수 있도록 빈 문자열로 둠
# -----------------------------
st.subheader("📋 열(컬럼) 정보")

# 각 열의 "값의 종류"를 만드는 함수
# - 값의 종류가 너무 많은 숫자형 열(예: age, bmi 등)은 "숫자형"이라고 표시
# - 종류가 적은 범주형 열은 실제 값들을 나열
def get_value_types(column):
    unique_values = df[column].dropna().unique()
    if len(unique_values) > 10:
        return "숫자형 (연속된 값)"
    else:
        # 값들을 문자열로 바꿔서 보기 좋게 나열
        return ", ".join([str(v) for v in sorted(unique_values, key=str)])

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [""] * len(df.columns),  # 학생이 직접 채워 넣을 빈 칸
    "값의 종류": [get_value_types(col) for col in df.columns],
    "빈 값 개수": df.isnull().sum().values
})

# st.data_editor를 사용하면 표를 화면에서 직접 수정할 수 있음
# "우리말 뜻" 칸에 학생이 직접 타이핑해서 채울 수 있도록 함
edited_column_info = st.data_editor(
    column_info,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],  # 이 칸들은 수정 못 하게 막음
    hide_index=True
)

st.divider()

# -----------------------------
# 6. 데이터 처음 다섯 줄 그대로 보여주기
# -----------------------------
st.subheader("🔍 데이터 미리보기 (상위 5개)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# -----------------------------
# 7. 데이터 출처 작성란
#    - 학생이 교재를 보고 직접 입력할 수 있는 텍스트 입력창
# -----------------------------
st.subheader("📚 데이터 출처")
source_text = st.text_area(
    "교재를 참고하여 데이터 출처를 적어보세요.",
    placeholder="여기에 데이터 출처를 입력하세요."
)

if source_text:
    st.info(f"✅ 작성한 출처: {source_text}")
