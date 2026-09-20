import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

# -----------------------------
# 1. 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 분류 모델 만들기")
st.write("뇌졸중을 예측하는 두 가지 모델을 만들고 비교해보는 페이지입니다.")

st.divider()

# -----------------------------
# 2. 데이터 불러오기 (앞 페이지와 동일)
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="UTF-8")
    return df

df = load_data()

# 번호(id) 순으로 정렬 (요청사항: "번호 순으로 정렬해 앞 세 명을 테스트용으로 고정")
df_sorted = df.sort_values("id").reset_index(drop=True)

RANDOM_SEED = 42  # 난수 고정용 숫자

# -----------------------------
# 3. 입력 속성 선택
# -----------------------------
st.subheader("1️⃣ 입력으로 사용할 속성 고르기")

all_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
default_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi 제외한 넷

selected_features = st.multiselect(
    "모델의 입력으로 쓸 속성을 골라보세요. (최소 2개 이상)",
    options=all_features,
    default=default_features
)

# 두 개보다 적게 고르면 안내 후 멈춤
if len(selected_features) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해야 모델을 만들 수 있습니다.")
    st.stop()

st.success(f"선택한 속성: {', '.join(selected_features)}")

st.divider()

# -----------------------------
# 4. 테스트용 3명 고정, 나머지로 학습
# -----------------------------
st.subheader("2️⃣ 데이터 준비하기")

# 번호 순 정렬된 데이터에서 앞 3명은 테스트용, 나머지는 학습용
test_df_raw = df_sorted.iloc[:3].copy()
train_df_raw = df_sorted.iloc[3:].copy()

st.write(f"- 테스트용 데이터: {len(test_df_raw)}명 (번호 순 앞 3명 고정)")
st.write(f"- 학습에 사용할 수 있는 원본 데이터: {len(train_df_raw)}명")

# bmi를 선택한 경우에만, 훈련용 데이터의 중앙값으로 결측치를 채움
if "bmi" in selected_features:
    bmi_median = train_df_raw["bmi"].median()
    train_df_raw["bmi"] = train_df_raw["bmi"].fillna(bmi_median)
    test_df_raw["bmi"] = test_df_raw["bmi"].fillna(bmi_median)
    st.info(f"bmi 결측치는 훈련용 데이터의 중앙값인 **{bmi_median:.2f}**로 채웠습니다.")

# -----------------------------
# 5. 학습 데이터의 크기 맞추기 (언더샘플링)
#    - 뇌졸중 있는 사람 수에 맞춰서, 없는 사람 중 무작위로 같은 수만큼 뽑음
# -----------------------------
positive_train = train_df_raw[train_df_raw["stroke"] == 1]
negative_train = train_df_raw[train_df_raw["stroke"] == 0]

n_positive = len(positive_train)

negative_train_sampled = negative_train.sample(n=n_positive, random_state=RANDOM_SEED)

balanced_train_df = pd.concat([positive_train, negative_train_sampled]).reset_index(drop=True)

st.write(f"- 크기 맞추기 후 학습 데이터: 양성 {n_positive}명 + 음성 {n_positive}명 = 총 {len(balanced_train_df)}명")

# 학습용 X, y / 테스트용 X, y 만들기
X_train = balanced_train_df[selected_features]
y_train = balanced_train_df["stroke"]

X_test = test_df_raw[selected_features]
y_test = test_df_raw["stroke"]

st.divider()

# -----------------------------
# 6. 모델 세 가지 만들기
#    - 로지스틱 회귀 (확률로 답하는 모델)
#    - 의사결정트리 (질문으로 답하는 모델, 깊이 3, 마지막 마디 최소 5명)
#    - 가장 많은 쪽으로만 답하는 모델 (베이스라인)
# -----------------------------
st.subheader("3️⃣ 모델 학습하고 정확도 비교하기")

# 로지스틱 회귀
log_model = LogisticRegression(random_state=RANDOM_SEED)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)

# 의사결정트리 (질문 최대 3번 = max_depth=3, 마지막 마디 5명 미만이면 안 나눔 = min_samples_split=5)
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    random_state=RANDOM_SEED
)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_acc = accuracy_score(y_test, tree_pred)

# 아무것도 보지 않고 많은 쪽으로만 답하는 모델
dummy_model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
dummy_model.fit(X_train, y_train)
dummy_pred = dummy_model.predict(X_test)
dummy_acc = accuracy_score(y_test, dummy_pred)

# 큰 숫자 카드 세 개로 정확도 비교
card1, card2, card3 = st.columns(3)

with card1:
    st.metric("로지스틱 회귀 (확률로 답하는 모델)", f"{log_acc*100:.1f}%")

with card2:
    st.metric("의사결정트리 (질문으로 답하는 모델)", f"{tree_acc*100:.1f}%")

with card3:
    st.metric("아무것도 보지 않고 많은 쪽으로만 답하는 모델", f"{dummy_acc*100:.1f}%")

st.divider()

# -----------------------------
# 7. 산점도 + 로지스틱 회귀 경계선 + 트리 배경색
# -----------------------------
st.subheader("4️⃣ 두 속성으로 그려보는 그림")

if len(selected_features) < 2:
    st.stop()

col_x, col_y = st.columns(2)
with col_x:
    x_axis = st.selectbox("가로축으로 쓸 속성", options=selected_features, index=0)
with col_y:
    remaining = [f for f in selected_features if f != x_axis]
    y_axis = st.selectbox("세로축으로 쓸 속성", options=remaining, index=0)

# 두 축이 아닌 나머지 속성은 테스트 데이터의 중앙값으로 고정
other_features = [f for f in selected_features if f not in [x_axis, y_axis]]
fixed_values = {}
for f in other_features:
    fixed_values[f] = X_test[f].median()

if fixed_values:
    fixed_text = ", ".join([f"{f} = {v:.2f}" for f, v in fixed_values.items()])
    st.write(f"📌 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
else:
    st.write("📌 선택한 속성이 두 개뿐이라 고정할 속성이 없습니다.")

# 그림을 그릴 범위 설정 (테스트 데이터 기준으로 여유 있게)
x_min, x_max = X_test[x_axis].min(), X_test[x_axis].max()
y_min, y_max = X_test[y_axis].min(), X_test[y_axis].max()

# 점이 몰려있을 경우를 대비해 약간의 여백을 줌 (범위가 0이면 임의로 여백을 줌)
x_margin = (x_max - x_min) * 0.3 if x_max > x_min else max(abs(x_min) * 0.3, 1)
y_margin = (y_max - y_min) * 0.3 if y_max > y_min else max(abs(y_min) * 0.3, 1)

x_range_min, x_range_max = x_min - x_margin, x_max + x_margin
y_range_min, y_range_max = y_min - y_margin, y_max + y_margin

# 배경(의사결정트리의 칸)을 계산하기 위한 촘촘한 격자 만들기
grid_steps = 150
xx, yy = np.meshgrid(
    np.linspace(x_range_min, x_range_max, grid_steps),
    np.linspace(y_range_min, y_range_max, grid_steps)
)

# 격자 각 점에 대해 전체 속성값을 만들어줌 (축이 아닌 속성은 고정값 사용)
grid_df = pd.DataFrame({x_axis: xx.ravel(), y_axis: yy.ravel()})
for f in other_features:
    grid_df[f] = fixed_values[f]
grid_df = grid_df[selected_features]  # 학습 때와 같은 열 순서로 맞춤

# 의사결정트리로 격자의 각 점을 예측해서 배경색(칸)을 만듦
tree_grid_pred = tree_model.predict(grid_df)
zz_tree = tree_grid_pred.reshape(xx.shape)

# 산점도 + 배경 + 경계선을 그릴 Figure 생성
fig = go.Figure()

# (1) 의사결정트리의 칸을 옅은 색 등고선(contour)로 표시
fig.add_trace(go.Contour(
    x=np.linspace(x_range_min, x_range_max, grid_steps),
    y=np.linspace(y_range_min, y_range_max, grid_steps),
    z=zz_tree,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, "lightblue"], [1, "lightsalmon"]],
    contours=dict(start=0, end=1, size=1),
    name="트리 칸",
    hoverinfo="skip"
))

# (2) 테스트 데이터 점 찍기 (실제 뇌졸중 여부로 색 구분)
test_plot_df = test_df_raw.copy()
test_plot_df["실제 뇌졸중 여부"] = test_plot_df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

for label, color in [("뇌졸중 없음", "blue"), ("뇌졸중 있음", "red")]:
    subset = test_plot_df[test_plot_df["실제 뇌졸중 여부"] == label]
    fig.add_trace(go.Scatter(
        x=subset[x_axis],
        y=subset[y_axis],
        mode="markers",
        marker=dict(size=14, color=color, line=dict(width=1, color="black")),
        name=label
    ))

# (3) 로지스틱 회귀의 0.5 경계선 계산해서 그리기
# 로지스틱 회귀 식: w1*x1 + w2*x2 + ... + b = 0 이 되는 지점이 확률 0.5 지점
coef = log_model.coef_[0]
intercept = log_model.intercept_[0]

feature_index = {f: i for i, f in enumerate(selected_features)}
x_idx = feature_index[x_axis]
y_idx = feature_index[y_axis]

# 고정된 속성들이 만드는 절편 보정값 계산
fixed_sum = intercept
for f in other_features:
    fixed_sum += coef[feature_index[f]] * fixed_values[f]

w_x = coef[x_idx]
w_y = coef[y_idx]

line_drawn = False
line_out_of_range = False

if abs(w_y) > 1e-10:
    # y = -(w_x * x + fixed_sum) / w_y 형태로 계산
    x_line = np.linspace(x_range_min, x_range_max, 100)
    y_line = -(w_x * x_line + fixed_sum) / w_y

    # 경계선이 그림 범위 안에 있는지 확인
    in_range_mask = (y_line >= y_range_min) & (y_line <= y_range_max)

    if in_range_mask.any():
        fig.add_trace(go.Scatter(
            x=x_line[in_range_mask],
            y=y_line[in_range_mask],
            mode="lines",
            line=dict(color="green", width=3, dash="dash"),
            name="로지스틱 회귀 경계선(0.5)"
        ))
        line_drawn = True
        if not in_range_mask.all():
            line_out_of_range = True  # 일부만 범위 안에 있는 경우
    else:
        line_out_of_range = True
else:
    line_out_of_range = True

fig.update_layout(
    title=f"{x_axis} vs {y_axis} 산점도와 경계선",
    xaxis_title=x_axis,
    yaxis_title=y_axis,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

if not line_drawn or line_out_of_range:
    st.warning("⚠️ 로지스틱 회귀의 경계선이 그림 범위를 벗어나거나 그릴 수 없어서, 일부 또는 전체가 보이지 않습니다.")

st.divider()

# -----------------------------
# 8. 의사결정트리 가지 그림 (graphviz)
# -----------------------------
st.subheader("5️⃣ 의사결정트리 가지 그림")

def build_tree_graph(tree_model, feature_names, X_train, y_train):
    """
    graphviz.Digraph를 이용해 의사결정트리를 직접 그려주는 함수.
    각 마디에 인원 수, 뇌졸중인 사람 수, 비율을 함께 표시하고,
    답을 내는 마디는 예측 결과에 따라 색을 다르게 칠한다.
    """
    tree_ = tree_model.tree_
    graph = graphviz.Digraph()
    graph.attr("node", shape="box", fontname="Malgun Gothic")
    graph.attr("edge", fontname="Malgun Gothic")

    # 각 마디(노드)에 도달하는 훈련 데이터의 인덱스를 계산하기 위해 decision_path 사용
    node_indicator = tree_model.decision_path(X_train)
    leaf_id_all = tree_model.apply(X_train)

    def get_node_samples(node_id):
        # 해당 노드를 지나가는 모든 훈련 데이터의 위치(True/False)를 가져옴
        sample_mask = node_indicator[:, node_id].toarray().ravel().astype(bool)
        return y_train[sample_mask]

    def recurse(node_id):
        samples_y = get_node_samples(node_id)
        n_total = len(samples_y)
        n_positive = int(samples_y.sum())
        ratio = n_positive / n_total * 100 if n_total > 0 else 0

        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]

        if is_leaf:
            # 답을 내는 마디: 다수결로 예측값 결정
            predicted_class = 1 if n_positive >= (n_total - n_positive) else 0
            label = f"인원 {n_total}명\n뇌졸중 {n_positive}명 ({ratio:.1f}%)\n예측: {'뇌졸중' if predicted_class==1 else '아님'}"
            fill_color = "lightsalmon" if predicted_class == 1 else "lightblue"
            graph.node(str(node_id), label=label, style="filled", fillcolor=fill_color)
        else:
            feature = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            label = f"{feature} <= {threshold:.2f} ?\n인원 {n_total}명\n뇌졸중 {n_positive}명 ({ratio:.1f}%)"
            graph.node(str(node_id), label=label, style="filled", fillcolor="white")

            left_id = tree_.children_left[node_id]
            right_id = tree_.children_right[node_id]

            recurse(left_id)
            recurse(right_id)

            graph.edge(str(node_id), str(left_id), label="예")
            graph.edge(str(node_id), str(right_id), label="아니요")

    recurse(0)
    return graph

tree_graph = build_tree_graph(tree_model, selected_features, X_train.reset_index(drop=True), y_train.reset_index(drop=True))
st.graphviz_chart(tree_graph)

# -----------------------------
# 9. 트리 요약 정보
# -----------------------------
st.subheader("6️⃣ 트리 요약")

tree_ = tree_model.tree_
n_nodes = tree_.node_count
is_leaf_array = (tree_.children_left == tree_.children_right)
leaf_indices = np.where(is_leaf_array)[0]

# 각 잎(답을 내는 마디)의 예측값 계산
node_indicator = tree_model.decision_path(X_train)
leaf_predict_counts = {"뇌졸중": 0, "아님": 0}

for leaf_id in leaf_indices:
    sample_mask = node_indicator[:, leaf_id].toarray().ravel().astype(bool)
    samples_y = y_train.reset_index(drop=True)[sample_mask]
    n_total = len(samples_y)
    n_positive = int(samples_y.sum())
    predicted_class = 1 if n_positive >= (n_total - n_positive) else 0
    if predicted_class == 1:
        leaf_predict_counts["뇌졸중"] += 1
    else:
        leaf_predict_counts["아님"] += 1

total_leaves = len(leaf_indices)
no_stroke_leaves = leaf_predict_counts["아님"]

st.write(f"- 답을 내는 마디(잎)는 모두 **{total_leaves}칸**입니다.")
st.write(f"- 그중 **{no_stroke_leaves}칸**이 '아님'이라고 답합니다.")

# 실제로 트리가 물어본 속성(질문에 사용된 feature)만 추출
used_features_idx = set(tree_.feature[tree_.feature >= 0])
used_features = [selected_features[i] for i in used_features_idx]

st.write("- 이 나무가 실제로 물어본 속성:")
for f in used_features:
    st.write(f"  - {f}")

if len(used_features) == 0:
    st.write("  - (트리가 한 번도 나뉘지 않아서 물어본 속성이 없습니다.)")
