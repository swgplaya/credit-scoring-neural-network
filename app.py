import pandas as pd
import streamlit as st

from src.predict import CreditRiskPredictor


st.set_page_config(
    page_title="Credit Default Risk",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> CreditRiskPredictor:
    return CreditRiskPredictor()


def payment_status_input(
    label: str,
    key: str,
) -> int:
    options = {
        "No consumption / no balance (-2)": -2,
        "Paid in full (-1)": -1,
        "Revolving credit / on time (0)": 0,
        "1 month late (1)": 1,
        "2 months late (2)": 2,
        "3 months late (3)": 3,
        "4 months late (4)": 4,
        "5 months late (5)": 5,
        "6 months late (6)": 6,
        "7 months late (7)": 7,
        "8 months late (8)": 8,
    }

    selected_label = st.selectbox(
        label,
        options=list(options.keys()),
        index=2,
        key=key,
    )

    return options[selected_label]


predictor = load_predictor()

st.title("Credit Default Risk Prediction")

st.write(
    "This application estimates the probability that a credit card "
    "client will default in the following month."
)

st.info(
    "This is an educational machine learning project. "
    "The prediction must not be used for real lending decisions."
)

with st.sidebar:
    st.header("Decision settings")

    threshold = st.slider(
        "Classification threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
    )

    st.caption(
        "A lower threshold identifies more potential defaults, "
        "but also produces more false alarms."
    )

st.subheader("Client profile")

profile_column_1, profile_column_2 = st.columns(2)

with profile_column_1:
    limit_balance = st.number_input(
        "Credit limit",
        min_value=0.0,
        value=100000.0,
        step=10000.0,
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
    )

    sex = st.selectbox(
        "Sex",
        options=[
            "Male",
            "Female",
        ],
    )

with profile_column_2:
    education = st.selectbox(
        "Education",
        options=[
            "Graduate school",
            "University",
            "High school",
            "Other",
        ],
    )

    marriage = st.selectbox(
        "Marital status",
        options=[
            "Married",
            "Single",
            "Other",
        ],
    )

sex_mapping = {
    "Male": 1,
    "Female": 2,
}

education_mapping = {
    "Graduate school": 1,
    "University": 2,
    "High school": 3,
    "Other": 4,
}

marriage_mapping = {
    "Married": 1,
    "Single": 2,
    "Other": 3,
}

st.subheader("Payment history")

payment_columns = st.columns(3)

payment_statuses = []

payment_labels = [
    "Most recent month",
    "2 months ago",
    "3 months ago",
    "4 months ago",
    "5 months ago",
    "6 months ago",
]

for index, label in enumerate(payment_labels):
    with payment_columns[index % 3]:
        payment_statuses.append(
            payment_status_input(
                label=label,
                key=f"pay_status_{index}",
            )
        )

st.subheader("Monthly statements and payments")

month_names = [
    "Most recent month",
    "2 months ago",
    "3 months ago",
    "4 months ago",
    "5 months ago",
    "6 months ago",
]

bill_amounts = []
payment_amounts = []

for index, month_name in enumerate(month_names):
    st.markdown(f"**{month_name}**")

    bill_column, payment_column = st.columns(2)

    with bill_column:
        bill_amounts.append(
            st.number_input(
                "Bill amount",
                value=0.0,
                step=1000.0,
                key=f"bill_amount_{index}",
            )
        )

    with payment_column:
        payment_amounts.append(
            st.number_input(
                "Payment amount",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                key=f"payment_amount_{index}",
            )
        )

predict_button = st.button(
    "Estimate default risk",
    type="primary",
    use_container_width=True,
)

if predict_button:
    client_data = pd.DataFrame([
        {
            "LIMIT_BAL": limit_balance,
            "SEX": sex_mapping[sex],
            "EDUCATION": education_mapping[education],
            "MARRIAGE": marriage_mapping[marriage],
            "AGE": age,
            "PAY_0": payment_statuses[0],
            "PAY_2": payment_statuses[1],
            "PAY_3": payment_statuses[2],
            "PAY_4": payment_statuses[3],
            "PAY_5": payment_statuses[4],
            "PAY_6": payment_statuses[5],
            "BILL_AMT1": bill_amounts[0],
            "BILL_AMT2": bill_amounts[1],
            "BILL_AMT3": bill_amounts[2],
            "BILL_AMT4": bill_amounts[3],
            "BILL_AMT5": bill_amounts[4],
            "BILL_AMT6": bill_amounts[5],
            "PAY_AMT1": payment_amounts[0],
            "PAY_AMT2": payment_amounts[1],
            "PAY_AMT3": payment_amounts[2],
            "PAY_AMT4": payment_amounts[3],
            "PAY_AMT5": payment_amounts[4],
            "PAY_AMT6": payment_amounts[5],
        }
    ])

    probability = predictor.predict_probability(
        client_data
    )

    predicted_class = int(
        probability >= threshold
    )

    st.divider()
    st.subheader("Prediction result")

    result_column_1, result_column_2 = st.columns(2)

    with result_column_1:
        st.metric(
            "Estimated default probability",
            f"{probability:.1%}",
        )

    with result_column_2:
        st.metric(
            "Decision threshold",
            f"{threshold:.0%}",
        )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    if predicted_class == 1:
        st.error(
            "The client is classified as higher risk "
            f"at the selected threshold of {threshold:.0%}."
        )
    else:
        st.success(
            "The client is classified as lower risk "
            f"at the selected threshold of {threshold:.0%}."
        )