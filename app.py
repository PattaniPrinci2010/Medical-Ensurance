
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Medical Insurance Prediction",
    page_icon="🏥",
    layout="wide"
)

# Sidebar
st.sidebar.title("🏥 MediCare")
st.sidebar.write("Medical Insurance System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Analytics",
        "Dataset Explorer",
        "Insurance Prediction",
        "Model Information",
        "About Project"
    ]
)

# Sample data for UI
df = pd.DataFrame({
    "Age": [19, 25, 31, 36, 42, 48, 52, 27],
    "Sex": ["female", "male", "female", "male",
            "female", "male", "female", "male"],
    "BMI": [22.5, 27.9, 24.1, 31.2, 28.4, 33.0, 29.5, 23.8],
    "Children": [0, 1, 2, 0, 3, 1, 2, 0],
    "Smoker": ["no", "no", "no", "yes",
               "no", "yes", "no", "no"],
    "Region": ["southwest", "southeast", "northwest",
               "northeast", "southwest", "southeast",
               "northwest", "northeast"],
    "Charges": [2500, 3200, 4100, 18000,
                5200, 22000, 6800, 2900]
})

# Dashboard
if page == "Dashboard":

    st.title("🏥 Medical Insurance Dashboard")
    st.write("Welcome to the Medical Insurance Prediction System")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", "1,338")
    col2.metric("Average Charges", "$13.2K")
    col3.metric("Average Age", "39 Years")
    col4.metric("Features", "6")

    st.divider()

    st.subheader("📊 Insurance Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Average Charges by Age Group")

        age_data = pd.DataFrame({
            "Age Group": ["18-25", "26-35", "36-45", "46-55", "56+"],
            "Charges": [5500, 8500, 12000, 16000, 19500]
        })

        st.bar_chart(age_data.set_index("Age Group"))

    with col2:
        st.write("Insurance Charges by Smoking Status")

        smoker_data = pd.DataFrame({
            "Smoking Status": ["Non-Smoker", "Smoker"],
            "Charges": [8400, 32000]
        })

        st.bar_chart(smoker_data.set_index("Smoking Status"))

    st.subheader("✨ Project Features")

    col1, col2, col3 = st.columns(3)

    col1.info("📊 Data Analytics\n\nExplore insurance data.")
    col2.info("🔮 Prediction Form\n\nEnter customer details.")
    col3.info("📁 Dataset Explorer\n\nView sample records.")

# Analytics
elif page == "Analytics":

    st.title("📊 Insurance Analytics")
    st.write("Explore insurance data with charts.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        st.bar_chart(df["Age"].value_counts().sort_index())

    with col2:
        st.subheader("Smoking Status")
        st.bar_chart(df["Smoker"].value_counts())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regional Distribution")
        st.bar_chart(df["Region"].value_counts())

    with col2:
        st.subheader("Charges Distribution")
        st.line_chart(df["Charges"])

    st.subheader("BMI Analysis")
    st.scatter_chart(df, x="BMI", y="Charges")

# Dataset Explorer
elif page == "Dataset Explorer":

    st.title("📁 Dataset Explorer")
    st.write("Explore and search insurance records.")
    st.divider()

    search = st.text_input("🔍 Search dataset")

    filtered_df = df.copy()

    if search:
        mask = filtered_df.astype(str).apply(
            lambda col: col.str.contains(
                search, case=False, na=False
            )
        ).any(axis=1)

        filtered_df = filtered_df[mask]

    col1, col2 = st.columns(2)

    col1.metric("Records", len(filtered_df))
    col2.metric("Columns", len(filtered_df.columns))

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(info_df, use_container_width=True)

    st.download_button(
        "⬇️ Download Dataset",
        data=filtered_df.to_csv(index=False),
        file_name="medical_insurance.csv",
        mime="text/csv"
    )

# Insurance Prediction
elif page == "Insurance Prediction":

    st.title("🔮 Medical Insurance Prediction")
    st.write("Enter customer information below.")
    st.divider()

    with st.form("insurance_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )

            sex = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=70.0,
                value=25.0
            )

        with col2:
            children = st.selectbox(
                "Number of Children",
                [0, 1, 2, 3, 4, 5]
            )

            smoker = st.selectbox(
                "Smoker",
                ["No", "Yes"]
            )

            region = st.selectbox(
                "Region",
                ["Southwest", "Southeast", "Northwest", "Northeast"]
            )

        submitted = st.form_submit_button(
            "Submit Customer Details"
        )

    if submitted:
        st.success("Customer details submitted successfully!")

        st.subheader("Customer Information")

        st.write("Age:", age)
        st.write("Gender:", sex)
        st.write("BMI:", bmi)
        st.write("Children:", children)
        st.write("Smoker:", smoker)
        st.write("Region:", region)

        st.warning(
            "This is only the UI. Connect your trained model "
            "to display actual predicted insurance charges."
        )

# Model Information
elif page == "Model Information":

    st.title("🧠 Model Information")
    st.divider()

    st.subheader("Decision Tree Regression")

    st.write(
        "Decision Tree Regression is a machine learning algorithm "
        "used to predict continuous numerical values."
    )

    st.subheader("Project Workflow")

    st.info("1. Data Collection")
    st.info("2. Data Preprocessing")
    st.info("3. Exploratory Data Analysis")
    st.info("4. Model Training")
    st.info("5. Model Evaluation")
    st.info("6. Insurance Charges Prediction")

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "Not Connected")
    col2.metric("RMSE", "Not Connected")
    col3.metric("R² Score", "Not Connected")

# About Project
elif page == "About Project":

    st.title("ℹ️ About Project")
    st.divider()

    st.subheader("Medical Insurance Prediction System")

    st.write(
        "This project provides a user interface for exploring "
        "medical insurance data and entering customer details."
    )

    st.subheader("Technologies Used")

    st.success("Python")
    st.success("Streamlit")
    st.success("Pandas")
    st.success("Machine Learning")

    st.subheader("Project Details")

    st.write("Project Name: Medical Insurance Prediction")
    st.write("Interface: Streamlit")
    st.write("Application Type: UI Prototype")
    st.write("Prediction Model: Not Connected")

    st.caption("Medical Insurance Prediction | Streamlit UI")
