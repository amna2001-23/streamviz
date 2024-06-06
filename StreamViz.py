import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Page 1: Data Upload
def page_data_upload():
    st.title("Data Upload")
    st.write("Upload your CSV or Excel file.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  # or pd.read_excel(uploaded_file)
        st.write("Preview of the uploaded data:")
        st.write(df.head())
        st.session_state['data'] = df
        st.success("File uploaded successfully!")

# Page 2: Data Cleaning and Encoding
def data_Encoding():
    st.header("Data Encode")

    # Get DataFrame from session state
    if 'data' not in st.session_state:
        st.warning("Please upload data from the Data Upload section first.")
        return

    df = st.session_state['data']

    # Encode Categorical Variables
    st.subheader("Encode Categorical Variables")
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        target_column = st.selectbox("Select target column for encoding:", categorical_columns)
        if target_column:
            if st.button("Encode Target Variable"):
                le = LabelEncoder()
                df[target_column] = le.fit_transform(df[target_column])
                st.session_state['data'] = df
                st.success(f"Target column '{target_column}' encoded.")
    else:
        st.info("No categorical columns available for encoding.")

    if 'data' in st.session_state:
        st.subheader("Transformed Data")
        st.write(df)

    # Select Target and Input Features
    def select_target_and_features():
        st.header("Select Target and Input Features")

        if 'data' not in st.session_state:
            st.warning("Please upload data from the Data Upload section first.")
            return

        df = st.session_state['data']
        columns = df.columns.tolist()

        st.session_state['target'] = st.selectbox("Select Target Column:", columns)
        st.session_state['inputs'] = st.multiselect("Select Input Feature Columns:", [col for col in columns if col != st.session_state['target']])

        if st.button("Set Target and Features"):
            st.success(f"Target and features set.\nTarget: {st.session_state['target']}\nFeatures: {st.session_state['inputs']}")

    select_target_and_features()

    

# Page 3: Visualization


# Page 4: Classification and Visualization
def page_classification():
    st.header("Classification")

    # Check if data is uploaded
    if st.session_state.get('data') is None:
        st.warning("Please upload data from the Data Upload section first.")
        return

    df = st.session_state['data']

    # Check if the target column is selected
    if 'target' not in st.session_state or 'inputs' not in st.session_state:
        st.warning("Please select the target column and input features in the Data Cleaning and Encoding section.")
        return

    target_column = st.session_state['target']
    input_features = st.session_state['inputs']

    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in the uploaded data.")
        return

    # Model Selection
    model_options = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    selected_model_name = st.selectbox("Select a classification model:", list(model_options.keys()))
    selected_model = model_options[selected_model_name]

    # Perform Classification
    X = df[input_features]
    y = df[target_column]

    # Print the shape and other information about X and y
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Info about X:", X.info())
    print("Info about y:", y.info())

    # Check for missing values in X and y
    print("Missing values in X:", X.isnull().sum())
    print("Missing values in y:", y.isnull().sum())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the shape of X_train, X_test, y_train, y_test
    print("Shapes after train-test split:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # Fit the model and handle the error if any
    try:
        selected_model.fit(X_train, y_train)
        y_pred = selected_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Display classification results
        st.write("Classification Results:")
        st.write("Model:", selected_model_name)
        st.write("Accuracy:", selected_model.score(X_test, y_test))
        st.write("Confusion Matrix:")
        st.write(cm)

    except Exception as e:
        st.error(f"Error occurred during model fitting: {str(e)}")



def page_visualization():
    st.write("Data Visualization")

    # Check if data is uploaded
    if st.session_state.get('data') is None:
        st.warning("Please upload data from the Data Collection section first.")
        return

    df = st.session_state['data']

    # Allow user to select a plot type
    plot_type = st.selectbox("Select a plot type:", ["Pie Chart", "Bar Plot", "Heatmap", "Distribution Plot", "Violin Plot", "Box Plot"])

    if plot_type:
        # For plots that require selecting two columns
        if plot_type in ["Violin Plot", "Box Plot"]:
            # Allow user to select two columns for visualization
            selected_columns = st.multiselect("Select two columns for visualization:", df.columns)

            if len(selected_columns) != 2:
                st.warning("Please select exactly two columns for this plot type.")
                return

            if st.button("Generate Plot"):
                # Plot based on user selection
                if plot_type == "Violin Plot":
                    st.subheader("Violin Plot")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.violinplot(data=df, x=selected_columns[0], y=selected_columns[1], ax=ax)
                    ax.set_xlabel(selected_columns[0])
                    ax.set_ylabel(selected_columns[1])
                    ax.set_title('Violin Plot')
                    st.pyplot(fig)

                elif plot_type == "Box Plot":
                    st.subheader("Box Plot")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.boxplot(data=df, x=selected_columns[0], y=selected_columns[1], ax=ax)
                    ax.set_xlabel(selected_columns[0])
                    ax.set_ylabel(selected_columns[1])
                    ax.set_title('Box Plot')
                    st.pyplot(fig)
        else:
            # Allow user to select a single column for visualization
            selected_column = st.selectbox("Select a column for visualization:", df.columns)

            if st.button("Generate Plot"):
                # Plot based on user selection
                if plot_type == "Pie Chart":
                    st.subheader("Pie Chart")
                    pie_data = df[selected_column].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
                    st.pyplot(fig)

                elif plot_type == "Bar Plot":
                    st.subheader("Bar Plot")
                    bar_data = df[selected_column].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x=bar_data.index, y=bar_data.values, ax=ax)
                    ax.set_xlabel(selected_column)
                    ax.set_ylabel('Count')
                    ax.set_title('Bar Plot')
                    st.pyplot(fig)

                elif plot_type == "Heatmap":
                    st.subheader("Heatmap")
                    corr = df.corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title('Heatmap')
                    st.pyplot(fig)

                elif plot_type == "Distribution Plot":
                    st.subheader("Distribution Plot")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(data=df, x=selected_column, kde=True, ax=ax)
                    ax.set_xlabel(selected_column)
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution Plot')
                    st.pyplot(fig)

# Navbar
with st.sidebar:
    selected = option_menu(
        "Main Menu", ["Data Upload", "Encoding", "Classification", "Visualization"],
        icons=["cloud-upload", "pencil-square",  "check2-circle", "bar-chart"],
        menu_icon="cast", default_index=0,
    )
# Main Content
if selected == "Data Upload":
    page_data_upload()
elif selected == "Encoding":  # Corrected the page name
    data_Encoding()  # Corrected the function name
elif selected == "Classification":
    page_classification()
elif selected == "Visualization":
    page_visualization()

