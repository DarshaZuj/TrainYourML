import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from streamlit_lottie import st_lottie
import json
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load lottie animation function
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Set Streamlit page config
st.set_page_config(page_title="TrainYourML", layout="wide")

# Inject CSS
st.markdown("""
    <style>
    header, .css-18e3th9 {
        background-color: #deddd7;
        padding-top: 10px;
    }
    .main {
        background-color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        color: #000;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        font-weight: bold;
        border-bottom: 2px solid #ff4b4b;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 10.0rem;
        font-weight: bold;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)



tab1, tab2, tab3, tab4 = st.tabs(["\U0001F3E0 Home", "⚙️ Model Explorer", "\U0001F4D8 User Guide", "ℹ️ About"])

# --- Home Tab ---
with tab1:
    # Load all three Lottie animations
    lottie_center = load_lottiefile("assets/Animation - 1746327305438.json")
    lottie_left = load_lottiefile("assets/Animation - 1746337360103.json")
    lottie_right = load_lottiefile("assets/Animation - 1746337731316.json")

    st.markdown("""
        <style>
            .home-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 50px 20px;
                font-family: 'Segoe UI', sans-serif;
                text-align: center;
            }
            .home-title {
                font-size: 3.5em;
                font-weight: bold;
                color: #141414;
                margin-bottom: 0.2em;
            }
            .home-tagline {
                font-size: 1.5em;
                color: #333333;
                margin-bottom: 2em;
            }
            .home-description {
                max-width: 800px;
                font-size: 1.6em;
                color: #555;
                margin-top: 0em;
            }
        </style>
        <div class="home-container">
            <div class="home-title">TrainYourML</div>
            <div class="home-tagline">ML made simple, smart, and yours.</div>
        </div>
    """, unsafe_allow_html=True)

    # Display three animations: left, center, right
    col1, col2, col3 = st.columns(3)
    with col1:
        st_lottie(lottie_left, height=280, key="home_animation_left")
    with col2:
        st_lottie(lottie_center, height=250, key="home_animation_center")
    with col3:
        st_lottie(lottie_right, height=250, key="home_animation_right")

    # Description below the animations
    st.markdown("""
        <div class="home-container">
            <div class="home-description">
                TrainYourML empowers you to upload your own datasets and explore machine learning effortlessly.<br><br>
                Whether it's classification or regression, feature selection or model comparison — you're in control.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
            <div style='text-align: center; padding: 20px; font-size: 14px; color: gray;'>
            © 2025 TrainYourML | Built using Streamlit<br>
            <a href='https://github.com/your-repo' target='_blank'>GitHub</a> |
            <a href='mailto:darshanazujam@gmail'>Contact</a>
            </div>
        """, unsafe_allow_html=True)



# --- App Tab ---
with tab2:
    st.title("Customize your ML Model 🤖")
    with st.expander("Upload and Configure", expanded=True):
        uploaded_file = st.file_uploader("1. Upload CSV or TSV file", type=["csv", "tsv"])
        df, target_column, problem_type, X, y = None, None, None, None, None

        if uploaded_file is not None:
            sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
            df = pd.read_csv(uploaded_file, sep=sep)
            st.subheader("Dataset Preview")
            st.dataframe(df.head(15))

            target_column = st.selectbox("2. Select Target Column", df.columns)
            problem_type = "classification" if df[target_column].dtype == "object" or len(df[target_column].unique()) <= 10 else "regression"
            st.markdown(f"**Problem Type:** {problem_type.capitalize()}")

            all_features = [col for col in df.columns if col != target_column]
            feature_columns = st.multiselect("3. Select Feature Columns", all_features, default=all_features)

            df = df[feature_columns + [target_column]].dropna()
            X = df[feature_columns].copy()
            y = df[target_column].copy()

            for col in X.select_dtypes(include=["object"]).columns:
                X[col] = LabelEncoder().fit_transform(X[col])

            if problem_type == "classification" and y.dtype == "object":
                y = LabelEncoder().fit_transform(y)

            st.session_state.update({
                'df': df, 'X': X, 'y': y,
                'target_column': target_column,
                'problem_type': problem_type,
                'feature_columns': feature_columns
            })

        if 'df' in st.session_state:
            df = st.session_state.df
            X = st.session_state.X
            y = st.session_state.y
            target_column = st.session_state.target_column
            problem_type = st.session_state.problem_type
            feature_columns = st.session_state.feature_columns

            use_feature_selection = st.checkbox("Enable Automatic Feature Selection")
            if use_feature_selection:
                score_func = f_classif if problem_type == "classification" else f_regression
                k = st.slider("Number of top features to select", 1, len(feature_columns), min(5, len(feature_columns)))
                selector = SelectKBest(score_func=score_func, k=k)
                X_new = selector.fit_transform(X, y)
                selected_features = selector.get_support(indices=True)
                selected_column_names = [feature_columns[i] for i in selected_features]
                st.markdown("**Selected Features:**")
                st.write(selected_column_names)
                X = pd.DataFrame(X_new, columns=selected_column_names)

            test_size = st.slider("Test set size", 0.1, 0.5, 0.3)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            st.subheader("Model Comparison")
            selected_models = st.multiselect("4. Select Models to Compare", [
                "Random Forest", "Logistic Regression", "SVM",
                "Gradient Boosting", "KNN", "Decision Tree",
                "AdaBoost", "XGBoost"
            ], default=["Random Forest", "Logistic Regression"])

            model_map = {
                "Random Forest": RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "SVM": SVC(probability=True) if problem_type == "classification" else SVR(),
                "Gradient Boosting": GradientBoostingClassifier() if problem_type == "classification" else GradientBoostingRegressor(),
                "KNN": KNeighborsClassifier() if problem_type == "classification" else KNeighborsRegressor(),
                "Decision Tree": DecisionTreeClassifier() if problem_type == "classification" else DecisionTreeRegressor(),
                "AdaBoost": AdaBoostClassifier() if problem_type == "classification" else AdaBoostRegressor(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss') if problem_type == "classification" else xgb.XGBRegressor()
            }

            if st.button("Compare Models"):
                results = []
                for name in selected_models:
                    model = model_map[name]
                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("model", model)
                    ])
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    if problem_type == "classification":
                        score = pipeline.score(X_test, y_test)
                        results.append({
                            "Model": name,
                            "Accuracy": score,
                            "F1 Score": f"{classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']:.4f}"
                        })
                    else:
                        results.append({
                            "Model": name,
                            "R² Score": r2_score(y_test, y_pred),
                            "MSE": mean_squared_error(y_test, y_pred)
                        })

                st.subheader("Comparison Results")
                st.dataframe(pd.DataFrame(results))

            # Train Model Option
            model_type = st.selectbox("Train a Model", list(model_map.keys()))

            if st.button("Train Model"):
                model = model_map[model_type]
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", model)
                ])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                st.success("Model trained successfully!")
                st.subheader("Evaluation Results")

                if problem_type == "classification":
                    st.text("Classification Report")
                    st.text(classification_report(y_test, y_pred))
                    st.text("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)
                else:
                    st.text(f"R² Score: {r2_score(y_test, y_pred):.4f}")
                    st.text(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

                with open("trained_pipeline.pkl", "wb") as f:
                    joblib.dump(pipeline, f)
                st.download_button("Download Trained Pipeline", data=open("trained_pipeline.pkl", "rb").read(), file_name="trained_pipeline.pkl")
        else:
            st.info("Please upload a dataset to get started.")

# --- User Guide Tab ---
with tab3:
    st.title("User Guide")
    st.markdown("""
        ## 🚀 Welcome to TrainYourML! 🚀
        TrainYourML is your go-to open-source AutoML tool to easily upload datasets, train models, and compare performance — all with just a few clicks! 🖱️ Whether you're working on classification or regression tasks, we’ve got you covered! Let's dive in and get started. 😎

        ### 📝 Getting Started
        1. **Upload Your Dataset:**
            - Click on the "Upload CSV or TSV file" button to upload your dataset. 📂
            - We support CSV and TSV formats, so you’re good to go if your data is in those formats.
        2. **Select the Target Column:**
            - Choose the column you want to predict! 🏆
            - Pick your target column from the list of columns in your dataset.
        3. **Choose Your Features:**
            - You can either select the features manually or enable Automatic Feature Selection! 🔍
            - We’ll help you pick the top features using statistical methods, so you can focus on the important ones!
        4. **Split Your Data:**
            - Choose the size of your test set, and we’ll handle the rest by splitting your data into training and testing sets. 📊
        5. **Model Comparison:**
            - Want to compare multiple models? 🤖
            - Select from a variety of models (like Random Forest, Logistic Regression, XGBoost, and more), and let the app work its magic! 🚀
            - We’ll display the performance comparison of each model, so you can easily choose the best one for your problem.
        6. **Train a Model:**
            - Pick a single model to train. Once trained, we’ll evaluate it using the test set and show you key metrics like accuracy, precision, recall, R², and Mean Squared Error. 📈
            - After training, you can download your trained model as a `.pkl` file to use later! 💾

        
        ### 📦 How to Use the Exported `.pkl` Model in Python
        After downloading your trained model (`trained_pipeline.pkl`), you can use it like this:

        ```python
        import joblib
        import pandas as pd

        # Load your trained pipeline
        model = joblib.load('trained_pipeline.pkl')

        # Prepare new input data
        new_data = pd.DataFrame({
        "feature1": [value1],
        "feature2": [value2],
        # Add all required features
        })

        # Use your model to make predictions
        predictions = model.predict(new_data)

        # Display the result
        print("Predictions:", predictions)
        
        ```

        We hope this guide helps you get the most out of TrainYourML. Happy modeling! 🎉
""")
    st.markdown("---")
    st.markdown("""
            <div style='text-align: center; padding: 20px; font-size: 14px; color: gray;'>
            © 2025 TrainYourML | Built using Streamlit<br>
            <a href='https://github.com/your-repo' target='_blank'>GitHub</a> |
            <a href='mailto:darshanazujam@gmail'>Contact</a> 
            </div>
        """, unsafe_allow_html=True)




# --- About Tab ---
with tab4:
    st.title("About")
    st.markdown("""
        Hello! I'm **Darshana V Zujam** 👋
        
        I’m an MSc Bioinformatics student at DES Pune Univrsity, Pune, Maharashtra, India. I created TrainYourML to make machine learning more accessible and fun for everyone – especially beginners! 🎉 Whether you're just starting your ML journey or looking for a simple tool to play around with, this app has got you covered. 🤖

        ## 🎯 Purpose Behind TrainYourML

        The goal behind TrainYourML is simple: I wanted to create a platform that allows anyone – no matter their experience level – to upload datasets, explore machine learning models, and start training them with just a few clicks! ✨ This app lets you focus on the fun part of ML while doing the heavy lifting behind the scenes.

        ## 💡 Why Should You Use TrainYourML?
        Here’s why you’ll love it:

        1. **Easy Upload** 📂: Upload your dataset in CSV or TSV format and start working right away.

        2. **Automatic Feature Selection** 🔍: Let the app automatically choose the best features for you!

        3. **Model Comparison** 🤖: Compare multiple models side-by-side to find the one that works best.

        4. **Train Your Models** 🏋️: Train your chosen models, evaluate them, and make predictions with just a few clicks.

        5. **Download Your Trained Model** 💾: Export your model as a .pkl file and use it for predictions later.

        6. **No Need for Code** 🧑‍💻: Don’t worry about writing code – everything’s done for you in a simple, interactive interface!

        ## 🛠️ Tools and Technologies Used
        This app wouldn’t have been possible without these amazing tools and technologies:

        + **Streamlit**: For building the interactive web app 🌐

        + **Scikit-learn**: For all the machine learning algorithms and models 🤖

        + **XGBoost**: For powerful gradient boosting models 🚀

        + **Pandas**: For data manipulation and processing 📊

        + **NumPy**: For numerical computations 🔢

        + **Matplotlib & Seaborn**: For data visualization 📈

        + **Joblib**: For saving and loading models 💾

        + **Lottie**: For fun and engaging animations ✨

        + **Python**: The language that makes everything work 🐍

        ## 🙏 Acknowledgements
        A huge thank you to my professor, **Dr. Kushagra Kashyap**, for being an amazing mentor. His guidance and support were invaluable! 🙌 I also want to express my gratitude to DES Pune University for providing me with the resources and environment to create this app. 🎓

        ## 📬 Get in Touch
        Feel free to reach out or connect with me! I’m always open to feedback, collaboration, or just talking about ML! 😄

        + **GitHub**: Visit my GitHub Repo

        + **LinkedIn**: [Darshana Zujam](www.linkedin.com/in/darshana-zujam-66a5aa285)

        + **Email**: [darshanazujam@gmail.com](darshanazujam@gmail.com)


    
        I hope you enjoy using TrainYourML as much as I enjoyed building it! 😎 Happy modeling! 🎉
""")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 20px; font-size: 14px; color: gray;'>
        © 2025 TrainYourML | Built using Streamlit<br>
        <a href='https://github.com/your-repo' target='_blank'>GitHub</a> |
        <a href='mailto:darshanazujam@gmail'>Contact</a> 
        </div>
    """, unsafe_allow_html=True)
