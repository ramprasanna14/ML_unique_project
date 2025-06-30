# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from feature_engine.outliers import Winsorizer

# Page Config
st.set_page_config(page_title="⚡ SmartGrid Forecast", layout="wide")
st.title("⚡ Electricity Demand Forecasting & Grid Load Optimization")
st.markdown("> 💬 *“Smart grids aren’t the future – they’re the now. Measure. Forecast. Automate.”*")

# ----------- DATA LOADING & CLEANING ----------- #
@st.cache_data
def load_and_clean():
    df = pd.read_csv("energy_dataset1.csv")

    # Drop rows with missing target
    df.dropna(subset=['total load actual'], inplace=True)

    outlier_cols=['generation biomass', 'generation fossil brown coal/lignite',
       'generation fossil gas', 'generation fossil hard coal',
       'generation fossil oil', 'generation hydro pumped storage consumption',
       'generation hydro run-of-river and poundage',
       'generation hydro water reservoir',
       'generation nuclear', 'generation other', 'generation other renewable',
       'generation solar', 'generation waste', 'generation wind onshore',
       'forecast solar day ahead', 'forecast wind onshore day ahead',
       'total load actual', 'price day ahead', 'price actual']

    df_box = df[outlier_cols].copy()

    # Fill missing values
    missing_before = df.isna().sum()

    df.fillna(method='ffill', inplace=True)

    missing_after = df.isna().sum()
    df.drop(['generation wind offshore','generation marine','generation hydro pumped storage aggregated','generation fossil coal-derived gas','generation geothermal','forecast wind offshore eday ahead','generation fossil peat','generation fossil oil shale'],axis=1,inplace=True)
    # Feature engineering
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['month'] = df['time'].dt.month
    df['dayofweek'] = df['time'].dt.dayofweek
    df['hour'] = df['time'].dt.hour

    # Winsorization
    wins = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=outlier_cols, missing_values='ignore')
    df[outlier_cols] = wins.fit_transform(df[outlier_cols])

    return df, df_box, missing_before, missing_after

df, df_box, missing_before, missing_after = load_and_clean()


# ----------- SIDEBAR NAVIGATION ----------- #
st.sidebar.title("🔍 Navigation")
pages = ["Info", "EDA", "Prediction & Clustering", "Conclusion"]
choice = st.sidebar.radio("Go to", pages)

# ------------------- INFO ------------------- #
if choice == "Info":
    
    st.markdown("### 🎯 1. BACKSTORY & CONTEXT")
    st.markdown("""
    In the age of electrification and renewable transformation, traditional power grids are struggling to keep up with dynamic, non-linear demand patterns.

   Picture this:
It’s a scorching summer evening in a metro city. Homes crank up their air conditioners. Offices are still operational. EVs are charging in every second garage. Industries are running overnight production shifts. This synchronized energy pull leads to an unprecedented spike in electricity demand — and the grid buckles under pressure.
Despite having solar panels gleaming under the sun and wind turbines spinning off the coast, much of that renewable energy goes unused due to poor load forecasting and demand-response coordination.
Meanwhile, fossil-fuel plants are fired up just to meet the peak — leading to high carbon emissions, voltage instability, and blackouts.
    """)

    st.markdown("### 💥 2. PROBLEM STATEMENT")
    st.markdown("""
    Most grid operators rely on static regression models or rule-based forecasting that can't handle the stochastic behavior of today’s load profiles — driven by climate variability, distributed energy resources (DERs), and consumer-side fluctuations.
                
    🔹 **Goal 1:** Forecast total energy demand using supervised ML  
    🔹 **Goal 2:** Cluster periods by intensity using unsupervised ML

    This allows:
    - Avoiding grid overload
    - Smart renewable allocation
    - Demand flattening & emissions reduction
    """)

    st.markdown("### 🧑‍🤝‍🧑 3. STAKEHOLDERS")
    st.markdown("""
    - 🧠 Grid Operators  
    - 📈 Energy Analysts  
    - 🏭 Industrial Users  
    - 🏠 Smart Homes
    """)

    st.markdown("### 📦 4. DATASET INFO")
    st.markdown("""
    **Dataset:** [Integrated Energy Management & Forecasting (Kaggle)]  
    - Countries: multiple (Europe)  
    - Features: energy generation types, prices, load  
    - Duration: hourly data over years
    """)

    st.subheader("🔎 Sample Data")
    st.dataframe(df.head())

    st.markdown(f"**Shape of dataset:** `{df.shape}`")
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("🩺 Missing Values (Before Forward Fill)")
    st.dataframe(missing_before[missing_before > 0])

    st.success("✅ Missing values filled using Forward Fill (ffill)")
    st.subheader("🩺 Missing Values (After Fill)")
    st.dataframe(missing_after[missing_after > 0])

    st.subheader("🔁 Duplicate Rows")
    st.write(f"Duplicate entries in dataset: `{df.duplicated().sum()}`")

    st.subheader("📦 Outlier Visualization (Before Winsorization)")
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(data=df_box, orient='h', ax=ax)
    st.pyplot(fig)

    st.success("✅ Outliers treated using IQR-based Winsorization")

    st.subheader("📦 Boxplot After Winsorization")
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    sns.boxplot(data=df[df_box.columns], orient='h', ax=ax2)
    st.pyplot(fig2)

    
    
# ------------------- EDA ------------------- #
elif choice == "EDA":
    st.title("📈 Exploratory Data Analysis")

    eda_type = st.selectbox("Select EDA Type", ["Univariate", "Bivariate", "Multivariate"])

    if eda_type == "Univariate":
        col = st.selectbox("Choose column", df.select_dtypes(include='number').columns)
        #st.bar_chart(df[col])
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    elif eda_type == "Bivariate":
        x = st.selectbox("X-axis", df.select_dtypes(include='number').columns)
        y = st.selectbox("Y-axis", df.select_dtypes(include='number').columns)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x, y=y, ax=ax)
        st.pyplot(fig)

    elif eda_type == "Multivariate":
        st.subheader("📌 Correlation Heatmap")
        eda_features = ['generation biomass', 'generation fossil brown coal/lignite',
                        'generation fossil gas', 'generation fossil hard coal',
                        'generation nuclear', 'generation solar', 'generation wind onshore',
                        'generation waste', 'generation hydro water reservoir',
                        'price day ahead', 'hour', 'dayofweek', 'month']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[eda_features].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# ------------------- PREDICTION + CLUSTERING ------------------- #
elif choice == "Prediction & Clustering":
    st.title("🔋 Forecasting + Clustering")


    features = ['generation biomass', 'generation fossil brown coal/lignite',
       'generation fossil gas', 'generation fossil hard coal',
       'generation fossil oil', 'generation hydro pumped storage consumption',
       'generation hydro run-of-river and poundage',
       'generation hydro water reservoir', 'generation nuclear',
       'generation other', 'generation other renewable', 'generation solar',
       'generation waste', 'generation wind onshore',
       'forecast solar day ahead', 'forecast wind onshore day ahead',
       'price day ahead', 'price actual', 'month',
       'dayofweek', 'hour']
    target = 'total load actual'

    X = df[features]
    y = df[target]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # XGBoost Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    st.sidebar.header("🔢 Predict Load (Manual Input)")
    manual_input = []
    for f in features:
        val = st.sidebar.number_input(f, value=float(X[f].mean()), format="%.2f")
        manual_input.append(val)
    user_scaled = scaler.transform([manual_input])

    st.subheader("🔋 Step 1: Forecast Total Actual Load")
    if st.button("⚡ Predict Load"):
        pred_load = model.predict(user_scaled)[0]
        st.success(f"📊 **Predicted Load: {pred_load:.2f} MW**")
        st.markdown("📌 Use this load to perform clustering analysis below 👇")

    st.subheader("🔋 Step 2: Segment demand into clusters")
    if st.button("🔍 Perform Load Clustering"):
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df['cluster'] = cluster_labels

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        user_pca = pca.transform(user_scaled)
        user_cluster = kmeans.predict(user_scaled)[0]

        cluster_summary = df.groupby('cluster').agg({
            'total load actual': 'mean',
            'price day ahead': 'mean',
            'generation solar': 'mean',
            'generation wind onshore': 'mean',
            'generation nuclear': 'mean',
            'generation fossil gas': 'mean'
        }).rename(columns={
            'total load actual': 'Avg Load (MW)',
            'price day ahead': 'Avg Price',
            'generation solar': 'Solar',
            'generation wind onshore': 'Wind',
            'generation nuclear': 'Nuclear',
            'generation fossil gas': 'Gas'
        }).reset_index()

        sorted_clusters = cluster_summary.sort_values(by="Avg Load (MW)").reset_index(drop=True)
        cluster_roles = {
            sorted_clusters.loc[2, "cluster"]: {
                "Type": "🔥 Peak Load",
                "Description": "High demand with costly grid pressure. May require battery backup or demand reduction.",
                "Suggestions": ["Avoid EV charging", 
                                "Activate battery reserves",
                                "Send alerts to consumers"]
            },
            sorted_clusters.loc[0, "cluster"]: {
                "Type": "💤 Off-Peak",
                "Description": "Low demand period. Great for cost-efficient operations.",
                "Suggestions": ["Encourage EV charging",
                                "Store surplus energy",
                                 "Run maintenance"]
            },
            sorted_clusters.loc[1, "cluster"]: {
                "Type": "🔁 Shift Candidate",
                "Description": "Moderate load. Ideal for shifting demand from peak.",
                "Suggestions": ["Shift HVAC", 
                                "Smart schedule tasks", 
                                "Optimize grid"]
            }
        }

        cluster_summary["Type"] = cluster_summary["cluster"].map(lambda x: cluster_roles[x]["Type"])
        cluster_summary["Description"] = cluster_summary["cluster"].map(lambda x: cluster_roles[x]["Description"])
        cluster_summary = cluster_summary[[
        "cluster", "Type", "Description", "Avg Load (MW)", "Avg Price", "Solar", "Wind", "Nuclear", "Gas"]]
        st.markdown("### 📊 Cluster-Wise Load Characteristics")
        st.dataframe(cluster_summary)

        

        st.markdown("### 🔬 Cluster Visualization")
        fig, ax = plt.subplots()
        for i in range(3):
            ix = np.where(cluster_labels == i)
            ax.scatter(X_pca[ix, 0], X_pca[ix, 1], label=f"Cluster {i}")
        ax.scatter(user_pca[:, 0], user_pca[:, 1], color='red', edgecolors='black', s=150, label='Your Input')
        ax.set_title("PCA Cluster Distribution")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"### 🔍 You belong to Cluster {user_cluster} ({cluster_roles[user_cluster]['Type']})")
        st.markdown(f"**What it means:** {cluster_roles[user_cluster]['Description']}")
        for suggestion in cluster_roles[user_cluster]["Suggestions"]:
            st.markdown(f"✅ {suggestion}")
        
        st.markdown("### 🚀 Smart Grid Actions")
        if cluster_roles[user_cluster]["Type"] == "🔥 Peak Load":
            st.error("⚠️ Reduce non-critical load. Store renewable energy. Delay industrial tasks.")
        elif cluster_roles[user_cluster]["Type"] == "💤 Off-Peak":
            st.success("✅ Ramp up EV charging, storage and production.")
        else:
            st.info("🔁 Shift flexible load to Off-Peak (Cluster 1). Use storage.")


# ------------------- CONCLUSION ------------------- #
elif choice == "Conclusion":
    st.title("🎯 Project Conclusion")

    st.markdown("""
    ## 🧠 Key Takeaways
    - Accurate forecasting of total load can prevent overloads and energy losses.
    - Clustering helped segment load into actionable profiles: Peak, Off-Peak, Shiftable.
    - Demand forecasting + clustering = smarter, more flexible grid.

    ## 🌎 Real-World Impact
    - Prevent grid failures like California 2020 blackout.
    - Optimize renewable use and cut emissions.
    - Enable smart scheduling of EVs, HVACs, industries, etc.

    ## 🔋 Future Scope
    - Integrate weather + real-time IoT sensor data.
    - Deploy models on edge devices for real-time control.
    - Expand to multi-country or city-level segmentation.

    💡 *ML is not just for prediction — it's for action.*  
    """)



