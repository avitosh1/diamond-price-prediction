import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Diamond Dynamics",
    page_icon="💎",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("../data/processed_diamonds.csv")

df = load_data()

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():

    regression_model = joblib.load(
        "../models/best_model.pkl"
    )

    encoder = joblib.load(
        "../models/encoder.pkl"
    )

    scaler = joblib.load(
        "../models/scaler.pkl"
    )

    cluster_model = joblib.load(
        "../models/best_cluster_model.pkl"
    )

    cluster_encoder = joblib.load(
        "../models/cluster_encoder.pkl"
    )

    cluster_scaler = joblib.load(
        "../models/cluster_scaler.pkl"
    )

    cluster_names = joblib.load(
        "../models/cluster_names.pkl"
    )

    best_model_name = joblib.load(
        "../models/best_model_name.pkl"
    )

    metrics_df = pd.read_csv(
        "../models/model_metrics.csv",
        index_col=0
    )

    return (
        regression_model,
        encoder,
        scaler,
        cluster_model,
        cluster_encoder,
        cluster_scaler,
        cluster_names,
        best_model_name,
        metrics_df
    )

(
    regression_model,
    encoder,
    scaler,
    cluster_model,
    cluster_encoder,
    cluster_scaler,
    cluster_names,
    best_model_name,
    metrics_df
) = load_models()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("💎 Diamond Dynamics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Price Prediction",
        "Analytics"
    ]
)

# =====================================
# DASHBOARD
# =====================================

if page == "Dashboard":

    st.title("💎 Diamond Dynamics")

    st.markdown("""
    AI-Powered Diamond Price Prediction &
    Market Segmentation System
    """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Records", f"{len(df):,}")

    with c2:
        st.metric("Features", len(df.columns))

    with c3:
        st.metric("Best Model", best_model_name)

    with c4:
        st.metric(
            "Best R²",
            round(metrics_df["R²"].max(), 4)
        )

    st.divider()

    st.subheader("🏅 Selected Model")

    st.success(
        f"""
        Model: {best_model_name}

        RMSE: {metrics_df.loc[best_model_name,'RMSE']:.4f}

        R² Score: {metrics_df.loc[best_model_name,'R²']:.4f}
        """
    )

    st.divider()

    st.subheader("📈 Model Performance")

    st.dataframe(
        metrics_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("🎯 Market Segments")

    cluster_df = pd.DataFrame(
        {
            "Cluster ID": list(cluster_names.keys()),
            "Cluster Name": list(cluster_names.values())
        }
    )

    st.dataframe(
        cluster_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

# =====================================
# PRICE PREDICTION
# =====================================

elif page == "Price Prediction":

    st.title("💰 Diamond Price Prediction")

    cut_options = [
        'Fair',
        'Good',
        'Very Good',
        'Premium',
        'Ideal'
    ]

    color_options = [
        'J',
        'I',
        'H',
        'G',
        'F',
        'E',
        'D'
    ]

    clarity_options = [
        'I1',
        'SI2',
        'SI1',
        'VS2',
        'VS1',
        'VVS2',
        'VVS1',
        'IF'
    ]

    st.markdown(
        "Generate a random diamond or enter values manually."
    )

    # =====================================
    # SAMPLE GENERATOR
    # =====================================

    if st.button("🎲 Generate Sample Diamond"):

        sample = df.sample(1).iloc[0]

        st.session_state["carat"] = float(sample["carat"])
        st.session_state["depth"] = float(sample["depth"])
        st.session_state["table"] = float(sample["table"])
        st.session_state["y"] = float(sample["y"])
        st.session_state["z"] = float(sample["z"])
        st.session_state["volume"] = float(sample["volume"])

        st.session_state["cut"] = sample["cut"]
        st.session_state["color"] = sample["color"]
        st.session_state["clarity"] = sample["clarity"]

        st.rerun()

    col1, col2 = st.columns(2)

    # =====================================
    # INPUTS
    # =====================================

    with col1:

        carat = st.number_input(
            "Carat",
            min_value=0.01,
            value=float(
                st.session_state.get(
                    "carat",
                    1.0
                )
            )
        )

        depth = st.number_input(
            "Depth",
            min_value=0.0,
            value=float(
                st.session_state.get(
                    "depth",
                    61.0
                )
            )
        )

        table = st.number_input(
            "Table",
            min_value=0.0,
            value=float(
                st.session_state.get(
                    "table",
                    57.0
                )
            )
        )

        y = st.number_input(
            "Y Dimension",
            min_value=0.01,
            value=float(
                st.session_state.get(
                    "y",
                    5.5
                )
            )
        )

        z = st.number_input(
            "Z Dimension",
            min_value=0.01,
            value=float(
                st.session_state.get(
                    "z",
                    3.5
                )
            )
        )

        volume = st.number_input(
            "Volume",
            min_value=0.01,
            value=float(
                st.session_state.get(
                    "volume",
                    100.0
                )
            )
        )

    with col2:

        cut = st.selectbox(
            "Cut",
            cut_options,
            index=cut_options.index(
                st.session_state.get(
                    "cut",
                    "Ideal"
                )
            )
        )

        color = st.selectbox(
            "Color",
            color_options,
            index=color_options.index(
                st.session_state.get(
                    "color",
                    "G"
                )
            )
        )

        clarity = st.selectbox(
            "Clarity",
            clarity_options,
            index=clarity_options.index(
                st.session_state.get(
                    "clarity",
                    "VS1"
                )
            )
        )

    st.divider()

    # =====================================
    # PREDICT BUTTON
    # =====================================

    predict_btn = st.button(
        "🚀 Predict Diamond Price",
        use_container_width=True,
        type="primary"
    )

    if predict_btn:

        try:

            # =====================================
            # REGRESSION
            # =====================================

            cat_features = encoder.transform(
                [[cut, color, clarity]]
            )

            num_features = scaler.transform(
                [[
                    carat,
                    depth,
                    table,
                    y,
                    z,
                    volume
                ]]
            )

            X_pred = np.hstack(
                [num_features, cat_features]
            )

            prediction_log = regression_model.predict(
                X_pred
            )[0]

            prediction_inr = np.expm1(
                prediction_log
            )

            prediction_usd = (
                prediction_inr / 85
            )

            # =====================================
            # CLUSTERING
            # =====================================

            cluster_cat_df = pd.DataFrame(
                [[cut, color, clarity]],
                columns=[
                    "cut",
                    "color",
                    "clarity"
                ]
            )

            cluster_cat = (
                cluster_encoder.transform(
                    cluster_cat_df
                )
            )

            cluster_num_df = pd.DataFrame(
                [[
                    carat,
                    depth,
                    table,
                    y,
                    z,
                    volume
                ]],
                columns=[
                    "carat",
                    "depth",
                    "table",
                    "y",
                    "z",
                    "volume"
                ]
            )

            cluster_num = (
                cluster_scaler.transform(
                    cluster_num_df
                )
            )

            X_cluster = np.hstack(
                [cluster_num, cluster_cat]
            )

            cluster_id = int(
                cluster_model.predict(
                    X_cluster
                )[0]
            )

            cluster_name = (
                cluster_names.get(
                    cluster_id,
                    "Unknown Segment"
                )
            )

            # =====================================
            # RESULTS
            # =====================================

            st.success(
                "Prediction Complete!"
            )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "🇮🇳 Estimated Price (INR)",
                    f"₹ {prediction_inr:,.0f}"
                )

            with c2:

                st.metric(
                    "🇺🇸 Estimated Price (USD)",
                    f"$ {prediction_usd:,.0f}"
                )

            st.divider()

            st.subheader(
                "🎯 Market Segment"
            )

            st.info(cluster_name)

            if (
                cluster_name
                == "Premium Diamonds"
            ):

                st.success(
                    """
                    Premium Segment

                    • High-value diamonds

                    • Luxury market category

                    • Higher average prices

                    • Strong investment potential
                    """
                )

            elif (
                cluster_name
                == "Affordable Diamonds"
            ):

                st.info(
                    """
                    Affordable Segment

                    • Budget-friendly diamonds

                    • Suitable for retail buyers

                    • Lower average market value

                    • Excellent value for money
                    """
                )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

# =====================================
# ANALYTICS
# =====================================

elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    # =====================================
    # DATASET SUMMARY
    # =====================================

    st.subheader("📌 Dataset Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with c2:
        st.metric(
            "Total Features",
            len(df.columns)
        )

    with c3:

        avg_price = np.expm1(
            df["price_inr"]
        ).mean()

        st.metric(
            "Average Price (INR)",
            f"₹ {avg_price:,.0f}"
        )

    st.divider()

    # =====================================
    # MODEL PERFORMANCE
    # =====================================

    st.subheader(
        "🏆 Model Performance Comparison"
    )

    st.dataframe(
        metrics_df,
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    metrics_df["R²"].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "R² Comparison"
    )

    ax.set_ylabel(
        "R² Score"
    )

    st.pyplot(fig)

    st.divider()

    # =====================================
    # CORRELATION MATRIX
    # =====================================

    st.subheader(
        "🔥 Correlation Heatmap"
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    # =====================================
    # PRICE DISTRIBUTION
    # =====================================

    st.subheader(
        "💰 Price Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        np.expm1(df["price_inr"]),
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_xlabel(
        "Price (INR)"
    )

    st.pyplot(fig)

    st.divider()

    # =====================================
    # CARAT VS PRICE
    # =====================================

    st.subheader(
        "💎 Carat vs Price"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.scatterplot(
        data=df,
        x="carat",
        y=np.expm1(df["price_inr"]),
        alpha=0.5,
        ax=ax
    )

    ax.set_ylabel(
        "Price (INR)"
    )

    st.pyplot(fig)

    st.divider()

    # =====================================
    # CUT DISTRIBUTION
    # =====================================

    st.subheader(
        "✨ Cut Quality Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.countplot(
        data=df,
        x="cut",
        order=df["cut"].value_counts().index,
        ax=ax
    )

    plt.xticks(
        rotation=30
    )

    st.pyplot(fig)

    st.divider()

    # =====================================
    # CLUSTER DISTRIBUTION
    # =====================================

    st.subheader(
        "🎯 Market Segmentation"
    )

    if "cluster_name" in df.columns:

        cluster_counts = (
            df["cluster_name"]
            .value_counts()
        )

        st.bar_chart(
            cluster_counts
        )

        st.dataframe(
            cluster_counts
            .reset_index(),
            use_container_width=True
        )

    st.divider()

    # =====================================
    # PCA VISUALIZATION
    # =====================================

    if "cluster" in df.columns:

        st.subheader(
            "🧠 PCA Cluster Visualization"
        )

        pca_features = [
            "carat",
            "depth",
            "table",
            "y",
            "z",
            "volume"
        ]

        pca_features = [
            col
            for col in pca_features
            if col in df.columns
        ]

        pca = PCA(
            n_components=2
        )

        pca_values = pca.fit_transform(
            df[pca_features]
        )

        pca_df = pd.DataFrame(
            pca_values,
            columns=[
                "PC1",
                "PC2"
            ]
        )

        pca_df["cluster"] = (
            df["cluster"]
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="cluster",
            palette="tab10",
            alpha=0.7,
            ax=ax
        )

        st.pyplot(fig)

    st.divider()

    # =====================================
    # BUSINESS INSIGHTS
    # =====================================

    st.subheader(
        "📖 Key Insights"
    )

    st.success(
        """
        ✔ XGBoost achieved the best performance.

        ✔ Carat is the strongest driver of diamond prices.

        ✔ Larger diamonds generally command significantly higher prices.

        ✔ K-Means clustering identified two natural market segments.

        ✔ Premium Diamonds represent the luxury segment.

        ✔ Affordable Diamonds represent the value segment.

        ✔ Market segmentation can support inventory and marketing decisions.
        """
    )
