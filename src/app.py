import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from bertopic import BERTopic
from utils import *

# from transformers import pipeline

# load dataset
df = pd.read_csv('sample_data.csv')


# Sales forecasting
def sales_forecasting_section():
    st.header("Prognozowanie sprzedaży")
    # Symulacja danych sprzedażowych
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    sales_data = pd.DataFrame(
        {"Date": dates, "Sales": np.random.randint(1000, 5000, size=len(dates))}
    )

    # Wykres liniowy prognozowanej sprzedaży
    fig = px.line(
        sales_data,
        x="Date",
        y="Sales",
        title="Prognozowana sprzedaż",
        labels={"Date": "Data", "Sales": "Sprzedaż"},
    )

    fig.update_layout(xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)


# Client segmentation
def customer_segmentation_section():
    st.header("Segmentacja klientów")

    # Symulacja danych klientów
    np.random.seed(42)
    customers = pd.DataFrame(
        {
            "CustomerID": range(1, 101),
            "Age": np.random.randint(18, 65, size=100),
            "AnnualIncome": np.random.randint(20000, 100000, size=100),
            "PurchaseFrequency": np.random.randint(1, 20, size=100),
        }
    )

    # Wykres kołowy udziału klientów w różnych grupach wiekowych
    age_counts = customers["Age"].value_counts().sort_index()
    fig_age = px.pie(
        names=age_counts.index,
        values=age_counts.values,
        title="Segmentacja klientów - Grupy wiekowe",
    )
    fig_age.update_traces(textposition="inside", textinfo="percent+label")

    st.plotly_chart(fig_age)

    # Histogram dochodu rocznego klientów
    fig_income = px.histogram(
        customers,
        x="AnnualIncome",
        nbins=20,
        title="Histogram dochodu rocznego klientów",
    )
    st.plotly_chart(fig_income)

    # Histogram częstotliwości zakupów klientów
    fig_purchase = px.histogram(
        customers,
        x="PurchaseFrequency",
        nbins=20,
        title="Histogram częstotliwości zakupów klientów",
    )
    st.plotly_chart(fig_purchase)


# Topic modeling
def topic_modeling_section():
    st.header("Analiza opinii klientów")

    st.plotly_chart(create_gauge_chart(round(df['score'].mean(), 2), min_value=1, max_value = 5, 
                                        label = "średnia ocena zamówienia"))

    ## topic modeling
    topic_modeling_model = BERTopic(language="english", nr_topics = 5)
    topics, topic_assignments = topic_modeling_model.fit_transform(df['review'])

    # Display top words for each topic
    st.subheader("Wizualizacja grup tematycznych opinii klientów")
    st.plotly_chart(topic_modeling_model.visualize_topics(title=""))



def description_generation_section():
    st.header("Generowanie opisu produktu")
    # Wczytanie danych od użytkownika
    product_input = st.text_input("Wpisz nazwę i krótki opis produktu:")
    max_length = st.slider(
        "Maksymalna długość opisu:", min_value=50, max_value=200, value=100
    )

    if st.button("Generuj opis"):
        generated_description = generate_description(product_input, max_length)
        st.markdown(f"**Wygenerowany opis:** {generated_description}")




def main():
    st.title("Dashboard analizy danych BookCrafters Sp. z.o.o.")

    # Tworzenie zakładek dla każdej sekcji
    tabs = [
        "Prognozowanie sprzedaży",
        "Segmentacja klientów",
        "Analiza opinii klientów",
        "Generowanie opisu produktu",
    ]
    selected_tab = st.sidebar.radio("Wybierz zakładkę", tabs)

    if selected_tab == "Prognozowanie sprzedaży":
        sales_forecasting_section()
    elif selected_tab == "Segmentacja klientów":
        customer_segmentation_section()
    elif selected_tab == "Analiza opinii klientów":
        topic_modeling_section()
    elif selected_tab == "Generowanie opisu produktu":
        description_generation_section()


if __name__ == "__main__":
    main()
