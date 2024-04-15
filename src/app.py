import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob

# from transformers import pipeline

# Dane symulacyjne
opinions = [
    "Bardzo mi się podobały książki, świetna jakość!",
    "Obsługa klienta była bardzo pomocna i profesjonalna.",
    "Niestety, zeszyty były źle zapakowane i zostały uszkodzone w transporcie.",
    "Bardzo szybka dostawa, ale jakość nie jest zadowalająca.",
    "Kalendarze są pięknie zaprojektowane, ale cena trochę za wysoka.",
]

# Dane symulacyjne
products = [
    'Książka "Wiedźmin"',
    "Zeszyt A4 linia",
    "Kalendarz 2024",
    "Długopis żelowy",
    "Notes A5 kropka",
]
ratings = np.random.randint(1, 6, size=len(products))
prices = np.random.randint(10, 100, size=len(products))

df_products = pd.DataFrame(
    {"Product": products, "Rating": ratings, "Price (PLN)": prices, "Review": opinions}
)


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


# Sentiment analysis & topic modeling
def sentiment_analysis_section():
    st.header("Analiza sentymentu i modelowanie tematów")
    # Analiza sentymentu dla każdej opinii
    sentiment_scores = []
    for opinion in df_products["Review"]:
        sentiment = TextBlob(opinion).sentiment.polarity
        sentiment_scores.append(sentiment)

    # Przygotowanie danych do wykresu
    df_sentiment = pd.DataFrame(
        {"Opinion": opinions, "Sentiment Score": sentiment_scores}
    )

    # Wykres słupkowy sentymentu opinii
    fig_sentiment = px.bar(
        df_sentiment,
        x="Opinion",
        y="Sentiment Score",
        title="Analiza sentymentu opinii klientów",
        labels={"Opinion": "Opinia", "Sentiment Score": "Wartość sentymentu"},
    )
    st.plotly_chart(fig_sentiment)

    # Analiza sentymentu ogólnego
    overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    st.write(f"Średni sentyment opinii klientów: {overall_sentiment:.2f}")

    # Interpretacja sentymentu
    if overall_sentiment > 0:
        st.write("Ogólny sentyment opinii klientów jest pozytywny.")
    elif overall_sentiment < 0:
        st.write("Ogólny sentyment opinii klientów jest negatywny.")
    else:
        st.write("Ogólny sentyment opinii klientów jest neutralny.")


# Product recommendation
def product_recommendation_section():
    st.header("Personalizacje ofert i rekomendacje produktów")
    # sugeruj produkty w zestawach na podsawie podobnych zakupów użytkownik
    # Wykres słupkowy ocen produktów
    fig_ratings = px.bar(
        df_products,
        x="Product",
        y="Rating",
        title="Oceny produktów",
        labels={"Product": "Produkt", "Rating": "Ocena"},
    )
    st.plotly_chart(fig_ratings)

    # Wykres słupkowy cen produktów
    fig_prices = px.bar(
        df_products,
        x="Product",
        y="Price (PLN)",
        title="Ceny produktów",
        labels={"Product": "Produkt", "Price (PLN)": "Cena (PLN)"},
    )
    st.plotly_chart(fig_prices)

    # Analiza rekomendacji
    avg_rating = df_products["Rating"].mean()
    avg_price = df_products["Price (PLN)"].mean()
    st.write(f"Średnia ocena produktów: {avg_rating:.2f}")
    st.write(f"Średnia cena produktów: {avg_price:.2f} PLN")

    # Zalecenia na podstawie średnich wartości ocen i cen
    if avg_rating > 3.5 and avg_price < 50:
        st.write(
            "Rekomendacja: Polecamy produkty z wysokimi ocenami i przystępnymi cenami."
        )
    elif avg_rating < 3.5 and avg_price < 50:
        st.write(
            "Rekomendacja: Produkty mają niższe oceny, ale są dostępne w przystępnych cenach."
        )
    elif avg_rating > 3.5 and avg_price > 50:
        st.write("Rekomendacja: Produkty są dobrze oceniane, ale mają wyższe ceny.")
    else:
        st.write(
            "Rekomendacja: Produkty mają zarówno niskie oceny, jak i wysokie ceny."
        )


def description_generation_section():
    st.header("Generowanie opisu produktu")
    # Wczytanie danych od użytkownika
    product_input = st.text_input("Wpisz nazwę i opis produktu:")
    max_length = st.slider(
        "Maksymalna długość opisu:", min_value=50, max_value=200, value=100
    )

    # qa_pipeline = pipeline(
    #     "question-answering",
    #     model="henryk/bert-base-multilingual-cased-finetuned-polish-squad2",
    #     tokenizer="henryk/bert-base-multilingual-cased-finetuned-polish-squad2",
    # )

    # qa_pipeline(
    #     {
    #         "context": product_input,
    #         "question": "Wygeneruj opis produktu",
    #         max_length: max_length,
    #     }
    # )


def main():
    st.title("Dashboard Analizy Danych BookCrafters Sp. z.o.o.")

    # Tworzenie zakładek dla każdej sekcji
    tabs = [
        "Prognozowanie sprzedaży",
        "Segmentacja klientów",
        "Analiza sentymentu",
        "Rekomendacje produktów",
        "Generowanie opisu produktu",
    ]
    selected_tab = st.sidebar.radio("Wybierz zakładkę", tabs)

    if selected_tab == "Prognozowanie sprzedaży":
        sales_forecasting_section()
    elif selected_tab == "Segmentacja klientów":
        customer_segmentation_section()
    elif selected_tab == "Analiza sentymentu":
        sentiment_analysis_section()
    elif selected_tab == "Rekomendacje produktów":
        product_recommendation_section()
    elif selected_tab == "Generowanie opisu produktu":
        description_generation_section()


if __name__ == "__main__":
    main()
