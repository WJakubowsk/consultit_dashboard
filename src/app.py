import pandas as pd
import streamlit as st
from bertopic import BERTopic
from utils import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.manifold import TSNE

# from transformers import pipeline

# load dataset
df = pd.read_csv('sample_data.csv')


# Sales forecasting
def sales_forecasting_section():
    st.header("Prognozowanie sprzedaży")
    
    dfts= df.sort_values(by=['date'])

    #convert datetime from object to datetime type
    dfts['date'] = pd.to_datetime(dfts['date'])
    #set datetime as index
    dfts = dfts.set_index(dfts['date'])

    #drop datetime column
    dfts.drop('date', axis=1, inplace=True)

    #create hour, day and month variables from datetime index
    dfts['hour'] = dfts.index.hour
    dfts['day'] = dfts.index.day
    dfts['month'] = dfts.index.month

    #drop string-based columns
    dfts.drop(['product', 'review', 'age', 'sex'], axis=1, inplace=True)

    horizon=24*7
    X = dfts.drop('price', axis=1)
    y = dfts['price']
        
    #take last week of the dataset for validation
    X_train, X_test = X.iloc[:-horizon,:], X.iloc[-horizon:,:]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
        
    #create, train and do inference of the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dfts.index[0:111], y=y_test[0:111], mode='lines', name='Rzeczywisty trend', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=dfts.index[110:], y=predictions[110:], mode='lines', name='Przewidywany trend', line=dict(color='green', dash='dash')))

    fig.update_layout(title=f'Trend rzeczywisty vs przewidywany',
                    xaxis_title='Data i czas',
                    yaxis_title='Zarobiona kwota [PLN]',
                    legend=dict(font=dict(size=16)),
                    font=dict(size=16),
                    height=600,
                    width=1000,
                    template='plotly_white')
    
    st.plotly_chart(fig)

# Client segmentation
def customer_segmentation_section():
    st.header("Segmentacja klientów")

    dfc = df.drop('date', axis=1)
    
    ### operations for categorical columns with order or binary values
    ord_pipeline = Pipeline(steps=[
        ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    ### operations for categorical unordered columns
    cat_pipeline = Pipeline(steps=[
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    ### operations for numerical columns
    num_pipeline = Pipeline(steps=[
        ('discretize', KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform'))
    ])

    # Column transformer
    col_trans = ColumnTransformer(transformers=[
        ('ord_pipeline', ord_pipeline, ['score', 'sex']),
        ('cat_pipeline', cat_pipeline, ['product', 'review']),
        ('num_pipeline', num_pipeline, ['price', 'age'])
    ],
    remainder='drop',
    n_jobs=-1)

    # hierarchical clustering at the end of the pipeline (limit as desired number of people in the group)
    model_pipeline = Pipeline([
        ('preprocessing', col_trans),
    #    ('clustering', KMeans(linkage='ward'))
    ])

    # preprocess data
    data_preprocessed = model_pipeline.fit_transform(dfc)

    # convert compressed data to numpy array
    decompressed_data = data_preprocessed.toarray()

    # initialise model
    model = KMeans(n_clusters=2, random_state=42)

    # fit model
    clustered_data = model.fit_predict(decompressed_data)

    # Initialize TSNE model with desired parameters
    tsne = TSNE(n_components=2, random_state=42)

    # Perform t-SNE on the data
    tsne_result = tsne.fit_transform(decompressed_data)

    # Create a Plotly scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], mode='markers', marker=dict(color=clustered_data, colorscale='rainbow')))

    fig.update_layout(
                    xaxis_title='t-SNE 1',
                    yaxis_title='t-SNE 2',
                    width=800,
                    height=600)

    st.plotly_chart(fig)

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
