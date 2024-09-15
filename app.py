import streamlit as st
import pandas as pd
from pycaret.regression import RegressionExperiment
from pycaret.classification import ClassificationExperiment
import matplotlib.pyplot as plt

st.title('Aplikacja z PyCaret i Streamlit')

st.write('Załaduj swój plik CSV')

# Wczytanie pliku
uploaded_file = st.file_uploader('Wybierz plik CSV', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('Podgląd danych:')
    st.dataframe(data.head())

    # Wybór problemu: regresja czy klasyfikacja
    problem_type = st.selectbox('Wybierz typ problemu', ['Regresja', 'Klasyfikacja'])

    # Wybór zmiennej docelowej
    target = st.selectbox('Wybierz zmienną docelową (target)', data.columns)

    # Wybór predyktorów
    all_columns = list(data.columns)
    all_columns.remove(target)
    selected_features = st.multiselect('Wybierz kolumny do modelowania (predyktory)', all_columns, default=all_columns)

    if st.button('Uruchom PyCaret'):
        # Przygotowanie danych
        data_modeling = data[selected_features + [target]]

        if problem_type == 'Regresja':
            # Inicjalizacja eksperymentu regresji PyCaret
            exp = RegressionExperiment()
            with st.spinner('Przygotowywanie eksperymentu...'):
                exp.setup(data=data_modeling, target=target, session_id=123)

            # Porównanie modeli i wybór najlepszego
            with st.spinner('Trenowanie modeli...'):
                best_model = exp.compare_models()

            st.success('Najlepszy model wybrany!')
            st.write('Najlepszy model:', best_model)

            # Ewaluacja najlepszego modelu
            st.subheader('Ewaluacja modelu')

            # Wykres reszt
            with st.spinner('Generowanie wykresu reszt...'):
                exp.plot_model(best_model, plot='residuals', display_format='streamlit')

            # Ważność cech
            with st.spinner('Generowanie wykresu ważności cech...'):
                exp.plot_model(best_model, plot='feature', display_format='streamlit')

            # Wykres SHAP
            with st.spinner('Generowanie wykresu SHAP...'):
                exp.interpret_model(best_model, plot='summary', display_format='streamlit')

            # Predykcja na zbiorze testowym
            st.subheader('Predykcja na zbiorze testowym')
            holdout_pred = exp.predict_model(best_model)
            st.write('Predykcje na zbiorze testowym:')
            st.dataframe(holdout_pred.head())

            # Wykres rozrzutu Target vs Predicted
            st.subheader('Wykres rozrzutu: Rzeczywiste vs Przewidywane')
            actual = holdout_pred[target]
            predicted = holdout_pred['Label']

            plt.figure(figsize=(10,6))
            plt.scatter(actual, predicted, alpha=0.5)
            plt.xlabel('Rzeczywiste wartości')
            plt.ylabel('Przewidywane wartości')
            plt.title('Rzeczywiste vs Przewidywane')
            st.pyplot(plt)

            # Analiza dla pojedynczej obserwacji
            st.subheader('Analiza wpływu cech dla pojedynczej obserwacji')
            index = st.number_input('Wybierz indeks obserwacji do analizy', min_value=0, max_value=len(data_modeling)-1, value=0, step=1)

            with st.spinner('Generowanie wykresu SHAP dla wybranej obserwacji...'):
                exp.interpret_model(best_model, plot='force', observation=index, display_format='streamlit')

        elif problem_type == 'Klasyfikacja':
            # Inicjalizacja eksperymentu klasyfikacji PyCaret
            exp = ClassificationExperiment()
            with st.spinner('Przygotowywanie eksperymentu...'):
                exp.setup(data=data_modeling, target=target, session_id=123, silent=True, html=False)

            # Porównanie modeli i wybór najlepszego
            with st.spinner('Trenowanie modeli...'):
                best_model = exp.compare_models()

            st.success('Najlepszy model wybrany!')
            st.write('Najlepszy model:', best_model)

            # Ewaluacja najlepszego modelu
            st.subheader('Ewaluacja modelu')

            # Macierz konfuzji
            with st.spinner('Generowanie macierzy konfuzji...'):
                exp.plot_model(best_model, plot='confusion_matrix', display_format='streamlit')

            # Ważność cech
            with st.spinner('Generowanie wykresu ważności cech...'):
                exp.plot_model(best_model, plot='feature', display_format='streamlit')

            # Wykres SHAP
            with st.spinner('Generowanie wykresu SHAP...'):
                exp.interpret_model(best_model, plot='summary', display_format='streamlit')

            # Predykcja na zbiorze testowym
            st.subheader('Predykcja na zbiorze testowym')
            holdout_pred = exp.predict_model(best_model)
            st.write('Predykcje na zbiorze testowym:')
            st.dataframe(holdout_pred.head())

            # Wykres ROC/AUC
            with st.spinner('Generowanie wykresu ROC/AUC...'):
                exp.plot_model(best_model, plot='auc', display_format='streamlit')

            # Wykres Rzeczywiste vs Przewidywane
            st.subheader('Wykres: Rzeczywiste vs Przewidywane')
            actual = holdout_pred[target]
            predicted = holdout_pred['Label']

            # Tworzenie DataFrame z rzeczywistymi i przewidywanymi wartościami
            comparison_df = pd.DataFrame({'Rzeczywiste': actual, 'Przewidywane': predicted})

            # Wykres słupkowy liczności klas
            st.bar_chart(comparison_df.value_counts().unstack())

            # Analiza dla pojedynczej obserwacji
            st.subheader('Analiza wpływu cech dla pojedynczej obserwacji')
            index = st.number_input('Wybierz indeks obserwacji do analizy', min_value=0, max_value=len(data_modeling)-1, value=0, step=1)

            with st.spinner('Generowanie wykresu SHAP dla wybranej obserwacji...'):
                exp.interpret_model(best_model, plot='force', observation=index, display_format='streamlit')

        # Zapis modelu (opcjonalnie)
        # exp.save_model(best_model, 'best_pipeline')
