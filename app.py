import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()

# Интерфейс
st.set_page_config(page_title="AI-анализатор настроения", page_icon="🧠")
st.title("🧠 AI-анализатор настроения текста")
st.write("Введите текст на английском, чтобы узнать его эмоциональный тон.")

# Ввод текста
user_input = st.text_area("Ваш текст:")

if user_input:
    scores = analyzer.polarity_scores(user_input)
    compound = scores['compound']

    # Классификация
    if compound >= 0.5:
        mood = "😊 Позитивный"
    elif compound <= -0.5:
        mood = "😠 Негативный"
    else:
        mood = "😐 Нейтральный"

    # Вывод
    st.markdown(f"### Результат анализа: {mood}")
    st.write(f"**Оценки:** {scores}")
