import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    clean_tokens = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            clean_tokens.append(ps.stem(word))

    return " ".join(clean_tokens)


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


st.sidebar.title("📊 Project Info")
st.sidebar.markdown("""
**Spam Detection App**  
Built using:
- Machine Learning  
- TF-IDF Vectorizer  
- Naive Bayes  
- Streamlit  

👨‍💻 **Developer:** Alok Sahu  
""")

st.sidebar.info("This app classifies messages as **Spam** or **Not Spam**")


st.markdown(
    "<h1 style='text-align: center;'>📧 Email / SMS Spam Classifier</h1>",
    unsafe_allow_html=True
)

st.write("Paste a message below and click **Predict**")

input_sms = st.text_area(
    "✍️ Enter your message",
    height=150,
    placeholder="Type your email or SMS here..."
)


if st.button("🔍 Predict", use_container_width=True):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0]

        st.markdown("---")

        if result == 1:
            st.error("🚨 **SPAM MESSAGE DETECTED**")
            st.metric("Spam Probability", f"{prob[1]*100:.2f}%")
        else:
            st.success("✅ **THIS MESSAGE IS NOT SPAM**")
            st.metric("Not Spam Probability", f"{prob[0]*100:.2f}%")


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>Made with ❤️ using Streamlit | ML Project</p>",
    unsafe_allow_html=True
)

# Win  +  . presss for Emoji
# venv\Scripts\activate
# streamlit run app.py💖💖💖💖👌👌👌
