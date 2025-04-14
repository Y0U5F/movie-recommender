import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import os

# دالة تنظيف النصوص
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^\w\s]', '', text)  # إزالة علامات الترقيم
    return text.lower()

# تحميل البيانات بكفاءة
@st.cache_data
def load_data():
    data_path = os.path.join('Data', 'movies.csv.gz')  # المسار للملف المضغوط
    try:
        # اقرا أسماء الأعمدة أولاً
        df = pd.read_csv(data_path, compression='gzip', nrows=0)
        available_columns = df.columns.tolist()

        # الأعمدة الأساسية المطلوبة
        required_columns = ['index', 'title', 'genres', 'keywords', 'tagline', 'cast', 'director']
        # الأعمدة الاختيارية
        optional_columns = ['release_year', 'overview']

        # تحقق من وجود الأعمدة الأساسية
        missing_required = [col for col in required_columns if col not in available_columns]
        if missing_required:
            st.error(f"الأعمدة الأساسية التالية غير موجودة في الملف: {missing_required}")
            return None

        # الأعمدة اللي هنحملها فعلًا (الأساسية + الاختيارية الموجودة)
        columns_to_load = required_columns + [col for col in optional_columns if col in available_columns]

        # تحميل البيانات
        movies_data = pd.read_csv(data_path, compression='gzip', usecols=columns_to_load)
        return movies_data
    except FileNotFoundError:
        st.error(f"ملف movies.csv.gz غير موجود في {data_path}. تأكد إنه موجود في المجلد 'Data'.")
        return None
    except Exception as e:
        st.error(f"حصل خطأ أثناء تحميل البيانات: {str(e)}")
        return None

# معالجة البيانات
@st.cache_data
def process_data(movies_data):
    # تحديد المميزات الأساسية
    features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    
    # إضافة release_year و overview لو موجودين
    if 'release_year' in movies_data.columns:
        features.append('release_year')
        movies_data['release_year'] = movies_data['release_year'].astype(str).fillna('')
    if 'overview' in movies_data.columns:
        features.append('overview')
        movies_data['overview'] = movies_data['overview'].fillna('')
    
    # تنظيف المميزات
    for feature in features:
        movies_data[feature] = movies_data[feature].apply(clean_text)
    
    # إعطاء أهمية أكبر للـ genres بتكراره
    combined_features = (movies_data['genres'] + ' ' + movies_data['genres'] + ' ' +
                        movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' +
                        movies_data['cast'] + ' ' + movies_data['director'])
    
    # إضافة المميزات الإضافية
    if 'release_year' in movies_data.columns:
        combined_features += ' ' + movies_data['release_year']
    if 'overview' in movies_data.columns:
        combined_features += ' ' + movies_data['overview']
    
    # تحويل النصوص لفيكتورز
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    
    return similarity, movies_data

# دالة التوصية
def recommend_movies(movie_name, similarity, movies_data, top_n=20):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1)
    
    if not find_close_match:
        return None, "لم يتم العثور على فيلم مشابه للاسم المدخل."
    
    close_match = find_close_match[0]
    try:
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    except IndexError:
        return None, "خطأ في العثور على الفيلم في البيانات."
    
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i, movie in enumerate(sorted_similar_movies[:top_n], 1):
        index = movie[0]
        try:
            title = movies_data[movies_data.index == index]['title'].values[0]
            recommendations.append((i, title))
        except IndexError:
            continue
    
    return recommendations, None

# واجهة Streamlit
st.title("نظام توصية الأفلام 🎥")
st.write("ادخل اسم فيلم وهنقترحلك أفلام مشابهة!")

# تحميل البيانات
movies_data = load_data()

if movies_data is not None:
    # معالجة البيانات
    similarity, movies_data = process_data(movies_data)
    
    # إدخال اسم الفيلم
    movie_name = st.text_input("اكتب اسم الفيلم:", placeholder="مثال: Iron Man")
    
    if st.button("اعرض التوصيات"):
        if movie_name.strip() == "":
            st.warning("من فضلك اكتب اسم فيلم.")
        else:
            recommendations, error = recommend_movies(movie_name, similarity, movies_data)
            
            if error:
                st.error(error)
            else:
                st.success(f"التوصيات لفيلم: {movie_name}")
                # عرض التوصيات في جدول
                st.table(pd.DataFrame(recommendations, columns=["الترتيب", "اسم الفيلم"]))

else:
    st.stop()
