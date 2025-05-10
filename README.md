# Movie Recommender System

# ![download](https://github.com/user-attachments/assets/f56b1d20-3ad6-4d22-97ed-afa92b5e2109)

A content-based movie recommendation system built with Python and Streamlit. Enter a movie title, and the system suggests up to 20 similar movies based on features like genres, keywords, cast, director, and more.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Technologies Used](#technologies-used)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [How It Works](#how-it-works)
9. [Acknowledgments](#acknowledgments)
10. [License](#license)

## Project Overview
The **Movie Recommender System** uses content-based filtering to recommend movies similar to a user-specified title. It leverages **TF-IDF vectorization** and **cosine similarity** to analyze textual features, providing personalized recommendations through an interactive Streamlit web interface.

## Features
- Input a movie title to receive up to 20 similar movie recommendations.
- Fuzzy matching for movie titles to handle minor input errors.
- Clean, tabulated output of recommendations.
- Error handling for missing files, invalid inputs, or non-existent movies.
- Efficient data loading with caching (`@st.cache_data`).

## Dataset
- **File**: `movies.csv.gz` (compressed CSV).
- **Location**: `Data/` folder.
- **Columns Used**: `index`, `title`, `genres`, `keywords`, `tagline`, `cast`, `director`, `release_year`, `overview`.
- **Size**: Approximately 4,803 movies.
- **Description**: Contains metadata for movies, including genres, keywords, and cast details.

**Note**: Ensure the `movies.csv.gz` file is placed in the `Data/` folder before running the application.

## Technologies Used
- **Python 3.8+**
- **Libraries** (see `requirements.txt`):
  - `pandas`: Data manipulation.
  - `numpy`: Numerical computations.
  - `scikit-learn`: TF-IDF vectorization and cosine similarity.
  - `streamlit`: Web interface.
  - `scipy`: Sparse matrix operations.
- **Other**: `difflib` for fuzzy matching, `re` for text cleaning.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/movie-recommender-system.git
   cd movie-recommender-system
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**:
   - Place the `movies.csv.gz` file in the `Data/` folder. If you don’t have the dataset, contact the repository owner or use a compatible movie dataset with the required columns.

## Usage
1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Access the App**:
   - Open your browser and go to `http://localhost:8501`.
   - Enter a movie title (e.g., "Iron Man") in the text input field.
   - Click the **"Show Recommendations"** button to view a table of similar movies.

3. **Example**:
   - Input: "The matrix"
   - Output: A table listing up to 20 movies like "The Matrix Reloaded", "The Matrix Revolutions", etc.
![14-52-01](https://github.com/user-attachments/assets/c71264dc-84a8-4236-85d1-e3544b0d6507)

## Project Structure
```
movie-recommender-system/
│
├── Data/
│   └── movies.csv.gz        # Dataset (not included in repo, place here)
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## How It Works
1. **Data Loading**: Loads `movies.csv.gz` using Pandas, selecting relevant columns.
2. **Text Preprocessing**: Cleans text features (e.g., removes punctuation, converts to lowercase).
3. **Feature Combination**: Combines features like genres (weighted higher), keywords, cast, director, etc.
4. **Vectorization**: Converts combined text into TF-IDF vectors using `TfidfVectorizer`.
5. **Similarity Calculation**: Computes cosine similarity between all movies.
6. **Recommendation**: Matches the input title using `difflib`, retrieves the movie index, and ranks similar movies by similarity score.
7. **Interface**: Displays recommendations in a Streamlit table with error handling for edge cases.

## Acknowledgments
- **Instant Software Solutions**: For providing exceptional training.
- **Eng. Ahmed Hafez**: For mentorship and guidance.
- **Open-Source Community**: For tools like Streamlit and scikit-learn.

## License
This project is licensed under the [MIT License](LICENSE).
