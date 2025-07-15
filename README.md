# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Python, Flask, and cosine similarity. This project uses metadata like genres, keywords, cast, and crew to recommend similar movies from a dataset.

## 🚀 Demo

![Demo](movie-demo.gif)

## 🔍 Features

- Recommend movies based on content similarity
- Display movie posters using TMDb API
- Fast and lightweight Flask backend
- Clean and responsive UI using HTML/CSS
- Integrated search bar to fetch results dynamically

## 🧠 How It Works

1. Preprocess and clean movie metadata (genres, keywords, etc.)
2. Create a "tags" column combining all relevant info
3. Use CountVectorizer to convert text to vectors
4. Compute similarity using cosine similarity
5. Display top 5 most similar movies with posters


## 🛠️ Tech Stack

- **Languages:** Python, HTML/CSS
- **Libraries:** Flask, pandas, scikit-learn, numpy, requests
- **APIs:** TMDb API (for posters)
