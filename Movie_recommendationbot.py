import os
import requests
import random
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging

class DiaryMovieRecommender:
    def __init__(self, omdb_api_key: str):
        """
        Initialize the movie recommender with necessary models and API key
        
        Args:
            omdb_api_key (str): Your OMDB API key
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.omdb_api_key = omdb_api_key
        self.omdb_base_url = "http://www.omdbapi.com/"
        
        # Initialize sentiment analyzer
        try:
            # Use SentencePiece tokenizer (disable fast tokenizer)
            model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
            tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment", use_fast=False)

            # Create pipeline with correct model and tokenizer
            self.sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        except Exception as e:
            self.logger.error(f"Error initializing sentiment analyzer: {e}")
            # Fallback to another model in case of failure
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.logger.info("Using fallback sentiment analysis model.")

        # Initialize zero-shot classifier for themes/genres
        self.theme_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def recommend_movies(self, diary_entry: str):
        """
        Recommend movies based on the user's diary entry.
        
        Args:
            diary_entry (str): The user's diary entry.

        Returns:
            List[Dict[str, str]]: A list of recommended movies with explanations.
        """
        # Analyze sentiment of the diary entry
        sentiment = self.sentiment_analyzer(diary_entry)[0]  # Get first result
        self.logger.info(f"Sentiment analysis result: {sentiment}")
        
        # Map sentiment to a broader set of genres
        if sentiment['label'] == "POSITIVE":
            genre_keywords = ["comedy", "romantic", "adventure", "family", "animation"]
        elif sentiment['label'] == "NEGATIVE":
            genre_keywords = ["thriller", "drama", "horror", "mystery"]
        else:
            genre_keywords = ["drama", "action", "crime", "biography"]
        
        # Use random genre from the relevant genres based on sentiment
        selected_genre = random.choice(genre_keywords)
        self.logger.info(f"Selected genre based on sentiment: {selected_genre}")

        # Fetch random movies from OMDB using the selected genre
        response = requests.get(
            self.omdb_base_url,
            params={"s": selected_genre, "apikey": self.omdb_api_key}
        )
        
        if response.status_code != 200:
            self.logger.error("Failed to retrieve data from OMDB API.")
            return []

        data = response.json()
        if data['Response'] == 'False':
            self.logger.warning(f"No movies found for genre: {selected_genre}")
            return []

        # Shuffle the results to get different movies each time
        random.shuffle(data['Search'])

        # Get the first few movie results and explain recommendations
        recommendations = []
        for movie in data['Search'][:5]:  # Limit to 5 recommendations
            recommendations.append({
                "title": movie["Title"],
                "explanation": f"This movie reflects a '{selected_genre}' theme that matches your diary sentiment."
            })
        
        return recommendations

# Main function to test the recommender
def main():
    # Initialize the recommender
    omdb_api_key = os.getenv("OMDB_API_KEY")  # Ensure to set this environment variable
    if not omdb_api_key:
        print("Error: OMDB_API_KEY environment variable is not set.")
        return
    
    recommender = DiaryMovieRecommender(omdb_api_key)
    
    # Prompt user for diary entry
    print("Please enter your diary entry (press Enter twice to finish):")
    diary_entry_lines = []
    while True:
        line = input()
        if line:
            diary_entry_lines.append(line)
        else:
            break
    diary_entry = "\n".join(diary_entry_lines)
    
    # Get recommendations
    recommendations = recommender.recommend_movies(diary_entry)
    
    # Print recommendations
    print("\nBased on your diary entry, here are some movie recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Why: {rec['explanation']}\n")

if __name__ == "__main__":
    main()
