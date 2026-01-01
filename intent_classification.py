
# ===================================
# intent_data_generator.py
# ===================================
import pandas as pd
import random

def generate_intent_dataset(output_file="intent_training_data.csv"):
    """
    Generate a comprehensive intent classification dataset for movie chatbot
    Creates 1000+ samples across 6 intent categories
    """
    
    # Intent categories and their examples
    intents = {
        "GREETING": [
            "Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening",
            "Hi there", "Hello there", "Hey there", "Greetings", "What's up",
            "Howdy", "Hiya", "Yo", "Sup", "Morning", "Evening",
            "Hi how are you", "Hello how are you doing", "Hey what's up",
            "Good day", "Nice to meet you", "Pleased to meet you",
            "Hi I'm looking for help", "Hello can you help me", "Hey there friend",
            "Greetings friend", "Salutations", "Hi buddy", "Hello pal",
            "Hey mate", "What's going on", "How's it going",
            "Hi I need help with movies", "Hello I want movie suggestions",
            "Hey can you recommend something", "Hi I'm new here",
            "Hello I just started using this", "Hey first time here",
            "Good to see you", "Nice to chat with you", "Happy to be here",
            "Hi again", "Hello once more", "Hey I'm back",
            "Morning sunshine", "Evening friend", "Hello world",
            "Hi chatbot", "Hello assistant", "Hey AI",
            "Greetings assistant", "Hello movie bot", "Hi movie helper"
        ],
        
        "GOODBYE": [
            "Goodbye", "Bye", "See you", "See you later", "Take care",
            "Have a good day", "Have a nice day", "Catch you later",
            "Talk to you later", "Later", "Peace", "Cheers",
            "Have a good one", "Until next time", "Farewell",
            "See ya", "Bye bye", "Cya", "Ttyl", "Gotta go",
            "I have to go", "I need to leave", "Thanks and goodbye",
            "That's all for now", "I'm done", "That's it",
            "Thank you goodbye", "Thanks bye", "Appreciate it bye",
            "Okay I'm leaving", "Alright I'm out", "I'll be going now",
            "Time to go", "Heading out", "Signing off",
            "That's enough for today", "I'll come back later", "See you soon",
            "Until we meet again", "Goodbye for now", "Bye for now",
            "Take it easy", "Stay safe", "Be well",
            "Thanks for your help goodbye", "That was helpful bye",
            "Perfect thank you bye", "Got what I needed thanks bye",
            "Okay thanks see you", "Alright thank you goodbye"
        ],
        
        "MOVIE_QUERY": [
            "Recommend me a movie", "Suggest a good film", "What should I watch",
            "I want to watch something", "Looking for a movie", "Need movie suggestions",
            "What's a good action movie", "Best comedy films", "Top rated movies",
            "Movies with high IMDB rating", "Recent blockbusters", "Classic films",
            "What are the best movies of all time", "Show me drama movies",
            "I like science fiction", "Recommend thriller movies",
            "What's popular right now", "Trending movies", "Award winning films",
            "Movies from the 90s", "2000s movies", "Latest releases",
            "Tell me about Inception", "What's The Godfather about",
            "Who directed Pulp Fiction", "Cast of The Dark Knight",
            "Movies similar to Interstellar", "Films like Shawshank Redemption",
            "What's the rating of Titanic", "IMDB score for Avatar",
            "Best movies of 2020", "Top 10 movies", "Must watch films",
            "Underrated movies", "Hidden gems", "Cult classics",
            "Movies for family", "Kid friendly films", "Animated movies",
            "Horror movie recommendations", "Scary films", "Romance movies",
            "Love story films", "Action packed movies", "Adventure films",
            "Documentary recommendations", "True story movies", "Based on real events",
            "Movies with Leonardo DiCaprio", "Tom Hanks films", "Meryl Streep movies",
            "Christopher Nolan movies", "Quentin Tarantino films", "Steven Spielberg movies",
            "What's a good movie for tonight", "Weekend movie suggestions",
            "Movies to watch with friends", "Date night movies", "Solo watching recommendations",
            "Feel good movies", "Sad movies", "Emotional films",
            "Mind bending movies", "Psychological thrillers", "Plot twist movies",
            "Long movies", "Short films", "Epic movies",
            "What movie won Oscar", "Academy Award winners", "Golden Globe films",
            "Foreign films", "International cinema", "Bollywood movies",
            "Korean films", "Japanese movies", "French cinema",
            "Black and white movies", "Silent films", "Old classics",
            "Modern cinema", "Contemporary films", "Indie movies",
            "Low budget gems", "Box office hits", "Highest grossing movies",
            "Critics choice movies", "Audience favorites", "Controversial films",
            "Banned movies", "Censored films", "Director's cut movies",
            "Extended editions", "Remastered films", "4K movies",
            "IMAX movies", "3D films", "Surround sound movies",
            "Movies with great soundtracks", "Films with amazing music",
            "Movies with plot twists", "Unpredictable films", "Surprise ending movies",
            "Trilogies", "Movie series", "Franchises", "Sequels",
            "Prequels", "Spin offs", "Remakes", "Reboots",
            "Book adaptations", "Comic book movies", "Video game movies",
            "Musical films", "Dance movies", "Sports films",
            "War movies", "Historical films", "Period dramas",
            "Western movies", "Sci-fi epics", "Fantasy adventures",
            "Superhero movies", "Marvel films", "DC movies",
            "Disney movies", "Pixar films", "Studio Ghibli",
            "What's the best movie ever", "Greatest film of all time",
            "Most influential movies", "Game changing films", "Revolutionary cinema"
        ],
        
        "REJECT": [
            "No", "Nope", "No thanks", "Not interested", "I don't want that",
            "That's not what I want", "Cancel", "Nevermind", "Forget it",
            "I changed my mind", "Actually no", "On second thought no",
            "I don't think so", "Not really", "Not for me",
            "I'll pass", "Skip that", "Not my thing", "Not interested in that",
            "I don't like it", "That's not good", "That sounds bad",
            "No way", "Definitely not", "Absolutely not", "Hell no",
            "Not a chance", "I refuse", "I decline", "I reject that",
            "Stop", "Quit", "Exit", "End this", "I'm done with this",
            "This isn't working", "This isn't helpful", "Not useful",
            "Wrong answer", "That's incorrect", "Try again", "Give me something else",
            "I don't want to continue", "Stop suggesting", "No more",
            "That's enough", "I've had enough", "Stop it", "Cease",
            "Don't", "Please don't", "I'd rather not", "I prefer not to",
            "Not now", "Maybe later", "Not at this time", "Some other time",
            "I'm not ready", "Not yet", "Hold on", "Wait",
            "I disagree", "That's wrong", "Incorrect", "False", "Untrue"
        ],
        
        "CHITCHAT": [
            "How are you", "How are you doing", "What's up", "How's it going",
            "How have you been", "What are you doing", "What are you up to",
            "Tell me about yourself", "Who are you", "What can you do",
            "What are your capabilities", "How do you work", "Are you AI",
            "Are you a robot", "Are you human", "What's your name",
            "Do you have a name", "Where are you from", "Where do you live",
            "How old are you", "When were you created", "Who made you",
            "What's the weather like", "How's the weather", "Is it raining",
            "What day is it", "What time is it", "What's the date",
            "Tell me a joke", "Make me laugh", "Say something funny",
            "What's your favorite movie", "Do you watch movies", "Do you like films",
            "What do you think", "What's your opinion", "How do you feel",
            "Are you happy", "Are you sad", "Do you have emotions",
            "Can you think", "Are you smart", "Are you intelligent",
            "Do you learn", "Do you remember me", "Can you remember things",
            "What did I say earlier", "Do you recall our conversation",
            "Are you busy", "What keeps you occupied", "Do you sleep",
            "Do you eat", "Do you drink", "Do you get tired",
            "What's your hobby", "What do you like", "What do you enjoy",
            "Tell me something interesting", "Share a fact", "Random fact",
            "Surprise me", "Entertain me", "Impress me",
            "You're cool", "You're awesome", "You're great", "You're helpful",
            "Thank you", "Thanks", "Appreciate it", "You helped me",
            "That's nice", "That's good", "That's helpful", "Perfect",
            "Exactly", "Right", "Correct", "Yes", "Yeah", "Yep",
            "I understand", "I see", "Got it", "Makes sense", "Okay",
            "Alright", "Sure", "Fine", "Whatever", "Meh",
            "Interesting", "Cool", "Nice", "Good", "Great"
        ],
        
        "OTHER": [
            "What's 2 plus 2", "Solve this math problem", "Calculate this",
            "Book a flight", "Find me a hotel", "Reserve a table",
            "Order pizza", "Buy groceries", "Shopping cart", "Add to cart",
            "What's the capital of France", "Geography question", "History fact",
            "Translate this", "Speak Spanish", "Learn French",
            "Code in Python", "Write a program", "Debug this code",
            "Fix my computer", "Technical support", "IT help",
            "Medical advice", "Health question", "Am I sick",
            "Legal advice", "Law question", "Is this illegal",
            "Stock prices", "Cryptocurrency", "Bitcoin value",
            "Sports scores", "Who won the game", "Team standings",
            "News today", "Current events", "What happened",
            "Political opinion", "Election results", "Government policy",
            "Recipe for pasta", "Cooking instructions", "Baking tips",
            "Workout routine", "Exercise plan", "Fitness advice",
            "Travel destinations", "Where should I travel", "Vacation spots",
            "Fashion advice", "What to wear", "Style tips",
            "Dating advice", "Relationship help", "How to ask someone out",
            "Career advice", "Job search", "Resume writing",
            "Financial planning", "Investment advice", "Save money tips",
            "Learn guitar", "Music lessons", "How to play piano",
            "Pet care", "Dog training", "Cat behavior",
            "Gardening tips", "Plant care", "How to grow tomatoes",
            "Car repair", "Auto maintenance", "Fix my car",
            "Home improvement", "DIY projects", "Renovation ideas",
            "Psychology facts", "Philosophy question", "Meaning of life",
            "Science experiment", "Physics question", "Chemistry help",
            "Write an essay", "Homework help", "Study tips",
            "Tell me about dinosaurs", "Space facts", "Ocean creatures",
            "Random gibberish", "Asdfghjkl", "Qwertyuiop",
            "Test test test", "Hello world", "Lorem ipsum",
            "Blank", "Nothing", "I don't know what to say"
        ]
    }
    
    # Generate dataset
    data = []
    
    # Add base examples
    for intent, examples in intents.items():
        for example in examples:
            data.append({"intent": intent, "text": example})
    
    # Add variations with slight modifications
    variations = []
    for intent, examples in intents.items():
        for example in examples[:20]:  # Take first 20 of each
            # Add variations
            variations.append({"intent": intent, "text": example.lower()})
            variations.append({"intent": intent, "text": example.upper()})
            variations.append({"intent": intent, "text": f"{example}?"})
            variations.append({"intent": intent, "text": f"{example}!"})
            variations.append({"intent": intent, "text": f"Can you {example.lower()}"})
            variations.append({"intent": intent, "text": f"I want to {example.lower()}"})
    
    data.extend(variations)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ Dataset created: {output_file}")
    print(f"📊 Total samples: {len(df)}")
    print(f"\n📈 Distribution:")
    print(df['intent'].value_counts())
    
    return df

if __name__ == "__main__":
    generate_intent_dataset()