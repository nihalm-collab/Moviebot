import pandas as pd
import random

# Intent Classification Dataset Generator
# This creates 1000+ samples for training/testing

def generate_intent_dataset():
    """Generate a comprehensive intent classification dataset"""
    
    dataset = []
    
    # 1. GREETING intents (200 samples)
    greetings = [
        "Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening",
        "Hi there", "Hello there", "Hey there", "Greetings", "Howdy",
        "What's up", "Yo", "Hiya", "Heya", "Good day",
        "Top of the morning", "How do you do", "Pleased to meet you",
        "Nice to meet you", "Hey buddy", "Hi friend", "Hello friend",
        "Morning", "Evening", "Afternoon", "Hey man", "Hi dude",
        "Hello bot", "Hi assistant", "Hey chatbot", "Sup", "Wassup",
        "Good to see you", "Hey you", "Hi bot", "Hello AI",
        "Greetings human", "Salutations", "Welcome", "Aloha", "Bonjour",
        "Hola", "Ciao", "Namaste", "Konnichiwa", "Ni hao"
    ]
    
    for _ in range(200):
        greeting = random.choice(greetings)
        variations = [
            greeting,
            f"{greeting}!",
            f"{greeting}!!",
            f"{greeting}, how are you?",
            f"{greeting}, can you help me?",
            f"{greeting} there",
            f"Oh {greeting.lower()}",
            f"{greeting}, I need help",
        ]
        dataset.append({
            "intent": "GREETING",
            "text": random.choice(variations)
        })
    
    # 2. GOODBYE intents (200 samples)
    goodbyes = [
        "Goodbye", "Bye", "See you", "Take care", "Farewell",
        "See you later", "Catch you later", "Talk to you later",
        "Until next time", "Have a good day", "Have a great day",
        "Thanks bye", "Ok bye", "Alright bye", "Gotta go",
        "I have to go", "I need to leave", "Bye bye", "Bye for now",
        "Later", "Peace out", "Cheers", "Cya", "See ya",
        "TTYL", "Adios", "Au revoir", "Ciao", "Sayonara",
        "Good night", "Sleep well", "Sweet dreams", "Until we meet again"
    ]
    
    for _ in range(200):
        goodbye = random.choice(goodbyes)
        variations = [
            goodbye,
            f"{goodbye}!",
            f"{goodbye}!!",
            f"Thanks, {goodbye.lower()}",
            f"Ok {goodbye.lower()}",
            f"Alright, {goodbye.lower()}",
            f"{goodbye}, thanks for your help",
            f"{goodbye}, that was helpful",
        ]
        dataset.append({
            "intent": "GOODBYE",
            "text": random.choice(variations)
        })
    
    # 3. MOVIE_QUERY intents (350 samples)
    movie_queries = [
        # Recommendations
        "Recommend me a movie",
        "Can you suggest a good film?",
        "What's a good movie to watch?",
        "I'm looking for movie recommendations",
        "Suggest me an action movie",
        "What are some good drama films?",
        "Can you recommend a comedy?",
        "I want to watch a thriller",
        "Give me some horror movie suggestions",
        "What are the best sci-fi movies?",
        "Recommend a romantic movie",
        "What's a good family movie?",
        "Suggest some animated films",
        "I need a movie for tonight",
        "What should I watch this weekend?",
        
        # Specific queries
        "What's the rating of Inception?",
        "Who directed The Godfather?",
        "Tell me about The Dark Knight",
        "What year was Titanic released?",
        "Who stars in Pulp Fiction?",
        "What's the plot of Interstellar?",
        "Is The Shawshank Redemption good?",
        "What genre is Fight Club?",
        "How long is The Lord of the Rings?",
        "What's the IMDB rating of Forrest Gump?",
        
        # Genre-based
        "Show me action movies",
        "List comedy films",
        "What are the top rated dramas?",
        "Give me thriller recommendations",
        "What are some good horror movies?",
        "Recommend sci-fi films",
        "Show me romantic comedies",
        "What are the best war movies?",
        "List some crime movies",
        "What are good mystery films?",
        
        # Year/Era based
        "What are the best movies from the 90s?",
        "Recommend recent films",
        "Show me classic movies",
        "What are good 2000s movies?",
        "Suggest some old movies",
        "What are the latest blockbusters?",
        
        # Actor/Director based
        "What movies has Tom Hanks been in?",
        "Show me Leonardo DiCaprio films",
        "What did Christopher Nolan direct?",
        "Recommend movies with Morgan Freeman",
        "What are Brad Pitt's best movies?",
        
        # Rating based
        "What are the highest rated movies?",
        "Show me movies with rating above 8",
        "What are the top 10 movies?",
        "Which movies have perfect scores?",
        "What are critically acclaimed films?",
    ]
    
    for _ in range(350):
        query = random.choice(movie_queries)
        variations = [
            query,
            f"{query}?",
            f"Can you {query.lower()}?",
            f"Please {query.lower()}",
            f"I want to know {query.lower()}",
            f"Tell me {query.lower()}",
        ]
        dataset.append({
            "intent": "MOVIE_QUERY",
            "text": random.choice(variations)
        })
    
    # 4. REJECT intents (100 samples)
    rejects = [
        "No", "Nope", "No thanks", "No thank you", "I don't want that",
        "Not interested", "Never mind", "Nevermind", "Forget it",
        "Cancel", "Cancel that", "I changed my mind", "I don't like it",
        "That's not what I want", "That's wrong", "Not good",
        "I don't need that", "No way", "Absolutely not", "Not really",
        "I'm not interested", "Don't want it", "Stop", "Quit",
        "I disagree", "That's incorrect", "Wrong", "Negative",
        "I'll pass", "Skip", "Skip that", "Next", "Move on"
    ]
    
    for _ in range(100):
        reject = random.choice(rejects)
        variations = [
            reject,
            f"{reject}!",
            f"{reject}, try again",
            f"{reject}, that's not right",
            f"{reject}, I mean something else",
        ]
        dataset.append({
            "intent": "REJECT",
            "text": random.choice(variations)
        })
    
    # 5. CHITCHAT intents (100 samples)
    chitchats = [
        "How are you?", "How's it going?", "What's up?",
        "How are you doing?", "What are you doing?",
        "What's the weather like?", "How's your day?",
        "Are you a robot?", "Are you human?", "What are you?",
        "Who are you?", "What's your name?", "Tell me about yourself",
        "Where are you from?", "How old are you?",
        "Do you like movies?", "What's your favorite movie?",
        "Can you think?", "Are you intelligent?", "Do you have feelings?",
        "What can you do?", "What are your capabilities?",
        "Tell me a joke", "Make me laugh", "Say something funny",
        "What time is it?", "What day is it?", "How's the weather?",
        "I'm bored", "I'm tired", "I'm happy", "I'm sad"
    ]
    
    for _ in range(100):
        chitchat = random.choice(chitchats)
        variations = [
            chitchat,
            f"{chitchat}?",
            f"Hey, {chitchat.lower()}",
            f"Quick question: {chitchat.lower()}",
        ]
        dataset.append({
            "intent": "CHITCHAT",
            "text": random.choice(variations)
        })
    
    # 6. OTHER intents (50 samples)
    others = [
        "What's 2+2?", "How do I cook pasta?", "Tell me about cars",
        "What's the capital of France?", "How does a computer work?",
        "Explain quantum physics", "What is love?",
        "How to learn programming?", "What's the meaning of life?",
        "Tell me about sports", "How to play guitar?",
        "What's the best smartphone?", "How to lose weight?",
        "Explain machine learning", "What is cryptocurrency?",
        "How to start a business?", "Tell me about history",
        "What's the weather in Paris?", "How to travel cheap?",
        "What is blockchain?", "How does internet work?",
        "Tell me about politics", "What's the stock market?",
        "How to improve memory?", "What is meditation?",
        "Tell me about space", "How to learn languages?",
    ]
    
    for _ in range(50):
        other = random.choice(others)
        dataset.append({
            "intent": "OTHER",
            "text": other
        })
    
    return dataset

# Generate the dataset
data = generate_intent_dataset()

# Convert to DataFrame
df = pd.DataFrame(data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("intent_classification_dataset.csv", index=False)

print(f"✅ Dataset created with {len(df)} samples")
print(f"\nIntent distribution:")
print(df['intent'].value_counts())
print(f"\n📁 File saved as: intent_classification_dataset.csv")