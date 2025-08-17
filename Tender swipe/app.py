import os
import random
import json
import pandas as pd
import ast
import requests
from typing import List, Dict, Any, Optional
from flask import Flask, render_template, request, redirect, url_for, session
from groq import Groq

# --- 0. SETUP & CONFIGURATION ---

# Initialize Flask App
app = Flask(__name__)
# Sessions are used to track user's swipe count
app.secret_key = os.urandom(24) 

# Load API keys from environment variables
# Make sure you have set these in your terminal before running the app!
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")

# Define URLs for the raw CSV files on GitHub
USERS_URL = "https://raw.githubusercontent.com/galalqassas/tender/main/data/tinder_data/users.csv"
ACTIVITIES_URL = "https://raw.githubusercontent.com/galalqassas/tender/main/data/Activity.csv"
DISHES_URL = "https://raw.githubusercontent.com/galalqassas/tender/main/data/Dishes.csv"
ACCOMMODATIONS_URL = "https://raw.githubusercontent.com/galalqassas/tender/main/data/Accommodations.csv"

# --- 1. DATA LOADING & PREPARATION ---

def safe_literal_eval(val):
    """Safely evaluates a string literal; returns empty list on failure."""
    try:
        if pd.isna(val):
            return []
        evaluated = ast.literal_eval(val)
        return list(evaluated) if isinstance(evaluated, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []

print("--- 📥 Loading Data from GitHub... ---")

# Load and process USERS table
users_df = pd.read_csv(USERS_URL)
for col in ['interests', 'languages', 'preferredActivities', 'preferredCountries', 'likedUserIds', 'dislikedUserIds']:
    if col in users_df.columns:
        users_df[col] = users_df[col].apply(safe_literal_eval)
USERS = users_df.set_index('userId').to_dict('index')

# Load content tables
ACTIVITIES = pd.read_csv(ACTIVITIES_URL).to_dict('records')
DISHES = pd.read_csv(DISHES_URL).to_dict('records')
ACCOMMODATIONS = pd.read_csv(ACCOMMODATIONS_URL).to_dict('records')

ALL_CONTENT = {
    "Activity": ACTIVITIES,
    "Dish": DISHES,
    "Accommodation": ACCOMMODATIONS
}

# The UserSwipes Table (starts empty for each session)
USER_SWIPES = []
print("--- ✅ Data Loaded Successfully ---")


# --- 2. HELPER FUNCTIONS ---

def get_all_possible_preferences() -> Dict[str, set]:
    """Extracts all unique preference tags from the content data."""
    options = {"interests": set(), "travelStyle": set(), "preferredActivities": set()}
    # Define which columns map to which preference type
    activity_fields = {'Category', 'For', 'TypeOfTraveler'}
    dish_fields = {'Type', 'BestFor'}
    accommodation_fields = {'Type'}
    
    for item in ACTIVITIES:
        for field in activity_fields:
            if item.get(field) and pd.notna(item[field]):
                options['interests'].add(item[field])
                options['preferredActivities'].add(item[field])
    for item in DISHES:
        for field in dish_fields:
            if item.get(field) and pd.notna(item[field]):
                options['interests'].add(item[field])
    for item in ACCOMMODATIONS:
        for field in accommodation_fields:
            if item.get(field) and pd.notna(item[field]):
                options['travelStyle'].add(item[field])
    return {k: list(v) for k, v in options.items()}

def get_image_url(query: str, card_type: str) -> str:
    """Fetches a relevant image URL from Pexels API, avoiding people."""
    if not PEXELS_API_KEY:
        return "https://via.placeholder.com/400x300.png?text=Pexels+API+Key+Missing"

    # Add context to the query and exclude people for better results
    type_context = {
        "Activity": "landscape",
        "Dish": "food meal",
        "Accommodation": "building interior"
    }
    
    # Final query is more specific and tries to exclude people
    full_query = f"{query} {type_context.get(card_type, '')} -person -people"

    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": full_query, "per_page": 1, "orientation": "portrait"}
    try:
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["photos"]:
            return data["photos"][0]["src"]["large"]
    except requests.exceptions.RequestException as e:
        print(f"--- ❌ Pexels API Error: {e} ---")
    
    return "https://images.pexels.com/photos/3408744/pexels-photo-3408744.jpeg"

# --- 3. AI PREFERENCE SUGGESTION FUNCTION ---

def get_ai_preference_suggestions(liked_items: List[Dict], possible_preferences: Dict) -> List[str]:
    """Calls the Groq API to get real-time preference suggestions based on user likes."""
    print("\n--- 🤖 Calling Groq API to Synthesize Preferences ---")

    if not GROQ_API_KEY:
        print("--- ⚠️ GROQ_API_KEY not found. Skipping AI call. ---")
        return ["Outdoor", "Adventure"] # Fallback for demonstration

    client = Groq(api_key=GROQ_API_KEY)

    # Clean liked_items for the prompt, keeping only relevant fields
    prompt_items = [
        {k: v for k, v in item.items() if k in ['Activity', 'Dish Name', 'AccommodationName', 'Category', 'Type', 'For']}
        for item in liked_items
    ]

    prompt = f"""
    Analyze the user's recent liked items to understand their emerging travel preferences.
    Based ONLY on the provided list of liked items, suggest 3 to 5 new preference tags that best describe this user.

    RULES:
    1. Your suggestions MUST come from the "Possible Preference Tags" list provided below.
    2. Do not invent new tags.
    3. Return the answer as a single JSON object with one key: "suggestions", which holds a list of strings.

    **Liked Items:**
    {prompt_items}

    **Possible Preference Tags:**
    {possible_preferences}

    Example Response: {{"suggestions": ["Outdoor", "Budget Hostel", "Street Food"]}}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        suggestions = json.loads(response_content).get("suggestions", [])
        print(f"--- 🤖 Groq API Response Received: {suggestions} ---\n")
        return suggestions
    except Exception as e:
        print(f"--- ❌ Error calling Groq API: {e} ---")
        return []

# --- 4. THE CORE RECOMMENDER SYSTEM ---

class SwipeRecommender:
    def __init__(self):
        self.session_state = {}
        self.possible_preferences = get_all_possible_preferences()

    def _get_card_identifier(self, card: Dict) -> str:
        """Creates a unique string identifier for a card."""
        card_name = card.get('Activity') or card.get('Dish Name') or card.get('AccommodationName')
        return f"{card.get('type')}:{card_name}"

    def _get_user_session_state(self, user_id: int) -> Dict:
        if user_id not in self.session_state:
            profile = self._get_user_profile(user_id)
            has_prefs = bool(profile.get('interests') or profile.get('travelStyle') or profile.get('preferredActivities', []))
            discovery_streak = 10 if not has_prefs else 5
            self.session_state[user_id] = {
                "total_session_swipes": 0,
                "consecutive_similar_swipes": 0,
                "consecutive_dislikes": 0,
                "discovery_streak": discovery_streak
            }
        return self.session_state[user_id]

    def _get_user_profile(self, user_id: int) -> Dict:
        return USERS.get(user_id)

    def _get_similar_card(self, user_profile: Dict, seen_cards: list) -> Optional[Dict]:
        all_prefs = user_profile.get('interests', []) + user_profile.get('preferredActivities', [])
        if user_profile.get('travelStyle'):
            all_prefs.append(user_profile['travelStyle'])

        valid_prefs = [p for p in all_prefs if p and pd.notna(p)]
        if not valid_prefs:
            return None

        # Try multiple times to find a new card that matches preferences
        for _ in range(50): # Increased attempts
            chosen_pref = random.choice(valid_prefs)
            for content_type, content_list in ALL_CONTENT.items():
                # Shuffle list to get different results each time
                random.shuffle(content_list)
                for item in content_list:
                    full_card = {"type": content_type, **item}
                    identifier = self._get_card_identifier(full_card)
                    
                    if identifier in seen_cards:
                        continue # Skip if already seen
                    
                    if any(str(chosen_pref).lower() in str(v).lower() for v in item.values()):
                        return full_card
        return None # Return None if no unseen similar card is found

    def _get_discovery_card(self, seen_cards: list) -> Optional[Dict]:
        # Try multiple times to find a new, random card
        for _ in range(50): # Increased attempts
            content_type = random.choice(list(ALL_CONTENT.keys()))
            card_item = random.choice(ALL_CONTENT[content_type])
            card = {"type": content_type, **card_item}
            identifier = self._get_card_identifier(card)
            
            if identifier not in seen_cards:
                return card
        return None # Return None if no unseen discovery card is found

    def get_next_card(self, user_id: int, seen_cards: list) -> Optional[Dict]:
        session_data = self._get_user_session_state(user_id)
        profile = self._get_user_profile(user_id)
        
        if session_data['discovery_streak'] > 0:
            session_data['discovery_streak'] -= 1
            card = self._get_discovery_card(seen_cards)
            # If discovery fails, try finding a similar card as a fallback
            return card if card else self._get_similar_card(profile, seen_cards)
            
        if session_data['consecutive_similar_swipes'] > 0 and session_data['consecutive_similar_swipes'] % 5 == 0:
            card = self._get_discovery_card(seen_cards)
            return card if card else self._get_similar_card(profile, seen_cards)
        
        card = self._get_similar_card(profile, seen_cards)
        # If no similar card is found, try a discovery card
        return card if card else self._get_discovery_card(seen_cards)

    # process_swipe and other methods remain the same...
    def _trigger_preference_analysis(self, user_id: int) -> List[str]:
        """Analyzes recent likes and gets AI suggestions."""
        recent_likes = [
            swipe['card'] for swipe in USER_SWIPES
            if swipe['userId'] == user_id and swipe['liked']
        ][-10:]

        if not recent_likes:
            return []

        return get_ai_preference_suggestions(
            liked_items=recent_likes,
            possible_preferences=self.possible_preferences
        )

    def process_swipe(self, user_id: int, card: Dict, liked: bool):
        session_data = self._get_user_session_state(user_id)
        
        swipe_record = {
            "swipeId": len(USER_SWIPES) + 1, "userId": user_id, "cardType": card['type'],
            "cardIdentifier": self._get_card_identifier(card),
            "card": card, "liked": liked, "swipeTimestamp": "now"
        }
        USER_SWIPES.append(swipe_record)
        
        session_data['total_session_swipes'] += 1
        
        if liked:
            session_data['consecutive_similar_swipes'] += 1
            session_data['consecutive_dislikes'] = 0
        else:
            session_data['consecutive_similar_swipes'] = 0
            session_data['consecutive_dislikes'] += 1
            if session_data['consecutive_dislikes'] >= 3:
                session_data['discovery_streak'] = 5
                session_data['consecutive_dislikes'] = 0
        
        return {"status": "ok"}

    def confirm_and_update_preferences(self, user_id: int, confirmed_tags: List[str]):
        profile = self._get_user_profile(user_id)
        if not profile: return
        
        print(f"--- ✍️  Updating profile for User {user_id} with: {confirmed_tags} ---")
        for tag in confirmed_tags:
            if tag in self.possible_preferences['travelStyle'] and tag != profile.get('travelStyle'):
                profile['travelStyle'] = tag
            if tag in self.possible_preferences['interests'] and tag not in profile.get('interests', []):
                profile['interests'].append(tag)
            if tag in self.possible_preferences['preferredActivities'] and tag not in profile.get('preferredActivities', []):
                profile['preferredActivities'].append(tag)


# --- 5. FLASK APPLICATION ROUTES ---

# Instantiate the recommender
recommender = SwipeRecommender()
TARGET_USER_ID = 2  # Hardcoded to simulate for user "Ben"

# In app.py - Replace the @app.route('/') home function

@app.route('/')
def home():
    """Main route to display swipe cards."""
    if 'swipe_count' not in session:
        session['swipe_count'] = 0
        session['seen_cards'] = []  # Initialize seen cards for new session

    if session['swipe_count'] >= 20:
        return redirect(url_for('thank_you'))

    # Pass the list of seen cards to the recommender
    seen_cards_list = session.get('seen_cards', [])
    card = recommender.get_next_card(TARGET_USER_ID, seen_cards=seen_cards_list)

    # If no more unique cards can be found, end the session
    if not card:
        print("---  agotado All unique cards have been shown. ---")
        return redirect(url_for('thank_you'))

    # Add the new card's ID to the seen list and update the session
    card_id = recommender._get_card_identifier(card)
    seen_cards_list.append(card_id)
    session['seen_cards'] = seen_cards_list

    cleaned_card = {}
    for key, value in card.items():
        if pd.isna(value):
            cleaned_card[key] = None
        else:
            cleaned_card[key] = value
    
    card_json = json.dumps(cleaned_card)
            
    card_name = cleaned_card.get('Activity') or cleaned_card.get('Dish Name') or cleaned_card.get('AccommodationName')
    image_query = f"{card_name}, {cleaned_card.get('City')}"
    
    cleaned_card['image_url'] = get_image_url(image_query, cleaned_card.get('type'))
    
    return render_template('index.html', card=cleaned_card, card_json=card_json)


@app.route('/swipe', methods=['POST'])
def swipe():
    """Handles the like/dislike action."""
    card_data = json.loads(request.form.get('card_data'))
    action = request.form.get('action')
    liked = (action == 'like')
    
    session['swipe_count'] += 1
    print(f"--- Swipe #{session['swipe_count']} ---")
    print(f"Card: {card_data.get('Activity') or card_data.get('DishName') or card_data.get('AccommodationName')}, Action: {'Like' if liked else 'Dislike'}")

    recommender.process_swipe(TARGET_USER_ID, card_data, liked)
    
    # Check if it's time to refine preferences (at 10 swipes)
    if session['swipe_count'] % 10 == 0 and session['swipe_count'] > 0:
        ai_choices = recommender._trigger_preference_analysis(TARGET_USER_ID)
        if ai_choices:
            session['ai_choices'] = ai_choices
            return redirect(url_for('preferences'))

    return redirect(url_for('home'))

@app.route('/preferences')
def preferences():
    """Displays AI-suggested preferences for user confirmation."""
    choices = session.get('ai_choices', [])
    if not choices:
        return redirect(url_for('home'))
    return render_template('preferences.html', choices=choices)

@app.route('/update_preferences', methods=['POST'])
def update_preferences():
    """Updates user profile with confirmed tags."""
    confirmed_tags = request.form.getlist('confirmed_tags')
    recommender.confirm_and_update_preferences(TARGET_USER_ID, confirmed_tags)
    
    print("-" * 50 + "\nUpdated User Profile:")
    print(USERS[TARGET_USER_ID])
    print("-" * 50)
    
    # Clear the choices from session and continue swiping
    session.pop('ai_choices', None)
    return redirect(url_for('home'))

@app.route('/thank_you')
def thank_you():
    """Displays the final thank you page."""
    print("\n" + "="*50)
    print("🏁 SESSION FINISHED")
    print("="*50)
    print("Final User Profile for Ben:")
    print(USERS[TARGET_USER_ID])
    session.clear() # Clear session for a fresh start
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(debug=True)