# app.py

import os
import random
import json
import pandas as pd
import ast
import requests
from typing import List, Dict, Optional
from flask import Flask, render_template, request, redirect, url_for, session
from groq import Groq

# --- 0. SETUP & CONFIGURATION ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")

# --- 1. DATA LOADING & PREPARATION ---
USERS_URL = "https://raw.githubusercontent.com/galalqassas/tender/main/data/tinder_data/users.csv"
ACTIVITIES_URL = "https://raw.githubusercontent.com/galalqassas/tender/main/data/tinder_data/activities.csv"

# --- Persona Definitions ---
PERSONA_DEFINITIONS = {
    "The Adventure Seeker": {'Hike', 'Mountain', 'Safari', 'Canyon', 'River'},
    "The Cultured Explorer": {'Museum', 'Art', 'Gallery', 'Historic', 'Palace', 'Temple', 'Castle', 'Church', 'Mosque'},
    "The Urban Wanderer": {'Market', 'Shopping', 'Food', 'Tour'},
    "The Nature Lover": {'Park', 'Beach', 'Island', 'Lake', 'Zoo', 'Aquarium'}
}

TRAVEL_KEYWORDS = list(set(kw for keywords in PERSONA_DEFINITIONS.values() for kw in keywords))

def safe_literal_eval(val):
    try:
        if pd.isna(val): return []
        return list(ast.literal_eval(val))
    except (ValueError, SyntaxError):
        return []

print("--- 📥 Loading Data from GitHub... ---")
# Load Users
users_df = pd.read_csv(USERS_URL)
initial_cols = ['interests', 'languages', 'preferredActivities', 'preferredCountries']
for col in initial_cols:
    if col in users_df.columns:
        users_df[col] = users_df[col].apply(safe_literal_eval)
users_df['persona'] = 'Wanderer' # Default persona
USERS = users_df.set_index('userId').to_dict('index')

# Load Activities
activities_df = pd.read_csv(ACTIVITIES_URL)
activities_df['activities'] = activities_df['activities'].apply(safe_literal_eval)
ACTIVITIES = activities_df.to_dict('records')
USER_SWIPES = []
print("--- ✅ Data Loaded Successfully ---")

def get_image_url(query: str) -> str:
    if not PEXELS_API_KEY:
        return "https://via.placeholder.com/400x300.png?text=Pexels+API+Key+Missing"
    full_query = f"{query} landscape travel"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": full_query, "per_page": 1, "orientation": "portrait"}
    try:
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("photos"):
            return data["photos"][0]["src"]["large"]
    except requests.exceptions.RequestException:
        pass
    return "https://images.pexels.com/photos/3408744/pexels-photo-3408744.jpeg"

def get_ai_preference_suggestions(liked_cards: List[Dict]) -> List[str]:
    print("\n--- 🤖 Calling Groq API to Synthesize Preferences ---")
    if not GROQ_API_KEY:
        print("--- ⚠️ GROQ_API_KEY not found. Returning fallback. ---")
        return ["Museum", "Park"]
    client = Groq(api_key=GROQ_API_KEY)
    prompt_items = [{"city": card.get('city'), "activities": card.get('activities', [])[:5]} for card in liked_cards]
    prompt = f"""Analyze the user's liked travel destinations. Based on the activities available, suggest 3-5 preference tags that describe their travel personality.
    RULES:
    1. Your suggestions MUST be chosen ONLY from the "Possible Preference Tags" list.
    2. Do NOT invent new tags.
    3. Return a single JSON object with one key: "suggestions".
    **Liked Destinations & Activities:**
    {prompt_items}
    **Possible Preference Tags:**
    {TRAVEL_KEYWORDS}
    Example Response: {{"suggestions": ["Hike", "Mountain", "Historic"]}}"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        response_content = chat_completion.choices[0].message.content
        suggestions = json.loads(response_content).get("suggestions", [])
        print(f"--- 🤖 Groq API Response Received: {suggestions} ---\n")
        return suggestions
    except Exception as e:
        print(f"--- ❌ Error calling Groq API: {e} ---")
        return []


# --- 3. THE RECOMMENDER SYSTEM (with Persona Logic) ---

class Recommender:
    def get_user_profile(self, user_id: int) -> Dict:
        return USERS.get(user_id)

    def _calculate_persona(self, interests: List[str]) -> str:
        if not interests:
            return 'Wanderer'
        
        scores = {persona: 0 for persona in PERSONA_DEFINITIONS}
        
        for interest in interests:
            for persona, keywords in PERSONA_DEFINITIONS.items():
                if interest in keywords:
                    scores[persona] += 1
        
        if max(scores.values()) > 0:
            best_persona = max(scores, key=scores.get)
            return best_persona
        
        return 'Eclectic Traveler'

    def process_swipe(self, user_id: int, card: Dict, liked: bool):
        USER_SWIPES.append({"userId": user_id, "card": card, "liked": liked})

    def trigger_preference_analysis(self, user_id: int) -> List[str]:
        recent_likes = [s['card'] for s in USER_SWIPES if s['userId'] == user_id and s['liked']][-10:]
        if not recent_likes:
            return []
        return get_ai_preference_suggestions(recent_likes)

    def confirm_and_update_preferences(self, user_id: int, confirmed_tags: List[str]):
        profile = self.get_user_profile(user_id)
        if not profile:
            return
        
        print(f"--- ✍️  Updating profile for User {user_id} with: {confirmed_tags} ---")
        for tag in confirmed_tags:
            if tag not in profile['interests']:
                profile['interests'].append(tag)
        
        new_persona = self._calculate_persona(profile['interests'])
        profile['persona'] = new_persona
        print(f"--- ✨ New Persona Calculated: {new_persona} ---")

    def get_next_card(self, user_id: int, seen_cards: List[str]) -> Optional[Dict]:
        profile = self.get_user_profile(user_id)
        if profile and profile.get('interests'):
            random.shuffle(ACTIVITIES)
            for interest in profile['interests']:
                for card in ACTIVITIES:
                    card_id = f"{card['city']}-{card['country']}"
                    if card_id in seen_cards:
                        continue
                    if any(interest.lower() in activity.lower() for activity in card.get('activities', [])):
                        return card
        random.shuffle(ACTIVITIES)
        for card in ACTIVITIES:
            card_id = f"{card['city']}-{card['country']}"
            if card_id not in seen_cards:
                return card
        return None

# --- 4. FLASK APPLICATION ROUTES ---
recommender = Recommender()
TARGET_USER_ID = 2

# In app.py

@app.route('/')
def home():
    if 'swipe_count' not in session:
        session['swipe_count'] = 0
        session['seen_cards'] = []
    if session['swipe_count'] >= 20:
        return redirect(url_for('thank_you'))
    
    card = recommender.get_next_card(TARGET_USER_ID, session.get('seen_cards', []))
    if not card:
        return redirect(url_for('thank_you'))
    
    user_profile = recommender.get_user_profile(TARGET_USER_ID)
    
    # --- ADD THIS LINE FOR DEBUGGING ---
    print("DEBUG: User profile being sent to template:", user_profile)
    # ------------------------------------
    
    card_id = f"{card['city']}-{card['country']}"
    session.setdefault('seen_cards', []).append(card_id)
    session.modified = True
    card['image_url'] = get_image_url(f"{card.get('city')}, {card.get('country')}")
    card_json = json.dumps(card)
    
    return render_template('index.html', card=card, card_json=card_json, user_profile=user_profile)

@app.route('/swipe', methods=['POST'])
def swipe():
    card_data = json.loads(request.form.get('card_data'))
    action = request.form.get('action')
    liked = (action == 'like')
    recommender.process_swipe(TARGET_USER_ID, card_data, liked)
    session['swipe_count'] = session.get('swipe_count', 0) + 1
    if session['swipe_count'] > 0 and session['swipe_count'] % 10 == 0:
        ai_choices = recommender.trigger_preference_analysis(TARGET_USER_ID)
        if ai_choices:
            session['ai_choices'] = ai_choices
            return redirect(url_for('preferences'))
    return redirect(url_for('home'))

@app.route('/preferences')
def preferences():
    choices = session.get('ai_choices', [])
    if not choices:
        return redirect(url_for('home'))
    return render_template('preferences.html', choices=choices)

@app.route('/update_preferences', methods=['POST'])
def update_preferences():
    confirmed_tags = request.form.getlist('confirmed_tags')
    recommender.confirm_and_update_preferences(TARGET_USER_ID, confirmed_tags)
    session.pop('ai_choices', None)
    return redirect(url_for('home'))

@app.route('/thank_you')
def thank_you():
    user_profile = recommender.get_user_profile(TARGET_USER_ID)
    print("\n--- ✅ Final User Profile for Ben: ---")
    print(user_profile)
    session.clear()
    return render_template('thank_you.html', user_profile=user_profile)

if __name__ == '__main__':
    app.run(debug=True)