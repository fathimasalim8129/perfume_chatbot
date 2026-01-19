import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ---------------- LOAD ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Perfume Chatbot",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Perfume Chatbot")
st.caption("A friendly AI to help you discover perfumes ✨")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset/perfumes_dataset_cleaned.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧴 Explore Options")

# ---- Brands ----
with st.sidebar.expander("Available Brands"):
    brands = sorted(df["brand"].dropna().unique())
    for b in brands:
        st.write(f"• {b}")

# ---- Categories ----
with st.sidebar.expander("Fragrance Categories"):
    categories = sorted(df["category"].dropna().unique())
    for c in categories:
        st.write(f"• {c}")

# ✅ ---- Mood (ADDED) ----
if "mood" in df.columns:
    with st.sidebar.expander("Mood Ideas"):
        moods = sorted(df["mood"].dropna().unique())
        for m in moods:
            st.write(f"• {m}")

# ✅ ---- Occasion (ADDED) ----
if "occasion" in df.columns:
    with st.sidebar.expander("Occasion Ideas"):
        occasions = sorted(df["occasion"].dropna().unique())
        for o in occasions:
            st.write(f"• {o}")

# ---------------- SIDEBAR: MOOD IDEAS ----------------
with st.sidebar.expander("Mood Ideas"):
    mood_ideas = [
        "Fresh",
        "Romantic",
        "Bold",
        "Elegant",
        "Calm",
        "Energetic"
    ]
    for m in mood_ideas:
        st.write(f"• {m}")

# ---------------- SIDEBAR: OCCASION IDEAS ----------------
with st.sidebar.expander("Occasion Ideas"):
    occasion_ideas = [
        "Daily wear",
        "Office",
        "Party",
        "Date night",
        "Special events",
        "Casual outing"
    ]
    for o in occasion_ideas:
        st.write(f"• {o}")


# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.stage = "gender"
    st.session_state.user_pref = {
        "gender": None,
        "brand": None,
        "fragrance": None,
        "mood": None,
        "occasion": None
    }
    st.session_state.last_results = None

# ---------------- RESET ----------------
if st.button("🔄 Start New Conversation"):
    st.session_state.clear()
    st.rerun()

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- BOT FUNCTION ----------------
def bot_say(text):
    st.session_state.messages.append({"role": "assistant", "content": text})
    with st.chat_message("assistant"):
        st.write(text)

# ---------------- AI DESCRIPTION ----------------
def generate_ai_description(perfume):
    prompt = (
        f"Describe the perfume {perfume['perfume']} by {perfume['brand']} "
        f"in a warm, friendly, conversational tone. "
        f"It is a {perfume['category']} fragrance for {perfume['target_audience']}. "
        f"Write a complete paragraph and finish naturally."
    )
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350
        )
        return response.choices[0].message.content
    except:
        return (
            "This perfume has a beautiful balance of character and elegance, "
            "making it a wonderful choice for everyday wear ✨"
        )

# ---------------- FIRST BOT MESSAGE ----------------
if len(st.session_state.messages) == 0:
    bot_say(
        "Hi there! 👋\n"
        "I’ll help you discover a perfume you’ll love 🌸\n\n"
        "**Who are you shopping for?**\n"
        "👉 Male / Female / Unisex"
    )

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Type your message...")

if user_input:
    user_input = user_input.lower().strip()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # ---------- STAGE 1: GENDER ----------
    if st.session_state.stage == "gender":
        if user_input in ["male", "female", "unisex"]:
            st.session_state.user_pref["gender"] = user_input
            st.session_state.stage = "brand"
            bot_say("Great 😊 Do you have a **preferred brand**? (or say *no preference*)")
        else:
            bot_say("Please choose **Male / Female / Unisex** 🌸")

    # ---------- STAGE 2: BRAND ----------
    elif st.session_state.stage == "brand":
        if user_input in ["no", "no preference", "any", "anything"]:
            st.session_state.user_pref["brand"] = None
        else:
            st.session_state.user_pref["brand"] = user_input
        st.session_state.stage = "fragrance"
        bot_say("Nice 🌼 What **fragrance style** do you enjoy? (or *no preference*)")

    # ---------- STAGE 3: FRAGRANCE ----------
    elif st.session_state.stage == "fragrance":
        if user_input in ["no", "no preference", "any", "anything"]:
            st.session_state.user_pref["fragrance"] = None
        else:
            st.session_state.user_pref["fragrance"] = user_input
        st.session_state.stage = "mood"
        bot_say("Lovely! 😊 What **mood** are you looking for? (or *no preference*)")

    # ---------- STAGE 4: MOOD ----------
    elif st.session_state.stage == "mood":
        if user_input in ["no", "no preference", "any", "anything"]:
            st.session_state.user_pref["mood"] = None
        else:
            st.session_state.user_pref["mood"] = user_input
        st.session_state.stage = "occasion"
        bot_say("Great! 🌸 For which **occasion** will this perfume be used? (or *no preference*)")

    # ---------- STAGE 5: OCCASION ----------
    elif st.session_state.stage == "occasion":
        if user_input in ["no", "no preference", "any", "anything"]:
            st.session_state.user_pref["occasion"] = None
        else:
            st.session_state.user_pref["occasion"] = user_input

        gender = st.session_state.user_pref["gender"]
        brand = st.session_state.user_pref["brand"]
        fragrance = st.session_state.user_pref["fragrance"]
        mood = st.session_state.user_pref["mood"]
        occasion = st.session_state.user_pref["occasion"]

        results = df[df["target_audience"].str.lower() == gender]

        if brand:
            results = results[results["brand"].str.lower().str.contains(brand, na=False)]
        if fragrance:
            results = results[results["category"].str.lower().str.contains(fragrance, na=False)]
        if "mood" in df.columns and mood:
            results = results[results["mood"].str.lower().str.contains(mood, na=False)]
        if "occasion" in df.columns and occasion:
            results = results[results["occasion"].str.lower().str.contains(occasion, na=False)]

        if not results.empty:
            perfume = results.sample(1).iloc[0]
            description = generate_ai_description(perfume)
            bot_say(
                f"🌸 **{perfume['perfume']}** by **{perfume['brand']}**\n\n"
                f"{description}\n\nWould you like to explore another perfume? 😊"
            )
            st.session_state.stage = "done"
        else:
            st.session_state.last_results = df[df["target_audience"].str.lower() == gender]
            bot_say("Hmm 🤔 No exact match. Would you like me to suggest something **similar**?")
            st.session_state.stage = "similar"

    # ---------- STAGE 6: SIMILAR ----------
    elif st.session_state.stage == "similar":
        if user_input in ["yes", "y", "sure", "ok"]:
            perfume = st.session_state.last_results.sample(1).iloc[0]
            description = generate_ai_description(perfume)
            bot_say(
                f"🌸 **{perfume['perfume']}** by **{perfume['brand']}**\n\n"
                f"{description}\n\n"
                "Hope you like this one ✨\n\n"
                "If you want, you can start a new conversation by clicking the button above 🌸"
            )
        else:
            bot_say(
                "No worries 😊 You can start a new chat anytime 🌸\n\n"
                "Or type 'yes' if you want a similar recommendation 🌼"
            )

        st.session_state.stage = "done"
