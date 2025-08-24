from textwrap import dedent
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
import streamlit as st
from agno.models.google import Gemini
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
else:
    st.error("Google API Key is missing. Please set it in your .env file.")

st.title("✈️ AI Travel Planner")
st.caption("Plan your dream trip with personalized recommendations powered by Gemini and SerpAPI")

# API Key Inputs
gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
serp_api_key = st.text_input("Enter SerpAPI Key", type="password")

if gemini_api_key and serp_api_key:

    # Personalization Inputs
    destination = st.text_input("Where do you want to go?")
    num_days = st.number_input("Trip Duration (days)", min_value=1, max_value=30, value=5)

    budget = st.selectbox("What's your budget like?", ["Low", "Mid-range", "Luxury"])
    travel_style = st.selectbox("Your travel style?", ["Solo", "Couple", "Family", "Friends"])
    accommodation = st.selectbox("Preferred accommodation?", ["Hotel", "Hostel", "Airbnb", "Resort", "Doesn't Matter"])
    interests = st.multiselect(
        "What are you most interested in?",
        ["Nature", "Adventure", "Culture", "Nightlife", "Relaxation", "Shopping", "History", "Local Food"]
    )
    pace = st.radio("Preferred pace of travel", ["Relaxed", "Balanced", "Packed"])

    if st.button("Generate My Travel Itinerary") and destination:
        with st.spinner("Researching..."):

            researcher = Agent(
                name="Researcher",
                role="Travel search expert",
                model=Gemini(id="gemini-2.0-flash-exp"),
                description=dedent(f"""
                You are a travel researcher. Use the user's preferences to search for top activities, places to stay, and must-see spots in {destination}.
                The user plans to travel for {num_days} days, prefers a {budget} trip, is traveling as a {travel_style.lower()}, wants {accommodation.lower()} type accommodation,
                and is interested in {', '.join(interests) or "various attractions"}. The user prefers a {pace.lower()} pace.
                Generate 3 relevant search queries and summarize the 10 most relevant results.
                """),
                instructions=[
                    "Generate 3 relevant Google search terms based on user preferences.",
                    "Use search_google to retrieve results for each term.",
                    "Return 10 summarized, high-quality results most aligned with the user's interests."
                ],
                tools=[SerpApiTools(api_key=serp_api_key)],
                add_datetime_to_instructions=True
            )

            planner = Agent(
                name="Planner",
                role="Travel itinerary generator",
                model=Gemini(id="gemini-2.0-flash-exp"),
                description=dedent("""
                You are an expert travel planner. Create a detailed itinerary based on destination, user preferences, and research insights.
                """),
                instructions=[
                    "Include activities, food recommendations, and stay suggestions that align with the user's preferences.",
                    "Organize the itinerary day-wise with a balance of excitement and rest based on their chosen travel pace.",
                    "Suggest optional evening activities and highlight any local tips or cultural etiquette.",
                    "Be realistic with travel times and locations. Make it engaging, insightful, and personalized."
                ],
                add_datetime_to_instructions=True
            )

            user_profile = f"""
            Destination: {destination}
            Days: {num_days}
            Budget: {budget}
            Travel Style: {travel_style}
            Accommodation: {accommodation}
            Interests: {', '.join(interests) or "General tourism"}
            Pace: {pace}
            """

            research_prompt = f"Plan a {budget} {travel_style} trip to {destination} for {num_days} days. Preferences: {accommodation} accommodation, Interests: {', '.join(interests)}. Pace: {pace}."
            research_results = researcher.run(research_prompt, stream=False)

        st.success("Research complete ✅")

        with st.spinner("Generating personalized itinerary..."):
            plan_prompt = f"""
            Use the following details to generate a day-by-day itinerary:
            {user_profile}
            Research Summary: {research_results.content}
            """
            itinerary = planner.run(plan_prompt, stream=False)
            st.markdown("## ✨ Your Personalized Itinerary")
            st.write(itinerary.content)






