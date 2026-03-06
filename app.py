import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
CSV_PATH = "data/clean_for_app.parquet"
df = pd.read_parquet(CSV_PATH)

st.set_page_config(
    layout="wide",
    initial_sidebar_state=360   # px width (e.g., 320–420 works nicely)
)


# Map choices -> target feature values (minimal but effective)
target = {
    "danceability": 0.60,
    "energy": 0.50,
    "valence": 0.50,
    "acousticness": 0.30,
    "speechiness": 0.15,
    "instrumentalness": 0.10,
    "liveness": 0.20,
    "tempo_norm": 0.50,
}

# Pick the feature columns that exist in your file
FEATURES = [c for c in [
    "danceability","energy","valence","acousticness",
    "instrumentalness","liveness","speechiness","tempo_norm"] if c in df.columns]




# Sidebar
with st.sidebar:
  st.title("Your Singer Finder")

  # Quick preview
  with st.expander("👀 Peek at the raw data"):
      st.write("Rows:", len(df))
      st.dataframe(df.head(), use_container_width=True)

  # Personality quiz (3 simple questions)
  st.subheader("🧋 Quick vibe check")
  mood = st.radio(
      "1) How do you feel today?",
      ["🌙 Calm & cozy", "😊 Happy & sunny", "🖤 Moody & emotional", "🔥 Hyped & bold"],
      index=None, horizontal=True)

  energy_pref = st.radio(
      "2) Energy level?",
      ["Low", "Medium", "High"],
      index=None, horizontal=True)

  tempo_feel = st.radio(
      "3) Tempo feel?",
      ["Slow", "Mid", "Fast"],
      index=None, horizontal=True)

  # Mood mapping
  if mood == "🌙 Calm & cozy":
      target.update(valence=0.55, energy=0.35, acousticness=0.50, danceability=0.50)
  elif mood == "😊 Happy & sunny":
      target.update(valence=0.80, energy=0.60, acousticness=0.25, danceability=0.70)
  elif mood == "🖤 Moody & emotional":
      target.update(valence=0.30, energy=0.45, acousticness=0.35, danceability=0.55)
  elif mood == "🔥 Hyped & bold":
      target.update(valence=0.65, energy=0.80, acousticness=0.15, danceability=0.80)

  # Energy preference overrides
  if energy_pref == "Low":
      target["energy"] = 0.35
  elif energy_pref == "Medium":
      target["energy"] = max(target["energy"], 0.55)
  elif energy_pref == "High":
      target["energy"] = 0.80

  # Tempo mapping
  if tempo_feel == "Slow":
      target["tempo_norm"] = 0.35
  elif tempo_feel == "Mid":
      target["tempo_norm"] = 0.55
  elif tempo_feel == "Fast":
      target["tempo_norm"] = 0.75

  # Build the target vector in the same order as FEATURES
  target_vec = np.array([target.get(f, 0.5) for f in FEATURES]).reshape(1, -1)

  # The quiz result
  with st.expander("🔬 Your vibe result vector"):
    st.json({f: round(target.get(f, 0.5), 2) for f in FEATURES})


# Gate: require all radios to be selected before computing results
all_selected = all(v is not None for v in (mood, energy_pref, tempo_feel))

if not all_selected:
    # Big hero-style heading before any results
    st.markdown(
        """
        <div style="text-align:center; padding: 6rem 0 2rem;">
            <h1 style="font-size:3rem; margin:0;">So… who matches your vibe?</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Stop here until the user completes all radios
    st.stop()


# Make a clean working copy with only the rows that have all needed features
work = df.dropna(subset=FEATURES).copy()

# Compute cosine similarity to each song
song_matrix = work[FEATURES].to_numpy(dtype=float, copy=False)
sims = cosine_similarity(target_vec, song_matrix)[0]
work["similarity"] = sims

# Return top unique artists (avoid repeats)
TOP_K = 3
top_rows, seen = [], set()
primary_artist_col = "artist_primary" if "artist_primary" in work.columns else "artists"

# If your CSV doesn’t have 'artist_primary', try to derive it on the fly
if "artist_primary" not in work.columns and "artists" in work.columns:
    work["artist_primary"] = (
        work["artists"].astype(str).str.split(r";|,", regex=True).str[0].str.strip()
    )
    primary_artist_col = "artist_primary"

for _, row in work.sort_values("similarity", ascending=False).iterrows():
    artist_name = row[primary_artist_col]
    if artist_name not in seen:
        top_rows.append(row)
        seen.add(artist_name)
    if len(top_rows) >= TOP_K:
        break
# --- Display matches (Top 3 with BuzzFeed-style copy)
st.subheader("✨ Your Matches")

if not top_rows:
    st.info("No matches found. Try a different vibe.")
else:
    for rank, row in enumerate(top_rows, start=1):
        artist_name = row[primary_artist_col]
        track_name = row.get("track_name", row.get("name", "(track)"))
        score = f"{row['similarity']:.2f}"

        # Pick the BuzzFeed line per rank
        if rank == 1:
            # Option 1 — “We crunched the numbers…”
            line = (
                f"🎉 Wow this track is basically made for your vibe -- *{track_name}*. **Match level: {score}**."
                )
        elif rank == 2:
            # Option 2 — “A MATCH MADE IN CHAOTIC‑GOOD HEAVEN”
            line = (
                f"🔥 Let's try this one -- *{track_name}*. **Match level: {score}**."
                )
        else:  # rank == 3
            # Option 3 — “Scientists Hate It…”
            line = (
                f"🤯 Well… let’s give it a chance *{track_name}*. **Match level: {score}**."
                )

        with st.container(border=True):
            st.markdown(f"### {rank}. **{artist_name}**")
            st.write(line)

            # Quick badges (if those columns exist)
            badges = []
            for f, emoji in [("energy", "⚡"), ("danceability", "💃"), ("valence", "😊")]:
                if f in work.columns and pd.notna(row.get(f)):
                    badges.append(f"{emoji} {f}: {row[f]:.2f}")
            if badges:
                st.caption(" · ".join(badges))


