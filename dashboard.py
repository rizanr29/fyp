import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.graph_objects as go
import plotly.express as px

# ✅ Step 1: Authenticate Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credential.json", scope)
client = gspread.authorize(creds)

sheet = client.open("Sheetname").sheet1  # Replace with your actual sheet name

def load_data():
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# Load the data
df = load_data()

# Convert relevant columns to numeric
df["Improvement"] = pd.to_numeric(df["Improvement"], errors="coerce").fillna(0).astype(int)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0).astype(int)
df["Consistency"] = pd.to_numeric(df["Consistency"], errors="coerce").fillna(0)

# 🎯 **Header with Image**
st.image("1673861856868.jpeg", use_column_width=True)
st.markdown("<h1 class='header'>⚽ Juggling Performance Dashboard</h1>", unsafe_allow_html=True)

# 📌 **Grading Scale Section**
st.subheader("Grading Scale")
st.markdown("""
- **0-20**: Beginner
- **21-40**: Intermediate
- **41-60**: Advanced
- **61+**: Elite
""")

# 📊 **Grading Scale Visualization**
grading_scale = {
    "Grade": ["Beginner", "Intermediate", "Advanced", "Elite"],
    "Min Score": [0, 21, 41, 61],
    "Max Score": [20, 40, 60, 100]
}
grading_df = pd.DataFrame(grading_scale)

latest_rating = df.iloc[-1]["Rating"] if not df.empty else 0

fig_grading = go.Figure()
for i, row in grading_df.iterrows():
    fig_grading.add_trace(go.Bar(
        x=[row["Max Score"] - row["Min Score"]],  
        y=["Grading Scale"],  
        orientation="h",
        name=row["Grade"],
        marker_color=px.colors.qualitative.Plotly[i],  
        text=[f"{row['Min Score']}-{row['Max Score']}"],  
        textposition="inside",
        hoverinfo="none"
    ))

fig_grading.add_trace(go.Scatter(
    x=[latest_rating, latest_rating],  
    y=[-0.5, 1.5],  
    mode="lines",
    line=dict(color="red", width=3, dash="dash"),
    name=f"Latest Rating: {latest_rating}"
))

fig_grading.update_layout(
    title="Grading Scale with Latest Rating",
    xaxis_title="Score Range",
    yaxis_title="",
    barmode="stack",  
    showlegend=False,  
    height=300,  
    xaxis_range=[0, 100],  
    yaxis_visible=False,  
    annotations=[
        dict(
            x=latest_rating,
            y=1.5,
            xref="x",
            yref="y",
            text=f"Latest Rating: {latest_rating}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(size=12, color="red")
        )
    ]
)
st.plotly_chart(fig_grading)

# 📊 **Performance Metrics Visualization**
if not df.empty:
    latest_session = df.iloc[-1]

    st.subheader("Latest Performance Metrics")

    # 🔵 **Total Juggles**
    fig_juggles = go.Figure(go.Bar(
        x=["Total Juggles"], y=[latest_session["Total Juggles"]],
        text=[latest_session["Total Juggles"]], textposition="auto",
        marker=dict(color="blue")
    ))
    fig_juggles.update_layout(yaxis=dict(title="Juggles", range=[0, max(df["Total Juggles"]) + 10]))
    st.plotly_chart(fig_juggles)

    # 🟠 **Consistency (Max: 1.0)**
    fig_consistency = go.Figure(go.Bar(
        x=["Consistency"], y=[latest_session["Consistency"]],
        text=[round(latest_session["Consistency"], 2)], textposition="auto",
        marker=dict(color="orange")
    ))
    fig_consistency.update_layout(yaxis=dict(title="Consistency", range=[0, 1.0]))
    st.plotly_chart(fig_consistency)

    # 🟢 **Endurance**
    fig_endurance = go.Figure(go.Bar(
        x=["Endurance"], y=[latest_session["Endurance"]],
        text=[latest_session["Endurance"]], textposition="auto",
        marker=dict(color="green")
    ))
    fig_endurance.update_layout(yaxis=dict(title="Endurance", range=[0, max(df["Endurance"]) + 5]))
    st.plotly_chart(fig_endurance)

    # 🔴 **Improvement**
    fig_improvement = go.Figure(go.Bar(
        x=["Improvement"], y=[latest_session["Improvement"]],
        text=[latest_session["Improvement"]], textposition="auto",
        marker=dict(color="red")
    ))
    fig_improvement.update_layout(yaxis=dict(title="Improvement", range=[0, max(df["Improvement"]) + 5]))
    st.plotly_chart(fig_improvement)

    # 🟣 **Rating**
    fig_rating = go.Figure(go.Bar(
        x=["Rating"], y=[latest_session["Rating"]],
        text=[latest_session["Rating"]], textposition="auto",
        marker=dict(color="purple")
    ))
    fig_rating.update_layout(yaxis=dict(title="Rating", range=[0, 10]))
    st.plotly_chart(fig_rating)

    # 📌 **Player Improvement Feedback**
    st.subheader("📢 Player Improvement Feedback")
    
    feedback = ""
    if latest_session["Consistency"] < 0.6:
        feedback += "- Improve ball control by focusing on consistent touches.\n"
    if latest_session["Avg Time Gap (s)"] > 1.0:
        feedback += "- Try to maintain a steady rhythm with faster reactions.\n"
    if latest_session["Total Juggles"] < 20:
        feedback += "- Increase endurance by practicing longer juggling sessions.\n"
    if latest_session["Improvement"] == 0:
        feedback += "- Set personal goals for improvement and track progress.\n"
    
    st.write(feedback if feedback else "Great job! Keep practicing to refine your skills.")

else:
    st.write("No session data available.")
