"""
The Judgement Machine - Personal Statement Roast
AI for Teens | Agentic AI with CrewAI

Deployed version. Differences from the Colab version:
  - reads the API key from Streamlit secrets, not Colab secrets
  - caps input length so nobody can paste a novel
  - limits runs per session so one person cannot drain the API budget
  - caches the crew so it is not rebuilt on every rerun
"""

import os
import streamlit as st
from crewai import Agent, Task, Crew, Process

# ----------------------------------------------------------------------
# CONFIG - change these to control cost
# ----------------------------------------------------------------------
MAX_CHARS = 6000        # longest statement accepted
MIN_CHARS = 200         # shortest statement accepted
MAX_RUNS_PER_SESSION = 5
MODEL = "gpt-4o-mini"

st.set_page_config(
    page_title="The Judgement Machine",
    page_icon="⚖️",
    layout="wide",
)

# ----------------------------------------------------------------------
# API KEY - from Streamlit secrets
# ----------------------------------------------------------------------
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_MODEL_NAME"] = MODEL
except (KeyError, FileNotFoundError):
    st.error("No API key found. Add OPENAI_API_KEY in Settings -> Secrets.")
    st.stop()

# ----------------------------------------------------------------------
# THE CREW
# ----------------------------------------------------------------------
FAIRNESS_RULE = (
    "CRITICAL RULE: You criticise the WRITING, never the PERSON. "
    "Say 'this sentence says nothing specific', never 'you are boring'. "
    "Say 'this claim has no evidence behind it', never 'you are lazy'. "
    "The writer is a teenager applying to university. Be sharp about the "
    "text and never cruel about them."
)


@st.cache_resource
def build_crew():
    """Built once and reused. @st.cache_resource stops Streamlit
    rebuilding the whole crew every time the user types a character."""

    officer = Agent(
        role="University Admissions Officer",
        goal="Judge whether this statement would survive a real admissions pile",
        backstory=(
            "You have read 4,000 personal statements this year alone. You read "
            "each one for about 45 seconds. You have seen every opening line a "
            "thousand times and you are extremely hard to impress. You care "
            "about one thing: does this applicant show evidence, or are they "
            "just telling you they are passionate? " + FAIRNESS_RULE
        ),
    )

    detector = Agent(
        role="Cliche Detector",
        goal="Find every tired, overused, or empty phrase in the statement",
        backstory=(
            "You spot phrases that appear in thousands of other statements. "
            "'From a young age.' 'I have always been passionate about.' 'This "
            "sparked my interest.' You also catch empty claims - words like "
            "'hardworking' or 'dedicated' with no proof attached. You quote "
            "the exact phrase, then explain what is wrong with it. "
            + FAIRNESS_RULE
        ),
    )

    coach = Agent(
        role="Writing Coach",
        goal="Rewrite the weakest paragraph so it actually works",
        backstory=(
            "You take the single weakest paragraph and rebuild it - swapping "
            "vague claims for specific evidence, cutting filler, keeping the "
            "writer's own voice. You never invent facts about the student. If "
            "a claim needs evidence you do not have, you write [ADD SPECIFIC "
            "EXAMPLE HERE] so they can fill it in themselves. " + FAIRNESS_RULE
        ),
    )

    screen = Task(
        description=(
            "Read this personal statement for the course: {course}\n\n"
            "STATEMENT:\n{statement}\n\n"
            "Judge it as you would in a real admissions pile. Give a score out "
            "of 10 and justify it. Name the single biggest problem with it."
        ),
        expected_output=(
            "SCORE: X/10\n"
            "VERDICT: two sentences on whether this survives the pile\n"
            "BIGGEST PROBLEM: one paragraph"
        ),
        agent=officer,
    )

    cliches = Task(
        description=(
            "Hunt through this statement for cliches and empty claims.\n\n"
            "STATEMENT:\n{statement}\n\n"
            "Find the worst offenders. Quote the exact phrase and explain in "
            "one line why it is weak."
        ),
        expected_output=(
            'A numbered list of 3 to 5 items, each: "exact phrase" -> why it is weak'
        ),
        agent=detector,
    )

    fix = Task(
        description=(
            "Using what the other two found, pick the SINGLE weakest paragraph "
            "and rewrite it.\n\nSTATEMENT:\n{statement}\n\nCourse: {course}"
        ),
        expected_output=(
            "ORIGINAL:\n[the paragraph as written]\n\n"
            "REWRITTEN:\n[your version]\n\n"
            "WHAT CHANGED:\n[3 bullet points]"
        ),
        agent=coach,
        context=[screen, cliches],
    )

    return Crew(
        agents=[officer, detector, coach],
        tasks=[screen, cliches, fix],
        process=Process.sequential,
        verbose=False,          # keep server logs quiet in production
    )


# ----------------------------------------------------------------------
# RUN COUNTER
# ----------------------------------------------------------------------
if "runs" not in st.session_state:
    st.session_state.runs = 0

# ----------------------------------------------------------------------
# INTERFACE
# ----------------------------------------------------------------------
st.title("⚖️ The Judgement Machine")
st.caption("Three AI agents read your personal statement and tell you the truth.")

with st.sidebar:
    st.markdown("### Your crew")
    st.markdown(
        "**Admissions Officer** — scores it out of 10\n\n"
        "**Cliche Detector** — finds the waffle\n\n"
        "**Writing Coach** — rewrites your worst paragraph"
    )
    st.divider()
    st.markdown(
        f"**Runs left this session:** "
        f"{MAX_RUNS_PER_SESSION - st.session_state.runs}"
    )
    st.divider()
    st.caption(
        "These agents criticise the writing, never the writer. "
        "Nothing you paste is stored anywhere."
    )

col1, col2 = st.columns([3, 1])

with col2:
    course = st.text_input("Course applying for", value="Computer Science")

with col1:
    statement = st.text_area(
        "Paste your personal statement",
        height=340,
        max_chars=MAX_CHARS,
        placeholder="Paste the full statement here...",
    )
    st.caption(f"{len(statement)} / {MAX_CHARS} characters")

go = st.button("Judge it", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# GUARDS AND RUN
# ----------------------------------------------------------------------
if go:
    text = statement.strip()

    if st.session_state.runs >= MAX_RUNS_PER_SESSION:
        st.error("You have used all your runs for this session. Refresh to reset.")
    elif len(text) < MIN_CHARS:
        st.warning(f"That is too short — paste at least {MIN_CHARS} characters.")
    elif not course.strip():
        st.warning("Tell the crew which course you are applying for.")
    else:
        with st.spinner("Your crew is reading. This takes 30–60 seconds..."):
            try:
                result = build_crew().kickoff(
                    inputs={"statement": text, "course": course.strip()}
                )
                st.session_state.runs += 1
                st.success("Verdict in.")
                st.markdown("---")
                st.markdown(str(result))

                st.download_button(
                    "Download the feedback",
                    data=str(result),
                    file_name="statement_feedback.txt",
                    mime="text/plain",
                )
            except Exception as e:
                st.error("Something went wrong.")
                st.caption(f"Details: {type(e).__name__}: {e}")
