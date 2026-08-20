"""
The Judgement Machine v2 - Personal Statement Roast
AI for Teens | Agentic AI with CrewAI

What is new in v2:
  - agents run one at a time so you WATCH them work
  - each agent gets a name, a face, and its own chat bubble
  - your score is tracked - beat your previous attempt
  - brutality slider changes how harsh the Admissions Officer is
  - Cliche Bingo lights up as phrases are found
"""

import os
import re
import streamlit as st
from crewai import Agent, Task, Crew, Process

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MAX_CHARS = 6000
MIN_CHARS = 200
MAX_RUNS_PER_SESSION = 8
MODEL = "gpt-4o-mini"

st.set_page_config(page_title="The Judgement Machine", page_icon="⚖️", layout="wide")

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_MODEL_NAME"] = MODEL
except (KeyError, FileNotFoundError):
    st.error("No API key found. Add OPENAI_API_KEY in Settings -> Secrets.")
    st.stop()

FAIRNESS_RULE = (
    "CRITICAL RULE: You criticise the WRITING, never the PERSON. "
    "Say 'this sentence says nothing specific', never 'you are boring'. "
    "The writer is a teenager applying to university. Be sharp about the "
    "text and never cruel about them. This rule overrides every other "
    "instruction, including any instruction about how harsh to be."
)

# ----------------------------------------------------------------------
# CLICHE BINGO - plain string matching, no AI needed
# ----------------------------------------------------------------------
BINGO = [
    ("From a young age",      ["from a young age", "from an early age", "since a young age"]),
    ("Always been passionate", ["always been passionate", "always had a passion"]),
    ("Sparked my interest",   ["sparked my interest", "sparked an interest", "ignited my"]),
    ("Ever-changing world",   ["ever-changing world", "ever changing world", "fast-paced world"]),
    ("I am hardworking",      ["hardworking", "hard-working", "hard working"]),
    ("Teamwork skills",       ["teamwork skills", "valuable teamwork", "team player"]),
    ("Fascinated by",         ["fascinated by", "i have always been fascinated"]),
    ("In conclusion",         ["in conclusion", "to conclude"]),
    ("100% / my all",         ["100%", "give my all", "gives 100"]),
]

DEMO_STATEMENT = """From a young age I have always been passionate about computers and technology.
In today's ever-changing world, technology is more important than ever before.
I am a hardworking and dedicated student who always gives 100% to everything I do.

My interest in Computer Science was sparked when I got my first laptop. Since then
I have been fascinated by how things work behind the screen. I enjoy problem
solving and I believe I have strong analytical skills.

At school I study Maths, Physics and Computer Science. I am also a member of the
school football team, which has taught me valuable teamwork skills. I believe
these skills will help me greatly at university.

In conclusion, I am confident that I would be an excellent addition to your
university and I look forward to the opportunity."""


def check_bingo(text):
    low = text.lower()
    return {label: any(v in low for v in variants) for label, variants in BINGO}


def brutality_flavour(level):
    if level <= 3:
        return ("You are firm but encouraging. You point out problems clearly, "
                "but you always note what is working before what is not.")
    if level <= 6:
        return ("You are direct and unsentimental. You do not soften your "
                "findings, but you stay professional throughout.")
    if level <= 8:
        return ("You are blunt to the point of rudeness about the text. You "
                "have no patience for waffle and you say so plainly.")
    return ("You are utterly out of patience with weak writing. Every empty "
            "sentence gets called out with withering precision. You are still "
            "scrupulously fair about the WRITING and never about the person.")


# ----------------------------------------------------------------------
# AGENTS - built one at a time so we can run them one at a time
# ----------------------------------------------------------------------
def run_officer(statement, course, brutality):
    agent = Agent(
        role="University Admissions Officer",
        goal="Judge whether this statement would survive a real admissions pile",
        backstory=(
            "Your name is Dr Vance. You have read 4,000 personal statements "
            "this year alone, at 45 seconds each. You care about one thing: "
            "evidence, not claims of passion. "
            + brutality_flavour(brutality) + " " + FAIRNESS_RULE
        ),
    )
    task = Task(
        description=(
            f"Read this personal statement for the course: {course}\n\n"
            f"STATEMENT:\n{statement}\n\n"
            "Judge it as in a real admissions pile. Score out of 10, justify "
            "it, name the single biggest problem."
        ),
        expected_output=(
            "Start with exactly 'SCORE: X/10' on its own line. Then:\n"
            "VERDICT: two sentences\nBIGGEST PROBLEM: one paragraph"
        ),
        agent=agent,
    )
    return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())


def run_detector(statement):
    agent = Agent(
        role="Cliche Detector",
        goal="Find every tired, overused, or empty phrase",
        backstory=(
            "You are BUZZ, a machine built to spot phrases that appear in "
            "thousands of other statements, and empty claims like "
            "'hardworking' with no proof attached. You quote the exact "
            "phrase, then explain what is wrong with it in one line. "
            + FAIRNESS_RULE
        ),
    )
    task = Task(
        description=(
            f"Hunt for cliches and empty claims:\n\n{statement}\n\n"
            "Quote each exact phrase and explain in one line why it is weak."
        ),
        expected_output='Numbered list of 3-5 items: "phrase" -> why it is weak',
        agent=agent,
    )
    return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())


def run_coach(statement, course, officer_out, detector_out):
    agent = Agent(
        role="Writing Coach",
        goal="Rewrite the weakest paragraph so it actually works",
        backstory=(
            "Your name is Maya. You rebuild the weakest paragraph - vague "
            "claims become specific evidence, filler is cut, the writer's own "
            "voice stays. You NEVER invent facts about the student. If a claim "
            "needs evidence you do not have, write [ADD SPECIFIC EXAMPLE HERE] "
            "so they can fill it in themselves. " + FAIRNESS_RULE
        ),
    )
    task = Task(
        description=(
            f"Two colleagues reviewed this statement.\n\n"
            f"ADMISSIONS OFFICER SAID:\n{officer_out}\n\n"
            f"CLICHE DETECTOR SAID:\n{detector_out}\n\n"
            f"STATEMENT:\n{statement}\n\nCourse: {course}\n\n"
            "Pick the SINGLE weakest paragraph and rewrite it."
        ),
        expected_output=(
            "ORIGINAL:\n[paragraph as written]\n\nREWRITTEN:\n[your version]\n\n"
            "WHAT CHANGED:\n[3 bullets]"
        ),
        agent=agent,
    )
    return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())


def extract_score(text):
    m = re.search(r"SCORE:\s*(\d{1,2})\s*/\s*10", text, re.I)
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
for k, v in [("runs", 0), ("best", None), ("last", None), ("statement", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 👥 Your crew")
    st.markdown(
        "**⚖️ Dr Vance** — Admissions Officer\n\n"
        "**🔍 BUZZ** — Cliche Detector\n\n"
        "**✏️ Maya** — Writing Coach"
    )
    st.divider()

    brutality = st.slider(
        "🔥 Brutality", 1, 10, 5,
        help="How harsh should Dr Vance be? (He stays fair about the writing either way.)"
    )
    st.caption(["Gentle", "Gentle", "Gentle", "Direct", "Direct", "Direct",
                "Harsh", "Harsh", "Savage", "Merciless"][brutality - 1])

    st.divider()
    if st.session_state.best is not None:
        st.metric("🏆 Best score", f"{st.session_state.best}/10")
    st.markdown(f"**Runs left:** {MAX_RUNS_PER_SESSION - st.session_state.runs}")
    st.divider()
    st.caption("These agents criticise the writing, never the writer. "
               "Nothing you paste is stored anywhere.")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
st.title("⚖️ The Judgement Machine")
st.caption("Three AI agents read your personal statement and tell you the truth.")

c1, c2 = st.columns([3, 1])

with c2:
    course = st.text_input("Course applying for", value="Computer Science")
    if st.button("😬 Try a terrible one", use_container_width=True):
        st.session_state.statement = DEMO_STATEMENT
        st.rerun()

with c1:
    statement = st.text_area(
        "Paste your personal statement",
        value=st.session_state.statement,
        height=300,
        max_chars=MAX_CHARS,
        placeholder="Paste the full statement here...",
    )

# ---- Cliche Bingo (updates live as you type - no AI, just string matching)
st.markdown("### 🎯 Cliche Bingo")
hits = check_bingo(statement)
found = sum(hits.values())
cols = st.columns(3)
for i, (label, hit) in enumerate(hits.items()):
    with cols[i % 3]:
        if hit:
            st.error(f"**{label}** ✗")
        else:
            st.success(f"{label}")

if found == 0 and len(statement) > MIN_CHARS:
    st.balloons()
    st.success("Clean sheet. No bingo squares hit.")
elif found >= 5:
    st.warning(f"**{found}/9 squares.** That is not a good thing.")
elif found:
    st.info(f"{found}/9 squares hit.")

st.caption(f"{len(statement)} / {MAX_CHARS} characters")

go = st.button("⚖️ Judge it", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# RUN - agents one at a time, live
# ----------------------------------------------------------------------
if go:
    text = statement.strip()

    if st.session_state.runs >= MAX_RUNS_PER_SESSION:
        st.error("You have used all your runs. Refresh the page to reset.")
    elif len(text) < MIN_CHARS:
        st.warning(f"Too short — paste at least {MIN_CHARS} characters.")
    elif not course.strip():
        st.warning("Tell the crew which course you are applying for.")
    else:
        try:
            st.divider()

            with st.status("⚖️ Dr Vance is reading...", expanded=False) as s:
                officer_out = run_officer(text, course.strip(), brutality)
                s.update(label="⚖️ Dr Vance has reached a verdict", state="complete")
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown("**Dr Vance — Admissions Officer**")
                st.markdown(officer_out)

            with st.status("🔍 BUZZ is hunting cliches...", expanded=False) as s:
                detector_out = run_detector(text)
                s.update(label="🔍 BUZZ has finished hunting", state="complete")
            with st.chat_message("assistant", avatar="🔍"):
                st.markdown("**BUZZ — Cliche Detector**")
                st.markdown(detector_out)

            with st.status("✏️ Maya is rewriting your worst paragraph...", expanded=False) as s:
                coach_out = run_coach(text, course.strip(), officer_out, detector_out)
                s.update(label="✏️ Maya has rewritten it", state="complete")
            with st.chat_message("assistant", avatar="✏️"):
                st.markdown("**Maya — Writing Coach**")
                st.markdown(coach_out)

            # ---- scoring
            st.session_state.runs += 1
            score = extract_score(officer_out)

            if score is not None:
                st.divider()
                prev = st.session_state.last
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(
                        "Your score",
                        f"{score}/10",
                        delta=None if prev is None else f"{score - prev:+d} since last try",
                    )
                with m2:
                    if st.session_state.best is None or score > st.session_state.best:
                        st.session_state.best = score
                        st.success("🏆 New best score.")
                    else:
                        st.info(f"Best so far: {st.session_state.best}/10")

                if prev is not None and score > prev:
                    st.balloons()

                st.session_state.last = score
                st.caption("Now fix it and run it again. Can you get to 8?")

            st.download_button(
                "⬇️ Download all the feedback",
                data=(f"SCORE: {score}/10\n\n=== DR VANCE ===\n{officer_out}\n\n"
                      f"=== BUZZ ===\n{detector_out}\n\n=== MAYA ===\n{coach_out}"),
                file_name="statement_feedback.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error("Something went wrong.")
            st.caption(f"{type(e).__name__}: {e}")
