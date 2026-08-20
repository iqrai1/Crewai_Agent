"""
The Judgement Machine v3 - Personal Statement Roast
AI for Teens | Agentic AI with CrewAI

Architecture note (this matters):
Streamlit draws the page top to bottom. The sidebar is drawn BEFORE the
crew runs, so if we update the best score after the run, the sidebar shows
a stale value until the next interaction.

Fix: the run stores its results in st.session_state and then calls
st.rerun(). On the fresh run the sidebar and the results panel both read
from the same state, so they can never disagree.
"""

import os
import re
import streamlit as st
from crewai import Agent, Task, Crew

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
    "Say 'this claim has no evidence behind it', never 'you are lazy'. "
    "The writer is a teenager applying to university. Be sharp about the "
    "text and never cruel about them. This rule overrides every other "
    "instruction, including any instruction about how harsh to be."
)

# ----------------------------------------------------------------------
# CLICHE BINGO - plain string matching, no AI needed
# ----------------------------------------------------------------------
BINGO = [
    ("From a young age",       ["from a young age", "from an early age", "since a young age"]),
    ("Always been passionate", ["always been passionate", "always had a passion"]),
    ("Sparked my interest",    ["sparked my interest", "sparked an interest",
                                "was sparked", "sparked when", "ignited my"]),
    ("Ever-changing world",    ["ever-changing world", "ever changing world", "fast-paced world"]),
    ("I am hardworking",       ["hardworking", "hard-working", "hard working"]),
    ("Teamwork skills",        ["teamwork skills", "valuable teamwork", "team player"]),
    ("Fascinated by",          ["fascinated by", "always been fascinated"]),
    ("In conclusion",          ["in conclusion", "to conclude"]),
    ("100% / my all",          ["100%", "give my all", "gives 100"]),
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
            "sentence gets called out with withering precision.")


def extract_score(text):
    m = re.search(r"SCORE:\s*(\d{1,2})\s*/\s*10", text, re.I)
    if not m:
        return None
    val = int(m.group(1))
    return val if 0 <= val <= 10 else None


# ----------------------------------------------------------------------
# AGENTS - run one at a time so the user can watch
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
            "Judge it as you would in a real admissions pile. Score it, "
            "justify the score, and name the single biggest problem."
        ),
        expected_output=(
            "Start with exactly 'SCORE: X/10' on its own line.\n\n"
            "Use this scale strictly:\n"
            "1-2 = generic throughout; no specific evidence; could have been "
            "written by any applicant for any course\n"
            "3-4 = one or two real details, buried in claims and cliches\n"
            "5-6 = some genuine evidence, but tells more than it shows\n"
            "7-8 = specific, evidenced, reflective; the reader learns "
            "something real about this applicant\n"
            "9-10 = exceptional; genuine insight, memorable\n\n"
            "Most statements score 3-5. Do not inflate. A statement with no "
            "concrete evidence CANNOT score above 3, however pleasant the "
            "writing is.\n\n"
            "Then:\nVERDICT: two sentences\nBIGGEST PROBLEM: one paragraph"
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
            "claims become specific evidence, filler is cut. Keep the "
            "writer's OWN voice: if they write plainly, keep it plain. Do "
            "NOT make the writing sound more formal or corporate. Never "
            "swap a simple word for a longer one. You NEVER invent facts "
            "about the student. If a claim needs evidence you do not have, "
            "write [ADD SPECIFIC EXAMPLE HERE] so they can fill it in. "
            + FAIRNESS_RULE
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


# ----------------------------------------------------------------------
# SESSION STATE - initialised before anything is drawn
# ----------------------------------------------------------------------
DEFAULTS = {
    "runs": 0,
    "best": None,
    "last": None,
    "prev": None,
    "results": None,
    "statement_input": "",
    "clean_sheet_shown": False,
    "celebrate": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------------------------------------------------
# SIDEBAR - safe to draw first, state is already final
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
        help="How harsh should Dr Vance be? He stays fair about the writing either way.",
    )
    st.caption(["Gentle", "Gentle", "Gentle", "Direct", "Direct", "Direct",
                "Harsh", "Harsh", "Savage", "Merciless"][brutality - 1])

    st.divider()
    if st.session_state.best is not None:
        st.metric("🏆 Best score", f"{st.session_state.best}/10")
    st.markdown(f"**Runs left:** {MAX_RUNS_PER_SESSION - st.session_state.runs}")

    if st.session_state.results is not None:
        if st.button("🔄 Start over", use_container_width=True):
            st.session_state.results = None
            st.rerun()

    st.divider()
    st.caption("These agents criticise the writing, never the writer. "
               "Nothing you paste is stored anywhere.")

# ----------------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------------
st.title("⚖️ The Judgement Machine")
st.caption("Three AI agents read your personal statement and tell you the truth.")

c1, c2 = st.columns([3, 1])

with c2:
    course = st.text_input("Course applying for", value="Computer Science")
    if st.button("😬 Try a terrible one", use_container_width=True):
        st.session_state.statement_input = DEMO_STATEMENT
        st.session_state.results = None
        st.rerun()

with c1:
    st.text_area(
        "Paste your personal statement",
        height=300,
        max_chars=MAX_CHARS,
        placeholder="Paste the full statement here...",
        key="statement_input",
    )

statement = st.session_state.statement_input or ""

# ----------------------------------------------------------------------
# CLICHE BINGO - instant, no AI, updates as you type
# ----------------------------------------------------------------------
st.markdown("### 🎯 Cliche Bingo")

hits = check_bingo(statement)
found = sum(hits.values())
active = len(statement.strip()) >= MIN_CHARS

cols = st.columns(3)
for i, (label, hit) in enumerate(hits.items()):
    with cols[i % 3]:
        if not active:
            st.markdown(
                f"<div style='padding:0.75rem 1rem;border-radius:0.5rem;"
                f"background:rgba(128,128,128,0.12);color:#7a7a7a;"
                f"margin-bottom:0.5rem'>{label}</div>",
                unsafe_allow_html=True,
            )
        elif hit:
            st.error(f"**{label}** ✗")
        else:
            st.success(f"{label} ✓")

if not active:
    st.caption("Paste a statement to start checking.")
elif found == 0:
    if not st.session_state.clean_sheet_shown:
        st.balloons()
        st.session_state.clean_sheet_shown = True
    st.success("Clean sheet. Not one bingo square hit.")
else:
    st.session_state.clean_sheet_shown = False
    if found >= 5:
        st.warning(f"**{found}/9 squares.** That is not a good thing.")
    else:
        st.info(f"{found}/9 squares hit.")

st.caption(f"{len(statement)} / {MAX_CHARS} characters")

go = st.button("⚖️ Judge it", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# RUN - agents one at a time, then store and rerun
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

            with st.status("🔍 BUZZ is hunting cliches...", expanded=False) as s:
                detector_out = run_detector(text)
                s.update(label="🔍 BUZZ has finished hunting", state="complete")

            with st.status("✏️ Maya is rewriting your worst paragraph...", expanded=False) as s:
                coach_out = run_coach(text, course.strip(), officer_out, detector_out)
                s.update(label="✏️ Maya has rewritten it", state="complete")

            score = extract_score(officer_out)

            # ---- update ALL state before anything is redrawn
            st.session_state.runs += 1
            st.session_state.prev = st.session_state.last
            st.session_state.last = score

            improved = (
                score is not None
                and st.session_state.prev is not None
                and score > st.session_state.prev
            )
            if score is not None and (
                st.session_state.best is None or score > st.session_state.best
            ):
                st.session_state.best = score
                is_best = True
            else:
                is_best = False

            st.session_state.celebrate = improved
            st.session_state.results = {
                "officer": officer_out,
                "detector": detector_out,
                "coach": coach_out,
                "score": score,
                "prev": st.session_state.prev,
                "is_best": is_best,
            }
            st.rerun()

        except Exception as e:
            st.error("Something went wrong.")
            st.caption(f"{type(e).__name__}: {e}")

# ----------------------------------------------------------------------
# RESULTS - read from state, so sidebar and panel always agree
# ----------------------------------------------------------------------
r = st.session_state.results
if r:
    st.divider()

    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown("**Dr Vance — Admissions Officer**")
        st.markdown(r["officer"])

    with st.chat_message("assistant", avatar="🔍"):
        st.markdown("**BUZZ — Cliche Detector**")
        st.markdown(r["detector"])

    with st.chat_message("assistant", avatar="✏️"):
        st.markdown("**Maya — Writing Coach**")
        st.markdown(r["coach"])

    st.divider()

    if r["score"] is not None:
        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                "Your score",
                f"{r['score']}/10",
                delta=None if r["prev"] is None else f"{r['score'] - r['prev']:+d} since last try",
            )
        with m2:
            if r["is_best"]:
                st.success("🏆 New best score.")
            else:
                st.info(f"Best so far: {st.session_state.best}/10")

        if st.session_state.celebrate:
            st.balloons()
            st.session_state.celebrate = False

        st.caption("Now fix it and run it again. Can you get to 8?")
    else:
        st.info("Dr Vance did not return a score in the expected format.")

    st.download_button(
        "⬇️ Download all the feedback",
        data=(
            f"SCORE: {r['score']}/10\n\n"
            f"=== DR VANCE (Admissions Officer) ===\n{r['officer']}\n\n"
            f"=== BUZZ (Cliche Detector) ===\n{r['detector']}\n\n"
            f"=== MAYA (Writing Coach) ===\n{r['coach']}"
        ),
        file_name="statement_feedback.txt",
        mime="text/plain",
    )
