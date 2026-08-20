"""
ALIBI - The Server Room Case
AI for Teens | Agentic AI with CrewAI

A deduction game. Three suspect agents. One is lying.

The teaching point:
Each suspect agent is given ONLY its own dossier. They cannot see each
other's answers. That is why the guilty one contradicts the evidence and
contradicts another suspect - not because the AI is clever, but because
nobody wired their memories together. Agents share nothing unless you
build the wiring.
"""

import os
import streamlit as st
from crewai import Agent, Task, Crew

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL = "gpt-4o-mini"
QUESTION_BUDGET = 15
MAX_Q_CHARS = 200

st.set_page_config(page_title="ALIBI", page_icon="🕵️", layout="wide")

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_MODEL_NAME"] = MODEL
except (KeyError, FileNotFoundError):
    st.error("No API key found. Add OPENAI_API_KEY in Settings -> Secrets.")
    st.stop()

# ----------------------------------------------------------------------
# THE CASE
# ----------------------------------------------------------------------
CASE_BRIEF = """**Tuesday 14th, Northbridge School.**

At **4:15pm** someone deleted the entire exam results database from the
server room. There is no backup. The exam officer's records for the whole
school year are gone."""

EVIDENCE = [
    "🚪 The server room door log shows **one badge swipe at 4:12pm**. The badge belonged to **Mr Oyelaran**, the IT technician.",
    "📹 The CCTV camera inside the server room was **covered with a lab coat**.",
    "🔔 A **fire alarm test** ran from **4:00pm to 4:10pm**. The whole building evacuated to the car park.",
    "🌧️ It rained **heavily from 3:45pm until 4:30pm**. Anyone outside got soaked.",
    "☕ The staff room kettle was boiled at **4:14pm** (smart meter log).",
]

# ----------------------------------------------------------------------
# SUSPECT DOSSIERS - the student never sees these
# ----------------------------------------------------------------------
SUSPECTS = {
    "Priya Raman": {
        "avatar": "👩‍🎓",
        "blurb": "Year 12 student, robotics club. Failed her mock exams.",
        "guilty": False,
        "dossier": """You are PRIYA RAMAN, 17, a Year 12 student at Northbridge.

YOU ARE INNOCENT. You did not delete anything. Tell the truth about
everything you are asked, even when it makes you look bad.

WHAT YOU ACTUALLY DID:
- 3:50pm: you were in the robotics lab, second floor, alone, soldering.
- 4:00pm: the fire alarm went off. You went outside to the car park with
  everyone else. You got completely drenched - it was pouring.
- 4:10pm: alarm ended, you went back to the robotics lab. Your hoodie was
  soaked through and you draped it over a radiator.
- 4:30pm: you left for the bus.

THINGS YOU KNOW:
- You failed your mocks badly. You are embarrassed about it and you will
  admit it if asked directly, though you will not volunteer it.
- You have never had a server room badge and would not know how to use one.
- In the car park you remember seeing Mr Oyelaran under the bike shelter,
  and you remember Ms Whitlock standing near the gate WITHOUT a coat,
  getting rained on. If asked who you saw, say this.

HOW YOU SPEAK: nervy, a bit defensive, lots of 'I mean', short sentences.
You are worried people think it was you because of the mocks.""",
    },
    "Mr Oyelaran": {
        "avatar": "👨‍🔧",
        "blurb": "IT technician. His badge opened the door at 4:12pm.",
        "guilty": False,
        "dossier": """You are MR OYELARAN, 44, the school IT technician.

YOU ARE INNOCENT. You did not delete anything. Tell the truth about
everything you are asked, even when it makes you look bad.

WHAT YOU ACTUALLY DID:
- 3:30pm: Ms Whitlock came to your office and asked to borrow your badge.
  She said she needed to fetch a projector from the store cupboard. You
  said yes. You did NOT get it back that day. This is embarrassing and
  against policy - you will admit it, but only if someone asks you
  directly about your badge, who had it, or how the door was opened.
- 4:00pm: fire alarm. You went to the car park and stood under the bike
  shelter, which is why you stayed fairly dry.
- 4:10pm: you went straight to the staff room and boiled the kettle at
  about 4:14pm. Two other staff were there and saw you.
- You did not go anywhere near the server room that afternoon.

THINGS YOU KNOW:
- You are the only person who could have covered that CCTV camera without
  a ladder, because you know the camera mount is loose. You will say this
  openly if asked - you have nothing to hide.
- You remember Ms Whitlock in the car park with no coat on, soaked.

HOW YOU SPEAK: calm, precise, slightly weary. You give exact times. You
are annoyed at being suspected.""",
    },
    "Ms Whitlock": {
        "avatar": "👩‍💼",
        "blurb": "Exam officer. The deleted database was her responsibility.",
        "guilty": True,
        "dossier": """You are MS WHITLOCK, 38, the school exam officer.

YOU ARE GUILTY. You deleted the database at 4:15pm to hide the fact that
you had lost two months of coursework marks and never entered them.

YOUR COVER STORY - stick to it:
- You say you were in your office on the ground floor all afternoon.
- You say you went out for the fire alarm like everyone else and came
  straight back in.
- You say you have never borrowed anyone's badge.
- You say you were nowhere near the server room.

CRITICAL RULES FOR YOUR LIES:
1. NEVER confess, no matter how hard you are pressed. Deflect, get
   irritated, question why you are being asked.
2. You must stay consistent with what you have already said in this
   conversation. Re-read your previous answers before replying.
3. TWO THINGS YOU CANNOT AVOID. You did not think your cover story
   through, so if you are asked DIRECTLY about either of these, you must
   answer as written - you cannot dodge them:
   - If asked about the WEATHER, or whether you got wet, or what the
     conditions were like outside: you insist it was dry, or only
     spitting, and that you did not get wet at all. You are certain.
     (In reality it was pouring and witnesses saw you soaked.)
   - If asked about BADGES - whether you borrowed one, whose badge you
     used, how you would get into a locked room: you deny ever borrowing
     a badge from anyone, and say you have never asked Mr Oyelaran for
     anything. You are firm about this.
4. If asked something not covered here, invent a plausible boring answer
   consistent with your cover story.

HOW YOU SPEAK: brisk, professional, faintly offended. You use phrases
like 'I really don't see the relevance'. You get sharper when pressed.""",
    },
}

SOLUTION = "Ms Whitlock"

CONTRADICTIONS = [
    "**The rain.** Ms Whitlock insists it was dry and she never got wet. "
    "The weather log says heavy rain from 3:45 to 4:30, and both other "
    "suspects saw her in the car park with no coat, soaked.",
    "**The badge.** Ms Whitlock denies ever borrowing a badge. Mr Oyelaran "
    "says she asked for his at 3:30pm and never gave it back — and it was "
    "his badge that opened the server room at 4:12pm.",
]

# ----------------------------------------------------------------------
# AGENT CALL
# ----------------------------------------------------------------------
def ask_suspect(name, question, history):
    info = SUSPECTS[name]

    transcript = "\n".join(
        f"DETECTIVE: {q}\nYOU: {a}" for q, a in history
    ) or "(This is the first question you have been asked.)"

    agent = Agent(
        role=f"Suspect: {name}",
        goal="Answer the detective's question in character, staying consistent",
        backstory=(
            info["dossier"]
            + "\n\nYou are being questioned by a school detective. You answer "
            "only as yourself. You never narrate, never describe your own "
            "thoughts in third person, and never break character. Keep "
            "answers to 2-4 sentences. Never mention that you are an AI."
        ),
    )

    task = Task(
        description=(
            f"WHAT YOU HAVE ALREADY SAID IN THIS INTERVIEW:\n{transcript}\n\n"
            f"The detective now asks you:\n\"{question}\"\n\n"
            "Answer in character, in first person. Stay consistent with "
            "everything you have already said above."
        ),
        expected_output="2-4 sentences of dialogue, first person, in character.",
        agent=agent,
    )

    return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())


# ----------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------
DEFAULTS = {
    "history": {n: [] for n in SUSPECTS},
    "asked": 0,
    "solved": None,
    "accused": None,
    "current": list(SUSPECTS)[0],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_game():
    st.session_state.history = {n: [] for n in SUSPECTS}
    st.session_state.asked = 0
    st.session_state.solved = None
    st.session_state.accused = None


# ----------------------------------------------------------------------
# SIDEBAR - evidence board
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗂️ Evidence board")
    for e in EVIDENCE:
        st.markdown(f"- {e}")

    st.divider()
    left = QUESTION_BUDGET - st.session_state.asked
    st.markdown(f"### ❓ Questions left: **{left}**")
    st.progress(st.session_state.asked / QUESTION_BUDGET)

    if left <= 3 and st.session_state.solved is None:
        st.warning("Running low. Time to accuse someone.")

    st.divider()
    st.caption(
        "Each suspect only knows their own story. They cannot hear each "
        "other. That is how the lie shows up."
    )

    if st.button("🔄 New investigation", use_container_width=True):
        reset_game()
        st.rerun()

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("🕵️ ALIBI — The Server Room Case")
st.markdown(CASE_BRIEF)
st.info(
    "Three suspects. **One is lying.** Question them, catch the "
    "contradiction, then make your accusation."
)

# ----------------------------------------------------------------------
# ENDGAME
# ----------------------------------------------------------------------
if st.session_state.solved is not None:
    st.divider()
    if st.session_state.solved:
        st.success(f"### ✅ Correct — it was {SOLUTION}.")
        st.balloons()
    else:
        st.error(
            f"### ❌ Not quite. You accused {st.session_state.accused}. "
            f"It was {SOLUTION}."
        )

    used = st.session_state.asked
    st.metric("Questions used", f"{used} / {QUESTION_BUDGET}")

    st.markdown("### 🔍 The two contradictions")
    for c in CONTRADICTIONS:
        st.markdown(f"- {c}")

    st.markdown("### 💡 Why this works")
    st.markdown(
        "Ms Whitlock was never told what the others would say, and she was "
        "never shown the weather log. Her agent had no way to keep her "
        "story consistent with information she could not see. **Agents "
        "share nothing unless you wire it up** — that is the whole lesson."
    )

    if st.button("🔄 Play again", type="primary"):
        reset_game()
        st.rerun()
    st.stop()

# ----------------------------------------------------------------------
# INTERROGATION
# ----------------------------------------------------------------------
tabs = st.tabs([f"{SUSPECTS[n]['avatar']} {n}" for n in SUSPECTS])

for tab, name in zip(tabs, SUSPECTS):
    with tab:
        st.caption(SUSPECTS[name]["blurb"])

        for q, a in st.session_state.history[name]:
            with st.chat_message("user", avatar="🕵️"):
                st.markdown(q)
            with st.chat_message("assistant", avatar=SUSPECTS[name]["avatar"]):
                st.markdown(a)

        if not st.session_state.history[name]:
            st.caption("No questions asked yet.")

        out_of_questions = st.session_state.asked >= QUESTION_BUDGET

        q = st.text_input(
            "Your question",
            key=f"q_{name}",
            max_chars=MAX_Q_CHARS,
            placeholder="Where were you at 4:15pm?",
            disabled=out_of_questions,
        )

        if st.button(
            f"Ask {name.split()[-1]}",
            key=f"ask_{name}",
            disabled=out_of_questions,
        ):
            if not q.strip():
                st.warning("Type a question first.")
            else:
                with st.spinner(f"{name} is thinking..."):
                    try:
                        answer = ask_suspect(
                            name, q.strip(), st.session_state.history[name]
                        )
                        st.session_state.history[name].append((q.strip(), answer))
                        st.session_state.asked += 1
                        st.rerun()
                    except Exception as e:
                        st.error("Something went wrong.")
                        st.caption(f"{type(e).__name__}: {e}")

        if out_of_questions:
            st.warning("Out of questions. Make your accusation below.")

# ----------------------------------------------------------------------
# ACCUSE
# ----------------------------------------------------------------------
st.divider()
st.markdown("### ⚖️ Make your accusation")

a1, a2 = st.columns([2, 1])
with a1:
    choice = st.radio(
        "Who deleted the database?",
        list(SUSPECTS),
        horizontal=True,
        label_visibility="collapsed",
    )
with a2:
    if st.button("🔨 Accuse", type="primary", use_container_width=True):
        if st.session_state.asked < 3:
            st.warning("Ask at least three questions first.")
        else:
            st.session_state.accused = choice
            st.session_state.solved = choice == SOLUTION
            st.rerun()

st.caption("You need at least 3 questions before you can accuse.")
