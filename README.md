# ⚖️ The Judgement Machine

**Live app → [academicagent.streamlit.app](https://academicagent.streamlit.app/)**

Three AI agents read your university personal statement and tell you the truth about it.

Built with **CrewAI** and **Streamlit**.

---

## What it does

Paste a personal statement, pick a course, and a crew of three agents goes to work — one at a time, so you can watch them think.

| Agent | What they do |
|---|---|
| ⚖️ **Dr Vance** — Admissions Officer | Scores it out of 10 against a strict rubric and names the biggest problem |
| 🔍 **BUZZ** — Cliché Detector | Quotes every tired phrase and empty claim back at you |
| ✏️ **Maya** — Writing Coach | Rewrites your weakest paragraph, keeping your own voice |

Plus two things that need no AI at all:

- **🎯 Cliché Bingo** — nine of the worst offenders, checked instantly as you type
- **🔥 Brutality slider** — 1 to 10, changing how harsh Dr Vance is

---

## Try it — copy and paste these

### 😬 The terrible one

Expect **9/9 bingo squares** and a score around **2–3**.

```
From a young age I have always been passionate about computers and technology.
In today's ever-changing world, technology is more important than ever before.
I am a hardworking and dedicated student who always gives 100% to everything I do.

My interest in Computer Science was sparked when I got my first laptop. Since then
I have been fascinated by how things work behind the screen. I enjoy problem
solving and I believe I have strong analytical skills.

At school I study Maths, Physics and Computer Science. I am also a member of the
school football team, which has taught me valuable teamwork skills. I believe
these skills will help me greatly at university.

In conclusion, I am confident that I would be an excellent addition to your
university and I look forward to the opportunity.
```

### ✅ The good one

Same course, same agents. Expect a **clean bingo sheet** and a score around **7–8**.

```
The first program I wrote that mattered was a scheduling script for my mother's
tailoring shop. She was tracking eleven customers' orders on paper and missing
deadlines. My script read a CSV of orders and printed what was due that week. It
worked for two months, then broke when she hired a second tailor, because I had
hardcoded a single worker. Rewriting it taught me more than building it did — I
had solved the problem in front of me rather than the problem she actually had.

That failure pushed me toward structure. I worked through CS50 last summer, then
wrote a small library system in Java specifically to force myself to use
inheritance rather than read about it. My current project is a Discord bot that
scrapes our exam timetables and sends reminders; it has forty users across two
schools, and their bug reports have been more instructive than the code. Someone
in a different timezone broke it within a day, which is how I learned that
storing local time was a mistake.

I take Maths, Physics and Computer Science at A Level. Physics has been
unexpectedly useful — mechanics trained me to sanity-check whether an answer is
plausible before trusting it, which is the same instinct that catches a function
returning negative durations.

I want to study Computer Science because every system I have built has failed in
a way I did not predict, and I would like to understand the principles well
enough to predict a few of them.
```

**Run the terrible one first, then the good one.** The score delta fires and the difference between them is the whole point: every claim in the second has something specific attached — eleven customers, forty users, a hardcoded worker, a timezone bug.

> Don't treat the second one as a template. What makes it work isn't the phrasing, it's that the details are real. A copied structure with swapped-in details scores worse than a plainer statement that's actually yours.

---

## One rule the agents can't break

Every agent's backstory ends with the same instruction: **criticise the writing, never the writer.**

"This sentence says nothing specific" — never "you are boring."

It's written to override everything else, including the brutality slider. At 10 Dr Vance is merciless about sentences and still can't touch the person. Maya is also forbidden from inventing facts — if a claim needs evidence she doesn't have, she writes `[ADD SPECIFIC EXAMPLE HERE]` rather than making up an achievement you never had.

---

## Concepts covered

| Concept | Where it appears |
|---|---|
| `Agent` — role, goal, backstory | The three characters |
| `Task` — description, expected_output | Each agent's job |
| `Crew` — assembling and running | One per agent, run in sequence |
| Passing context between agents | Maya reads Dr Vance's and BUZZ's output |
| Anchored scoring | The rubric that stops score inflation |
| Constraints in prompts | The fairness rule |
| Session state | Score tracking, run limits |
| Deployment | Streamlit Community Cloud |

---

## Files

| File | What it is |
|---|---|
| `Lab1_Judgement_Machine_CrewAI.ipynb` | The teaching notebook — build it from scratch in Colab |
| `app.py` | The Streamlit app |
| `requirements.txt` | Pinned dependencies |

---

## Running it yourself

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-proj-your-key-here"
```

Then:

```bash
streamlit run app.py
```

### Deploying

1. Upload to a **public** GitHub repo
2. **share.streamlit.io** → **Create app** → point at `app.py` on `main`
3. **In Advanced settings, set Python version to 3.12** ⚠️
4. Paste your key into the **Secrets** box in TOML format
5. Deploy

> **Python 3.12 is not optional.** CrewAI does not run on 3.13 or 3.14 — you get a wall of `pydantic` errors before the app even starts. You can't change the version after deploying, so getting it right first time saves you deleting the app and redoing it.

### A note on `async`

The notebook uses `await crew.kickoff_async()` because Colab has an event loop running. `app.py` uses plain `kickoff()` because Streamlit doesn't. Same library, different environment — this catches people out.

---

## Cost

Runs on `gpt-4o-mini`. One full review costs a fraction of a cent.

The app caps input length and limits runs per session, but a refresh resets the counter. If you deploy publicly, **set a monthly spend cap** at platform.openai.com → Settings → Limits. That's the guard that actually holds, because it lives at OpenAI rather than in code someone can route around.

---

## Built with

[CrewAI](https://docs.crewai.com) · [Streamlit](https://streamlit.io) · OpenAI `gpt-4o-mini`

Built as a teaching project for **AI for Teens** at atomcamp.
