import random
import itertools
import hashlib

# ---------------------------
# Expanded semantic space
# ---------------------------

CHARACTERS = [
    "a young woman", "an old man", "a traveler", "a student",
    "a detective", "a soldier", "a writer", "a stranger",
    "a scientist", "a child"
]

EMOTIONS = [
    "anxious", "calm", "confused", "determined", "uneasy",
    "curious", "fearful", "hopeful", "exhausted", "nostalgic",
    "focused", "disturbed"
]

SETTINGS = [
    "a quiet room", "a crowded station", "a dark forest",
    "a small village", "an abandoned building", "a rainy street",
    "a hospital corridor", "a library at night", "a distant city",
    "a mountain path", "a seaside town", "a dim hallway"
]

EVENTS = [
    ("received a letter", "something changed"),
    ("heard a noise", "uncertainty increased"),
    ("opened the door", "a hidden truth appeared"),
    ("read the message", "the situation shifted"),
    ("lost the map", "navigation became difficult"),
    ("met a stranger", "trust became uncertain"),
    ("found a note", "a mystery deepened"),
    ("saw the light", "hope returned"),
    ("entered the building", "tension increased"),
    ("answered the phone", "a decision became necessary"),
    ("left the room", "the silence felt heavier"),
    ("looked outside", "something was different"),
    ("waited too long", "opportunity faded"),
    ("remembered the past", "emotion resurfaced"),
    ("followed the path", "the destination changed"),
    ("ignored the warning", "consequences followed"),
    ("took the train", "distance increased"),
    ("opened the book", "knowledge expanded"),
    ("closed the door", "isolation increased"),
    ("turned back", "doubt intensified")
]

NARRATIVE_FRAMES = [
    "What happens when {char} {event}?",
    "Why does {char} feel {emotion} in {setting}?",
    "What does the situation suggest about {char}?",
    "How does {setting} influence the outcome?",
    "What changes after {char} {event}?",
    "What is implied by the event?",
    "Why is the character uncertain?",
    "What leads to the final outcome?"
]

ANS_FRAMES = [
    "When {char} {event}, {result}.",
    "{char} becomes {emotion} because {cause}.",
    "The situation suggests that {implication}.",
    "The environment of {setting} shapes the outcome by {effect}.",
    "After {char} {event}, {result}, changing the trajectory.",
]

# ---------------------------
# Paraphrase layer (important for diversity)
# ---------------------------

PREFIX_NOISE = [
    "", "In this moment, ", "At that point, ", "Suddenly, ",
    "Later, ", "Without warning, "
]

CAUSES = [
    "the circumstances were unclear",
    "previous events influenced perception",
    "external pressure increased",
    "internal doubt grew stronger",
    "memory affected interpretation"
]

IMPLICATIONS = [
    "something is being hidden",
    "the truth is not fully revealed",
    "the character is not fully aware",
    "events are connected in subtle ways",
    "there is an unresolved tension"
]


# ---------------------------
# Sample generator
# ---------------------------

def generate_sample():
    char = random.choice(CHARACTERS)
    emotion = random.choice(EMOTIONS)
    setting = random.choice(SETTINGS)
    event, result = random.choice(EVENTS)

    q_template = random.choice(NARRATIVE_FRAMES)
    a_template = random.choice(ANS_FRAMES)

    question = q_template.format(
        char=char,
        emotion=emotion,
        setting=setting,
        event=event
    )

    answer = a_template.format(
        char=char,
        emotion=emotion,
        setting=setting,
        event=event,
        result=result,
        cause=random.choice(CAUSES),
        implication=random.choice(IMPLICATIONS),
        effect=random.choice(CAUSES)
    )

    # stochastic paraphrasing layer
    prefix = random.choice(PREFIX_NOISE)

    question = prefix + question
    answer = prefix + answer

    return f"<qa> {question} </qa>\n<ans> {answer} </ans>"


# ---------------------------
# Dataset generator (100k+)
# ---------------------------

def generate_dataset(n=100000, output="bookcorpus_diverse.txt"):
    seen = set()

    with open(output, "w", encoding="utf-8") as f:
        i = 0
        while i < n:
            sample = generate_sample()

            # simple dedup (important!)
            h = hashlib.md5(sample.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            f.write(sample + "\n\n")

            i += 1
            if i % 10000 == 0:
                print(f"generated {i}")


if __name__ == "__main__":
    generate_dataset()