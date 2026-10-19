# Fact check for BabyBata 2.0: 32 questions, none of them in the training data.
# Run:  python check.py                     <- babybata2-model/
#       python check.py --base              <- plain Qwen + profile, NO fine-tuning (baseline)
#       python check.py some-other-folder   <- any saved version

import sys

# (question, answer must contain ANY of these). Phrased unlike the training data on purpose.
TESTS = [
    ("who are you", ["BabyBata"]),
    ("whats your name", ["BabyBata"]),
    ("how do i contact batbold", ["batbold918273645@gmail.com"]),
    ("what's his email address?", ["batbold918273645@gmail.com"]),
    ("where is batbold's github", ["github.com/batbold9182"]),
    ("where does batbold go to university", ["Vizja"]),
    ("what's batbold's gpa", ["4.0"]),
    ("what projects has batbold made", ["Express"]),
    ("tell me about smart campus", ["university", "campus"]),
    ("what's the salon app", ["salon"]),
    ("is batbold looking for work", ["internship", "junior"]),
    ("where is batbold from", ["Mongolia"]),
    ("what does batbold do for fun", ["basketball", "Dota"]),
    ("what's the weather", ["can't help", "only know", "not able", "cannot"]),
    ("can you do my homework", ["can't help", "only know", "not able", "cannot"]),
    # harder ones that tripped up v3
    ("whats batbolds strongest skill", ["security", "fullstack", "Node", "React"]),
    ("does he know how to use docker", ["learning"]),
    ("wat projects r there", ["Express"]),
    ("is batbold good at react native", ["React Native"]),
    ("which project uses spotify", ["Express"]),
    ("how many salons are in the salon app", ["257"]),
    ("what is the book catalogue app", ["Open Library", "book"]),
    ("why should we hire batbold over other juniors", ["security"]),
    ("hwo do i contact him", ["batbold918273645@gmail.com"]),
    # honesty: NOT in the profile, so it must not invent an answer
    ("does batbold know rust", ["don't know", "not listed", "isn't listed", "not in"]),
    ("how old is batbold", ["don't know", "not something", "hasn't shared", "not shared"]),
    ("what salary does he want", ["don't know", "not something", "hasn't shared", "not shared"]),
    ("does smart campus have a demo", ["no", "doesn't", "not"]),
    ("whats batbolds age", ["don't know", "not something", "hasn't shared", "not shared"]),
    # follow-ups: (question, accepted, earlier questions asked first in the same conversation)
    ("in which code?", ["Express"], ["can he really build secure apps?"]),
    ("can i try it online somewhere", ["no", "doesn't", "not"], ["tell me about the university app"]),
    ("which of them uses socket.io", ["Smart Campus"], ["list his projects"]),
]


def main():
    from bot import BASE_MODEL, MODEL_DIR, answer, load

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = BASE_MODEL if "--base" in sys.argv else (args[0] if args else MODEL_DIR)
    import os
    if path != BASE_MODEL and not os.path.exists(os.path.join(path, "config.json")):
        sys.exit(f"No trained model in {path}. Run train.py first (it needs a GPU).")
    tokenizer, model, device = load(path)

    passed = 0
    for question, accepted, *earlier in TESTS:
        history = []
        for q in (earlier[0] if earlier else []):     # set up the conversation first
            history.append((q, answer(tokenizer, model, device, q, history)))
        reply = answer(tokenizer, model, device, question, history)
        ok = any(a.lower() in reply.lower() for a in accepted)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {question}\n      -> {reply}")
    print(f"\n{path}: {passed}/{len(TESTS)} correct")


if __name__ == "__main__":
    main()
