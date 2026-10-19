# Exact facts BabyBata must never get wrong.
#
# A language model stores word PATTERNS, not text, so it can blend similar
# strings (it once turned the email into "batbold9182745" by mixing in the
# GitHub handle). So in the training data we write a slot like {EMAIL}
# instead, the model learns WHEN to say it, and fill() swaps in the real value.
#
# Change a value here and every answer updates, no retraining needed.

FACTS = {
    "EMAIL": "batbold918273645@gmail.com",
    "GITHUB": "github.com/batbold9182",
    "LEETCODE": "leetcode.com/u/batbold9182",
    "FULL_NAME": "Batbold Samdan",
    "UNIVERSITY": "Vizja University Warsaw",
    "GPA": "4.0 out of 5",
}


def fill(text):
    """Model output -> what the visitor sees:  '{EMAIL}' -> 'batbold918273645@gmail.com'"""
    for slot, value in FACTS.items():
        text = text.replace("{" + slot + "}", value)
    return text


def to_slots(text):
    """The reverse, for saving training data:  'batbold918273645@gmail.com' -> '{EMAIL}'"""
    for slot, value in sorted(FACTS.items(), key=lambda kv: -len(kv[1])):  # longest first
        text = text.replace(value, "{" + slot + "}")
    return text
