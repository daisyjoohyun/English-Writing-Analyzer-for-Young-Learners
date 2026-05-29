import re
from collections import Counter

import pandas as pd
import streamlit as st
from spellchecker import SpellChecker


# =========================================================
# Text processing
# =========================================================
def tokenize_words(text: str):
    return re.findall(r"[A-Za-z']+", text)


def get_sentences(text: str):
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_ttr(words):
    if not words:
        return 0
    lowered = [w.lower() for w in words]
    return len(set(lowered)) / len(lowered)


def preserve_case(original: str, suggestion: str):
    if original.isupper():
        return suggestion.upper()
    if original.istitle():
        return suggestion.title()
    return suggestion


# =========================================================
# Repeated-letter detection
# Example: soooo -> so, happpy -> happy
# =========================================================
def normalize_repeated_letters(word: str):
    lower = word.lower()
    one_repeat = re.sub(r"(.)\1{2,}", r"\1", lower)
    two_repeat = re.sub(r"(.)\1{2,}", r"\1\1", lower)
    return one_repeat, two_repeat


def detect_repeated_letter_errors(text: str, spell_checker, suggestion_limit: int = 5):
    rows = []
    repeated_spans = set()

    for match in re.finditer(r"[A-Za-z']+", text):
        word = match.group(0)

        if not re.search(r"(.)\1{2,}", word.lower()):
            continue

        one_repeat, two_repeat = normalize_repeated_letters(word)

        candidates = []
        if one_repeat in spell_checker:
            candidates.append(one_repeat)
        if two_repeat in spell_checker and two_repeat != one_repeat:
            candidates.append(two_repeat)

        if not candidates:
            guessed = spell_checker.correction(one_repeat)
            if guessed:
                candidates.append(guessed)

        best = candidates[0] if candidates else one_repeat
        best = preserve_case(word, best)

        rows.append({
            "Error Type": "Repeated Letters",
            "Detected": word,
            "Best Correction": best,
            "Suggestions": ", ".join([preserve_case(word, c) for c in candidates[:suggestion_limit]]),
            "Explanation": "Repeated letters were reduced before dictionary checking."
        })

        repeated_spans.add((match.start(), match.end()))

    return rows, repeated_spans


# =========================================================
# Spelling detection
# =========================================================
def detect_spelling_errors(text: str, spell_checker, suggestion_limit: int = 5, skip_spans=None):
    if skip_spans is None:
        skip_spans = set()

    words = []
    original_map = {}

    for match in re.finditer(r"[A-Za-z']+", text):
        if (match.start(), match.end()) in skip_spans:
            continue

        original = match.group(0)
        lower = original.lower()
        words.append(lower)
        original_map[lower] = original

    misspelled = sorted(spell_checker.unknown(words))

    rows = []
    for word in misspelled:
        best = spell_checker.correction(word) or word
        candidates = spell_checker.candidates(word) or set()
        candidates = list(candidates)[:suggestion_limit]
        original = original_map.get(word, word)

        rows.append({
            "Error Type": "Spelling",
            "Detected": original,
            "Best Correction": preserve_case(original, best),
            "Suggestions": ", ".join([preserve_case(original, c) for c in candidates]),
            "Explanation": "The word is not recognized in the dictionary."
        })

    return rows


# =========================================================
# Capitalization detection
# =========================================================
def detect_capitalization_errors(text: str):
    rows = []

    sentence_spans = re.finditer(r"[^.!?\n]+[.!?]?", text)
    for match in sentence_spans:
        sentence = match.group().strip()
        if not sentence:
            continue

        first_word = re.search(r"[A-Za-z']+", sentence)
        if first_word:
            word = first_word.group()
            if word[0].islower():
                rows.append({
                    "Error Type": "Capitalization",
                    "Detected": word,
                    "Best Correction": word[0].upper() + word[1:],
                    "Suggestions": word[0].upper() + word[1:],
                    "Explanation": "A sentence should begin with a capital letter."
                })

    for _ in re.finditer(r"\bi\b", text):
        rows.append({
            "Error Type": "Capitalization",
            "Detected": "i",
            "Best Correction": "I",
            "Suggestions": "I",
            "Explanation": "The pronoun 'I' should always be capitalized."
        })

    return rows


# =========================================================
# Missing punctuation detection
# =========================================================
def detect_missing_punctuation(text: str):
    rows = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # If the whole text has no ending punctuation
    if text.strip() and text.strip()[-1] not in ".!?":
        rows.append({
            "Error Type": "Missing Punctuation",
            "Detected": text.strip()[-25:],
            "Best Correction": text.strip() + ".",
            "Suggestions": "Add . / ! / ? at the end",
            "Explanation": "A complete sentence usually needs ending punctuation."
        })

    # Each non-empty line can also be checked
    for line in lines:
        if line and line[-1] not in ".!?":
            rows.append({
                "Error Type": "Missing Punctuation",
                "Detected": line[-25:],
                "Best Correction": line + ".",
                "Suggestions": "Add . / ! / ? at the end of the sentence",
                "Explanation": "This line appears to end without punctuation."
            })

    # Remove duplicate rows
    unique = []
    seen = set()
    for row in rows:
        key = (row["Error Type"], row["Detected"], row["Best Correction"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    return unique


# =========================================================
# Common grammar pattern detection
# =========================================================
def detect_common_grammar_patterns(text: str):
    rows = []

    grammar_patterns = [
        {
            "name": "enjoy + Ving",
            "pattern": r"\benjoy\s+([A-Za-z]+)\b",
            "condition": lambda v: not v.lower().endswith("ing"),
            "fix": lambda v: simple_ing(v),
            "message": "After 'enjoy', use a gerund (-ing form)."
        },
        {
            "name": "want + to V",
            "pattern": r"\bwant\s+([A-Za-z]+ing)\b",
            "condition": lambda v: True,
            "fix": lambda v: "to " + remove_ing(v),
            "message": "After 'want', use 'to + base verb'."
        },
        {
            "name": "listen to",
            "pattern": r"\blisten\s+(music|songs|song|radio|podcast|story|stories)\b",
            "condition": lambda v: True,
            "fix": lambda v: "to " + v,
            "message": "Use 'listen to + noun'."
        },
        {
            "name": "good at + Ving",
            "pattern": r"\bgood\s+at\s+([A-Za-z]+)\b",
            "condition": lambda v: not v.lower().endswith("ing"),
            "fix": lambda v: simple_ing(v),
            "message": "After 'good at', use a noun or a gerund (-ing form)."
        },
        {
            "name": "interested in + Ving",
            "pattern": r"\binterested\s+in\s+([A-Za-z]+)\b",
            "condition": lambda v: not v.lower().endswith("ing"),
            "fix": lambda v: simple_ing(v),
            "message": "After 'interested in', use a noun or a gerund (-ing form)."
        },
    ]

    for item in grammar_patterns:
        for match in re.finditer(item["pattern"], text, flags=re.IGNORECASE):
            verb_or_word = match.group(1)

            if not item["condition"](verb_or_word):
                continue

            fixed_part = item["fix"](verb_or_word)
            original_phrase = match.group(0)

            if item["name"] == "listen to":
                corrected_phrase = original_phrase.replace(verb_or_word, fixed_part, 1)
            elif item["name"] == "want + to V":
                corrected_phrase = original_phrase.replace(verb_or_word, fixed_part, 1)
            else:
                corrected_phrase = original_phrase.replace(verb_or_word, fixed_part, 1)

            rows.append({
                "Error Type": "Grammar Pattern",
                "Detected": original_phrase,
                "Best Correction": corrected_phrase,
                "Suggestions": item["name"],
                "Explanation": item["message"]
            })

    return rows


def simple_ing(verb: str):
    lower = verb.lower()

    irregular = {
        "run": "running",
        "swim": "swimming",
        "sit": "sitting",
        "begin": "beginning",
        "get": "getting",
        "shop": "shopping",
        "stop": "stopping",
        "write": "writing",
        "make": "making",
        "take": "taking",
        "dance": "dancing",
        "use": "using",
    }

    if lower in irregular:
        result = irregular[lower]
    elif lower.endswith("e") and lower not in ["see", "be"]:
        result = lower[:-1] + "ing"
    else:
        result = lower + "ing"

    return preserve_case(verb, result)


def remove_ing(word: str):
    lower = word.lower()

    irregular = {
        "going": "go",
        "doing": "do",
        "making": "make",
        "taking": "take",
        "writing": "write",
        "using": "use",
        "running": "run",
        "swimming": "swim",
        "sitting": "sit",
        "getting": "get",
        "shopping": "shop",
        "stopping": "stop",
    }

    if lower in irregular:
        result = irregular[lower]
    elif lower.endswith("ing"):
        result = lower[:-3]
    else:
        result = lower

    return preserve_case(word, result)


# =========================================================
# Corrected text
# =========================================================
def make_corrected_text(text: str, spell_checker):
    corrected = text

    # Repeated-letter correction first
    def fix_repeated(match):
        original = match.group(0)
        if not re.search(r"(.)\1{2,}", original.lower()):
            return original

        one_repeat, two_repeat = normalize_repeated_letters(original)

        if one_repeat in spell_checker:
            return preserve_case(original, one_repeat)
        if two_repeat in spell_checker:
            return preserve_case(original, two_repeat)

        guessed = spell_checker.correction(one_repeat)
        return preserve_case(original, guessed) if guessed else original

    corrected = re.sub(r"[A-Za-z']+", fix_repeated, corrected)

    # Basic spacing
    corrected = re.sub(r" {2,}", " ", corrected)
    corrected = re.sub(r"\s+([,.!?;:])", r"\1", corrected)
    corrected = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", corrected)

    # Common grammar patterns
    corrected = re.sub(
        r"\benjoy\s+([A-Za-z]+)\b",
        lambda m: m.group(0) if m.group(1).lower().endswith("ing") else m.group(0).replace(m.group(1), simple_ing(m.group(1)), 1),
        corrected,
        flags=re.IGNORECASE
    )
    corrected = re.sub(
        r"\bwant\s+([A-Za-z]+ing)\b",
        lambda m: m.group(0).replace(m.group(1), "to " + remove_ing(m.group(1)), 1),
        corrected,
        flags=re.IGNORECASE
    )
    corrected = re.sub(
        r"\blisten\s+(music|songs|song|radio|podcast|story|stories)\b",
        lambda m: m.group(0).replace(m.group(1), "to " + m.group(1), 1),
        corrected,
        flags=re.IGNORECASE
    )
    corrected = re.sub(
        r"\bgood\s+at\s+([A-Za-z]+)\b",
        lambda m: m.group(0) if m.group(1).lower().endswith("ing") else m.group(0).replace(m.group(1), simple_ing(m.group(1)), 1),
        corrected,
        flags=re.IGNORECASE
    )
    corrected = re.sub(
        r"\binterested\s+in\s+([A-Za-z]+)\b",
        lambda m: m.group(0) if m.group(1).lower().endswith("ing") else m.group(0).replace(m.group(1), simple_ing(m.group(1)), 1),
        corrected,
        flags=re.IGNORECASE
    )

    # Capitalization
    corrected = re.sub(r"\bi\b", "I", corrected)

    # General spelling correction after repeated letters and grammar patterns
    def replace_word(match):
        original = match.group(0)
        lower = original.lower()
        if lower in spell_checker:
            return original

        suggestion = spell_checker.correction(lower)
        if not suggestion:
            return original

        return preserve_case(original, suggestion)

    corrected = re.sub(r"[A-Za-z']+", replace_word, corrected)

    # Sentence-initial capitalization
    def cap_sentence(match):
        sentence = match.group(0)
        m = re.search(r"[A-Za-z']+", sentence)
        if not m:
            return sentence
        start, end = m.span()
        word = sentence[start:end]
        return sentence[:start] + word[0].upper() + word[1:] + sentence[end:]

    corrected = re.sub(r"[^.!?\n]+[.!?]?", cap_sentence, corrected)

    # Missing final punctuation
    if corrected.strip() and corrected.strip()[-1] not in ".!?":
        corrected = corrected.rstrip() + "."

    return corrected



# =========================================================
# Overall writing quality
# =========================================================
def evaluate_writing_quality(total_errors, word_count, ttr):
    """
    Simple classroom-friendly overall evaluation.
    This is not a final grade; it is a quick diagnostic label.
    """
    error_rate = (total_errors / word_count * 100) if word_count else 0

    # A simple diagnostic writing score.
    # Errors reduce the score, while reasonable lexical variety gives a small bonus.
    base_score = 100 - (error_rate * 1.2)
    ttr_bonus = 5 if ttr >= 0.50 else 0
    score = max(0, min(100, round(base_score + ttr_bonus)))

    if score >= 85:
        label = "Excellent"
        icon = "🏆"
        comment = "The writing is generally accurate and shows good lexical variety."
    elif score >= 70:
        label = "Good"
        icon = "👍"
        comment = "The writing is understandable, but some errors need revision."
    else:
        label = "Needs Improvement"
        icon = "⚠️"
        comment = "The writing needs more careful revision, especially in accuracy and basic patterns."

    return label, icon, comment, error_rate, score

# =========================================================
# Streamlit app
# =========================================================
st.set_page_config(
    page_title="English Writing Analyzer for Young Learners",
    page_icon="📝",
    layout="wide"
)

st.title("📝 English Writing Analyzer for Young Learners")
st.write(
    "This app analyzes young learners' English writing by checking spelling, capitalization, "
    "repeated letters, missing punctuation, common grammar patterns, word count, sentence count, TTR, and frequent words."
)

spell = SpellChecker()

with st.sidebar:
    st.header("Analysis Options")
    suggestion_limit = st.selectbox("Number of spelling suggestions", [1, 3, 5, 10], index=2)
    check_spelling = st.checkbox("Check spelling errors", value=True)
    check_repeated_letters = st.checkbox("Check repeated-letter errors", value=True)
    check_capitalization = st.checkbox("Check capitalization errors", value=True)
    check_missing_punctuation = st.checkbox("Check missing punctuation", value=True)
    check_grammar_patterns = st.checkbox("Check common grammar patterns", value=True)
    show_corrected_text = st.checkbox("Show corrected text preview", value=True)

st.subheader("1. Input Writing")

input_method = st.radio("Choose input method", ["Type or paste text", "Upload TXT file"])

text = ""

if input_method == "Type or paste text":
    text = st.text_area(
        "Paste student writing here",
        height=240,
        placeholder="Example: i enjoy play soccer. I want going home. I listen music. She is soooo kind"
    )
else:
    uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded text", text, height=240)

if st.button("Analyze Writing"):
    if not text.strip():
        st.warning("Please enter text or upload a TXT file first.")
    else:
        words = tokenize_words(text)
        lowered_words = [w.lower() for w in words]
        sentences = get_sentences(text)

        word_count = len(words)
        sentence_count = len(sentences)
        type_count = len(set(lowered_words))
        ttr = calculate_ttr(words)
        avg_sentence_length = word_count / sentence_count if sentence_count else 0

        repeated_rows, repeated_spans = detect_repeated_letter_errors(text, spell, suggestion_limit) if check_repeated_letters else ([], set())
        spelling_rows = detect_spelling_errors(text, spell, suggestion_limit, repeated_spans) if check_spelling else []
        capitalization_rows = detect_capitalization_errors(text) if check_capitalization else []
        punctuation_rows = detect_missing_punctuation(text) if check_missing_punctuation else []
        grammar_rows = detect_common_grammar_patterns(text) if check_grammar_patterns else []

        all_errors = spelling_rows + repeated_rows + capitalization_rows + punctuation_rows + grammar_rows

        st.subheader("2. Writing Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Word Count", word_count)
        col2.metric("Sentence Count", sentence_count)
        col3.metric("Types", type_count)
        col4.metric("TTR", f"{ttr:.3f}")
        col5.metric("Avg. Sentence Length", f"{avg_sentence_length:.2f}")

        st.metric("Total Detected Errors", len(all_errors))

        quality_label, quality_icon, quality_comment, error_rate, writing_score = evaluate_writing_quality(
            total_errors=len(all_errors),
            word_count=word_count,
            ttr=ttr
        )

        st.subheader("3. Overall Writing Quality")
        st.info(
            f"### {quality_icon} Overall Quality: {quality_label}\n"
            f"**Writing Score:** {writing_score} / 100  \n"
            f"**Error Rate:** {error_rate:.2f}%  \n\n"
            f"{quality_comment}"
        )

        error_type_counts = {
            "Spelling Errors": len(spelling_rows),
            "Repeated-Letter Errors": len(repeated_rows),
            "Capitalization Errors": len(capitalization_rows),
            "Missing Punctuation Errors": len(punctuation_rows),
            "Grammar Pattern Errors": len(grammar_rows),
        }

        st.subheader("4. Error Type Summary")
        summary_cols = st.columns(5)
        for col, (label, count) in zip(summary_cols, error_type_counts.items()):
            col.metric(label, count)

        if show_corrected_text:
            st.subheader("5. Corrected Text Preview")
            corrected_text = make_corrected_text(text, spell)
            st.success(corrected_text)

            st.download_button(
                label="Download Corrected Text",
                data=corrected_text.encode("utf-8"),
                file_name="corrected_writing.txt",
                mime="text/plain"
            )

        st.subheader("6. Error Analysis")

        if all_errors:
            error_df = pd.DataFrame(all_errors)
            st.dataframe(error_df, use_container_width=True)

            error_csv = error_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download Error Analysis CSV",
                data=error_csv,
                file_name="error_analysis.csv",
                mime="text/csv"
            )
        else:
            st.info("No selected error types were detected.")

        st.subheader("7. Top 10 Frequent Words")
        word_freq = Counter(lowered_words)
        freq_df = pd.DataFrame(word_freq.most_common(10), columns=["Word", "Frequency"])
        st.dataframe(freq_df, use_container_width=True)

        freq_csv = freq_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download Word Frequency CSV",
            data=freq_csv,
            file_name="word_frequency.csv",
            mime="text/csv"
        )

        st.subheader("8. Summary CSV")
        summary_df = pd.DataFrame([{
            "Word Count": word_count,
            "Sentence Count": sentence_count,
            "Types": type_count,
            "TTR": round(ttr, 3),
            "Average Sentence Length": round(avg_sentence_length, 2),
            "Total Errors": len(all_errors),
            "Error Rate (%)": round(error_rate, 2),
            "Writing Score": writing_score,
            "Overall Quality": quality_label,
            "Spelling Errors": len(spelling_rows),
            "Repeated-Letter Errors": len(repeated_rows),
            "Capitalization Errors": len(capitalization_rows),
            "Missing Punctuation Errors": len(punctuation_rows),
            "Grammar Pattern Errors": len(grammar_rows)
        }])
        st.dataframe(summary_df, use_container_width=True)

        summary_csv = summary_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download Summary CSV",
            data=summary_csv,
            file_name="writing_summary.csv",
            mime="text/csv"
        )
