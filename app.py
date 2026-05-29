import re
from collections import Counter

import pandas as pd
import streamlit as st
from spellchecker import SpellChecker


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


def detect_spelling_errors(text: str, spell_checker, suggestion_limit: int = 5):
    words = tokenize_words(text)
    lowered_words = [w.lower() for w in words]
    misspelled = sorted(spell_checker.unknown(lowered_words))

    rows = []
    for word in misspelled:
        best = spell_checker.correction(word) or word
        candidates = spell_checker.candidates(word) or set()
        candidates = list(candidates)[:suggestion_limit]

        rows.append({
            "Error Type": "Spelling",
            "Detected": word,
            "Best Correction": best,
            "Suggestions": ", ".join(candidates),
            "Explanation": "The word is not recognized in the dictionary."
        })

    return rows


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


def make_corrected_text(text: str, spell_checker):
    corrected = text
    corrected = re.sub(r"\bi\b", "I", corrected)
    corrected = re.sub(r" {2,}", " ", corrected)
    corrected = re.sub(r"\s+([,.!?;:])", r"\1", corrected)
    corrected = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", corrected)

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

    def cap_sentence(match):
        sentence = match.group(0)
        m = re.search(r"[A-Za-z']+", sentence)
        if not m:
            return sentence
        start, end = m.span()
        word = sentence[start:end]
        return sentence[:start] + word[0].upper() + word[1:] + sentence[end:]

    corrected = re.sub(r"[^.!?\n]+[.!?]?", cap_sentence, corrected)
    return corrected


st.set_page_config(
    page_title="English Writing Analyzer for Young Learners",
    page_icon="📝",
    layout="wide"
)

st.title("📝 English Writing Analyzer for Young Learners")
st.write(
    "This app analyzes young learners' English writing by checking spelling, "
    "capitalization, word count, sentence count, TTR, and frequent words."
)

spell = SpellChecker()

with st.sidebar:
    st.header("Analysis Options")
    suggestion_limit = st.selectbox("Number of spelling suggestions", [1, 3, 5, 10], index=2)
    check_spelling = st.checkbox("Check spelling errors", value=True)
    check_capitalization = st.checkbox("Check capitalization errors", value=True)
    show_corrected_text = st.checkbox("Show corrected text preview", value=True)

st.subheader("1. Input Writing")

input_method = st.radio("Choose input method", ["Type or paste text", "Upload TXT file"])

text = ""

if input_method == "Type or paste text":
    text = st.text_area(
        "Paste student writing here",
        height=240,
        placeholder="Example: i have a freind. She is soooo kind.I like play with her."
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

        spelling_rows = detect_spelling_errors(text, spell, suggestion_limit) if check_spelling else []
        capitalization_rows = detect_capitalization_errors(text) if check_capitalization else []
        all_errors = spelling_rows + capitalization_rows

        st.subheader("2. Writing Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Word Count", word_count)
        col2.metric("Sentence Count", sentence_count)
        col3.metric("Types", type_count)
        col4.metric("TTR", f"{ttr:.3f}")
        col5.metric("Avg. Sentence Length", f"{avg_sentence_length:.2f}")

        st.metric("Total Detected Errors", len(all_errors))

        if show_corrected_text:
            st.subheader("3. Corrected Text Preview")
            corrected_text = make_corrected_text(text, spell)
            st.success(corrected_text)

            st.download_button(
                label="Download Corrected Text",
                data=corrected_text.encode("utf-8"),
                file_name="corrected_writing.txt",
                mime="text/plain"
            )

        st.subheader("4. Error Analysis")

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
            st.info("No spelling or capitalization errors were detected.")

        st.subheader("5. Top 10 Frequent Words")
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

        st.subheader("6. Summary CSV")
        summary_df = pd.DataFrame([{
            "Word Count": word_count,
            "Sentence Count": sentence_count,
            "Types": type_count,
            "TTR": round(ttr, 3),
            "Average Sentence Length": round(avg_sentence_length, 2),
            "Total Errors": len(all_errors),
            "Spelling Errors": len(spelling_rows),
            "Capitalization Errors": len(capitalization_rows)
        }])
        st.dataframe(summary_df, use_container_width=True)

        summary_csv = summary_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download Summary CSV",
            data=summary_csv,
            file_name="writing_summary.csv",
            mime="text/csv"
        )
