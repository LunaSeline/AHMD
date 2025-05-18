import re
from spellchecker import SpellChecker
from huggingface_hub import notebook_login
from googletrans import Translator

# Optional: Login to HuggingFace (only if you need it)
# notebook_login()

# Initialize spell checker and translator
spell_checker = SpellChecker()
translator = Translator()

# Optional: Add custom spelling corrections
SPELLING_CORRECTIONS = {
    # Example: "teh": "the"
}

def clean_and_fix_text(text):
    # Fix stammering patterns like "S-so," -> "So," and "W-wait" -> "Wait"
    def fix_stammering(match):
        first = match.group(1)
        rest = match.group(2)
        punctuation = match.group(3) if match.group(3) else ''
        return first.upper() + rest + punctuation

    # Correct patterns like "S-so,", "W-wait." etc.
    text = re.sub(r'\b([A-Za-z])-([a-z]+)([.,!?]?)', fix_stammering, text)

    # Remove filler words
    text = re.sub(r'\b(um|uh|er|ah|eh|uhm|mm|hmm)\b', '', text, flags=re.IGNORECASE)

    # Clean up punctuation and spacing
    text = re.sub(r'\s*,\s*,', ',', text)
    text = re.sub(r'^\s*,', '', text)
    text = re.sub(r'\s*,', ',', text)
    text = re.sub(r'\s+', ' ', text)

    # Spell check each word
    words = text.split()
    corrected_words = []
    for word in words:
        lower_word = word.lower()
        if lower_word in SPELLING_CORRECTIONS:
            corrected = SPELLING_CORRECTIONS[lower_word]
            if word[0].isupper():
                corrected = corrected.capitalize()
            corrected_words.append(corrected)
        else:
            suggestion = spell_checker.correction(word)
            corrected_words.append(suggestion if suggestion else word)

    corrected_sentence = ' '.join(corrected_words)
    corrected_sentence = re.sub(r'\s+', ' ', corrected_sentence).strip()

    # Ensure proper punctuation and capitalization
    if corrected_sentence and corrected_sentence[-1] not in ".!?":
        corrected_sentence += "."
    if corrected_sentence:
        corrected_sentence = corrected_sentence[0].upper() + corrected_sentence[1:]

    return corrected_sentence

def process_uploaded_file(file_content, translate_to=None):
    lines = file_content.decode('utf-8').strip().split('\n')
    print("\nProcessed text:")
    for line in lines:
        if line.strip():
            cleaned = clean_and_fix_text(line.strip())
            if translate_to:
                try:
                    translated = translator.translate(cleaned, dest=translate_to).text
                    print(translated)
                except Exception as e:
                    print(f"Translation error: {e}")
                    print(cleaned)
            else:
                print(cleaned)


