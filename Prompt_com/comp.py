
import kenlm
kenlm_model = kenlm.Model('adl_model.klm')

file_path= 'adl_corpus.txt'
def load_corpus(file_path):
    """Load corpus from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

def kenlm_complete(prompt, max_words=10, corpus_file="adl_corpus.txt"):
    # Load corpus sentences from the file
    corpus_sentences = load_corpus(corpus_file)

    # Build a vocabulary from the corpus
    vocab = set()
    for sentence in corpus_sentences:
        for word in sentence.split():
            vocab.add(word)
    vocab = list(vocab)

    completion = prompt.strip()
    for _ in range(max_words):
        best_score = float('-inf')
        best_word = None

        # Evaluate candidates by appending each word from the vocabulary.
        for word in vocab:
            candidate = completion + ' ' + word
            score = kenlm_model.score(candidate, bos=False, eos=False)
            if score > best_score:
                best_score = score
                best_word = word

        if best_word is None:
            break
        # Avoid repeating words already present.
        if best_word in completion.split():
            break
        completion += ' ' + best_word

        # If punctuation is detected, stop early.
        if best_word.endswith(('.', '?', '!')):
            break
    return completion


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# Initialize GPT-2 fallback using DistilGPT2.
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
distilgpt_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
generator = pipeline("text-generation", model=distilgpt_model, tokenizer=tokenizer)

def complete_prompt(prompt):
    kenlm_result = kenlm_complete(prompt)
    # Score the generated text using KenLM.
    kenlm_score = kenlm_model.score(kenlm_result, bos=False, eos=False)

    # If KenLM didn't extend the prompt or produced a low-scoring output, use GPT-2 as fallback.
    if kenlm_result.strip() == prompt.strip() or kenlm_score < -10:
        gpt_result = generator(prompt, max_length=len(prompt.split()) + 20, do_sample=True, temperature=0.7)
        return gpt_result[0]['generated_text']
    else:
        return kenlm_result

test_prompt = "I want to eat "
completed_text = complete_prompt(test_prompt)
print("Completed Text:\n", completed_text)


