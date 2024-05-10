from spellchecker import SpellChecker


def calculate_misspelling_percentage(sentence):
    spell = SpellChecker()
    words = sentence.split()
    misspelled = spell.unknown(words)
    if len(words) == 0:
        return 0  # Avoid division by zero if the sentence is empty
    misspelling_percentage = (len(misspelled) / len(words)) * 100
    return misspelling_percentage


if __name__ == "__main__":
    sentence = "Thiss is an exampel of a sentence with speling errors."
    percentage = calculate_misspelling_percentage(sentence)
    print(f"Percentage of misspelled words: {percentage:.2f}%")
