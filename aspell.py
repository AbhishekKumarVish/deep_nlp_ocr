import aspell

# Specify the path to the ASpell dictionary directory
aspell_dict_path = 'F:\longevity_task\aspell6-en-2020.12.07-0'

# Create an instance of the ASpell dictionary
spell_checker = aspell.Speller('lang', 'en', config_dir=aspell_dict_path)

# Perform spell checking on a word
word = 'wrogn'
if not spell_checker.check(word):
    suggestions = spell_checker.suggest(word)
    print(f"Misspelled word: {word}")
    print("Suggestions:", suggestions)
else:
    print("Correctly spelled word.")

# Perform spell checking on a sentence
sentence = "Ths is a sentence with errros."
words = sentence.split()

for word in words:
    if not spell_checker.check(word):
        suggestions = spell_checker.suggest(word)
        print(f"Misspelled word: {word}")
        print("Suggestions:", suggestions)
