from typing import assert_never, Literal


def get_fun_fact(option: str) -> str:
    assert option in ["A", "B", "C", "D"], "option must be A, B, C, or D"
    narrowed: Literal["A", "B", "C", "D"] = option  # type: ignore
    if narrowed == "A":
        return """The letter A actually began upside down - it evolved from an ancient Egyptian hieroglyph of an ox head, which looked like ∀. When the Phoenicians adapted it around 1800 BCE, they called it "aleph," meaning ox.
A is one of only two letters that commonly appears as both an article and a word in English ("a" and "A" grade).
"A" holds the distinction of being one of only five letters that can represent a syllable by itself in English (as in "a dog" or "a cat").
It's the third most commonly used letter in English, after E and T, appearing in about 8.2% of all written text.
A is the only letter that retains the same pronunciation in every major European language.
The musical note A has been standardized at 440 Hz since 1939, becoming the universal tuning standard for musicians worldwide.
A's lowercase form emerged during the Middle Ages when scribes developed a faster way to write the capital letter, resulting in our modern "a".
Type A blood was the first blood type discovered, identified by Karl Landsteiner in 1901.
In binary ASCII code, both uppercase and lowercase "a" begin with 1100, making them part of the same "family" in computer encoding.
The perfect grade "A" tradition began in the 1880s at Mount Holyoke College, where they first implemented letter grades.
The letter A appears in the name of every continent except Europe.
The @ symbol, which contains a lowercase "a", was first used in commercial invoices in the 16th century to represent "at the rate of".
The A-frame house design gets its name from the distinctive triangular shape that resembles a capital A.
The A battery size was never mass produced, making it a curious gap in the standard battery naming system.
Ancient Romans used the letter A (Absolvo) on voting tablets to indicate "not guilty" in trials.
The International Phonetic Alphabet (IPA) uses "a" as the symbol for the open front unrounded vowel, one of the most common sounds in human languages.
The letter A is the only letter that appears in every U.S. state name that contains more than one word (New Hampshire, New Jersey, etc.).
In Boolean algebra, A often represents the first variable in logical expressions, influencing early computer programming conventions.
The shape of capital A inspired the design of many ancient religious symbols, including the All-Seeing Eye pyramid symbol.
The letter A is the most common starting grade in the modern U.S. education system, with grade inflation leading to an "A" average at many universities."""
    elif narrowed == "B":
        return """The letter B evolved from the Egyptian hieroglyph for "house," which looked like a floor plan of a home with rooms.
B is one of the most stable letters in the alphabet - its shape has changed very little since its Phoenician origins around 1000 BCE.
In music, 'B' has different meanings across cultures - in German musical notation, 'B' actually refers to what English speakers call 'B-flat'.
The letter B is the only letter that can be completely removed from the English alphabet and still maintain comprehensible written communication.
B is one of the rarest letters in English, appearing in only about 1.5% of written text.
The word 'bee' (the insect) and the letter 'B' (the letter) are etymologically unrelated - their similar sound is purely coincidental.
B was originally pronounced like 'bay' in ancient Semitic languages.
In chemistry, B is the symbol for Boron, which was first isolated in 1808 by Sir Humphry Davy.
The blood type B was discovered in 1900, making it the second blood type identified after type A.
In academic grading, B originally stood for 'Better than average' when letter grades were first standardized.
The letter B appears in the name of every major world religion except Shinto.
Plan B contraception got its name because it was literally the second option after Plan A (regular contraception).
B batteries were once common but fell out of use in the 1920s, creating a gap in the standard battery naming system.
The B programming language, predecessor to C, was created at Bell Labs in 1969.
In chess notation, B stands for Bishop, adopted as the standard in the 1850s.
The B-2 Spirit stealth bomber's distinctive shape was partly inspired by the letter B when viewed from above.
B positive is the rarest blood type among African Americans, occurring in only about 2% of the population.
The letter B is the only letter that appears twice in the FBI's official name (Federal Bureau of Investigation).
The B-side of vinyl records originally contained the less popular songs, leading to the term 'B-list' for secondary options.
The phrase 'Plan B' first appeared in military terminology in the 1840s."""
    elif narrowed == "C":
        return """The letter C comes from the Greek letter Gamma (Γ), which was adapted by the Etruscans and eventually evolved into its current form.
C is unique because it can make two distinct sounds in English - the 'soft C' (/s/) and the 'hard C' (/k/).
Before the letter K was widely adopted in English, C was used exclusively for the /k/ sound.
The copyright symbol © was first used in 1909 and is one of the most recognized uses of the letter C globally.
C is the only letter that appears in the Roman numeral system (C represents 100).
In music, 'Middle C' is considered the central reference point on a piano keyboard.
The programming language C, developed in 1972, was named simply because it came after another language called B.
Vitamin C was the first vitamin to be artificially synthesized, achieved in 1934.
C-level executives get their designation from the word 'Chief' - CEO, CFO, CTO, etc.
The grade C was originally meant to represent 'Competent' in early grading systems.
The letter C is the only letter that can be used to spell its own name in English.
C++ was named to indicate it was an increment over C, using the increment operator from the C language.
The C-note slang for a $100 bill comes from the Roman numeral C.
The C-clef in music notation can be moved to different positions on the staff, unlike other clefs.
The term 'C-section' comes from the Latin 'caesarean', named after Julius Caesar.
C is the speed of light constant in physics, chosen by Albert Einstein himself.
The C drive designation for computers' primary hard drives began with CP/M operating systems in the 1970s.
C batteries were originally designed for early portable radios in the 1920s.
The C programming language influenced nearly every major programming language that came after it.
The letter C begins more English words than any other letter except S."""
    elif narrowed == "D":
        return """The letter D began as a pictograph of a door in ancient Semitic writing systems, similar to a triangle on its side.
D is the most common starting letter for words in the English dictionary after S.
The letter D is used in Roman numerals to represent 500, though it wasn't part of the original system.
In music, D is one of the most common keys for rock music due to its suitability for standard guitar tuning.
The 'D' in D-Day doesn't actually stand for anything - in military terminology, 'D' just designates the start date of an operation.
D is the only letter that forms a word when spelled backwards ('deed').
In academic grades, D was traditionally meant to stand for 'Deficient' but passing.
The vitamin D family was discovered in the early 1920s during research on rickets prevention.
D is used in 3D to represent dimensional, coined in the 1840s as an abbreviation for three-dimensional.
The D battery size was standardized in 1926, becoming one of the most common battery formats.
The D programming language was created as a reimagining of C++ in 2001.
D.C. (District of Columbia) was named after Christopher Columbus in 1791.
The D-pad on gaming controllers was first introduced by Nintendo in 1982.
The letter D is the most common grade given in high school physics classes nationwide.
The D-ring was invented for parachute harnesses in World War II.
DVD originally stood for Digital Video Disc, though later redefined as Digital Versatile Disc.
The D note is the highest string on a standard violin.
The term 'D-list celebrity' was coined by entertainment columnist James Bacon in the 1950s.
D is the Roman numeral for 500 because it's half of M (1000), represented by half of the M symbol.
The letter D appears in the chemical symbol for the heaviest naturally occurring element, Uranium (U)."""
    else:
        assert_never(narrowed)
