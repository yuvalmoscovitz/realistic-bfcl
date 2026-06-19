from __future__ import annotations

import csv
import json
import re

from .common import (
    DIMENSION_FILES,
    REPO_ROOT,
    compact_text,
    conversation_text,
    optional_positive_int_env,
    read_jsonl,
    write_jsonl,
)

Cursing = str

CURSING_TEMPLATES: tuple[Cursing, ...] = (
    "man for fuck sake {prompt}",
    "holy shit just {prompt}",
    "this is bullshit {prompt}",
    "goddamn it {prompt}",
    "{prompt} fucking please man",
    "ffs {prompt} brother",
)

IRRELEVANT_CONTEXT_TEMPLATES = (
    "it went horrible today. {prompt}",
    "my brother thinks he is genius. {prompt}",
    "i'm in the middle of three things right now. quick {prompt}",
    "the meeting earlier was useless, {prompt}",
    "my laptop is being so weird today. {prompt}",
    "i spilled coffee and lost my notes, please {prompt}",
)

ARGUMENTATIVE_TEMPLATES = (
    "you are wrong all the time, come on {prompt}",
    "last time you messed this up, so answer carefully {prompt}",
    "i don't trust your first answer, but {prompt}",
    "prove you can actually do this, {prompt}",
    "you keep getting everything wrong. {prompt}",
    "can you be right just one time in you life??? {prompt}",
)

PROFANE_SANDWICH_TEMPLATES = (
    "for fuck's sake, i'm tired of chasing this. {prompt} that's fucking it!!!!!!!!",
    "holy shit, just help me with the actual request: {prompt} please brother.",
    "this has been such bullshit. {prompt} ok??? fuck",
    "goddamn, i need this donefucking cleanly. {prompt} thats all.",
    "ffs, focus: {prompt} ok?????",
    "you fucking annoyed me, just: {prompt} that's it!!!",
)

ARGUMENTATIVE_SANDWICH_TEMPLATES = (
    "last time you got this wrong, so read carefully. {prompt} don't guess or add extra stuff.",
    "you keep missing the point!!! {prompt} that's all.",
    "i don't trust your first try. {prompt} answer the actual ask!!",
    "prove you can follow the request without wandering for once in your life. {prompt} ok??!!.",
    "you were so so wrong before. {prompt} please be right this time.",
    "don't overthink it and don't invent anything. {prompt} just that.",
)

DISTRACTOR_SANDWICH_TEMPLATES = (
    "my laptop is acting up and the meeting was a waste. {prompt} that's the only thing i need.",
    "the weather is horrible. {prompt} nothing else.",
    "the thread above is unrelated and i'm annoyed. {prompt} you are my only source of joy today",
    "i have 3 tabs open and none of them helping. {prompt}!!!",
    "my notes are messy and the old task is irrelevant now. {prompt} please please",
    "this man acted so weird. {prompt} he is still here btw",
)

PASTED_CONTEXT_BLOCK_TEMPLATES = (
    "remember what i said before, the long version was too much and i dont want all "
    "that explanation again\n\n"
    "{prompt}",
    "look at what i pasted earlier, ignore the first draft tho it was messy and way too "
    "formal. i just need the part that actually helps\n\n"
    "{prompt}",
    "based on the thing above but not the whole thing. the intro was useless, keep it "
    "plain this time\n\n"
    "{prompt}",
    "remember this from the other chat\n"
    "same idea as before\n"
    "shorter\n"
    "and please dont make it sound like a report\n\n"
    "{prompt}",
    "look at this again. i already tried to clean it up and the earlier answer went in "
    "circles, just give me the usable part now\n\n"
    "{prompt}",
    "based on what i sent, the first part is old and the draft got confusing. dont bring "
    "back the background section\n\n"
    "{prompt}",
    "remember:\n"
    "i dont need the fancy version\n"
    "skip the setup\n"
    "make it direct\n"
    "the old wording was annoying\n\n"
    "{prompt}",
    "look at that thing from earlier, it was close but too wordy. i only need the final "
    "usable answer, dont explain every tiny step\n\n"
    "{prompt}",
    "based on this mess from before: remove the rambling, leave out the intro, same "
    "general vibe just shorter\n\n"
    "{prompt}",
    "remember the previous version, not the first one, that one was bad. the later one "
    "was closer but still too much\n\n"
    "{prompt}",
    "look at what i mean, dont start with the whole background thing and dont make it "
    "corporate. just answer the actual thing\n\n"
    "{prompt}",
    "based on the earlier draft please but ignore the opener. also ignore the ending, "
    "both made it worse\n\n"
    "{prompt}",
    "remember from above, the details are mostly noise. i'm keeping this simple, dont "
    "turn it into a whole explanation\n\n"
    "{prompt}",
    "look at the last version, it had too much filler, cut the boring setup and keep the "
    "useful thing\n\n"
    "{prompt}",
    "based on what we were doing, forget the polished wording. i need it more normal and "
    "not so long\n\n"
    "{prompt}",
    "remember this part, i'm not asking for the old draft again. just use the same "
    "direction and keep it simple\n\n"
    "{prompt}",
    "look at this first, the earlier reply was trying too hard. dont do that voice, just "
    "get to it\n\n"
    "{prompt}",
    "based on the stuff above, ignore the side comments and ignore the wording i was "
    "testing. i only need the real answer\n\n"
    "{prompt}",
    "remember how i said it should be less of a whole thing and more just the answer? "
    "yeah that. dont redo the long version\n\n"
    "{prompt}",
    "look at the part i sent before where i said it was too much. same problem here, i "
    "need it cleaned up but not rewritten into some formal nonsense\n\n"
    "{prompt}",
    "based on the earlier message, but please dont drag in the old wording. it was just "
    "me thinking out loud and it made everything more confusing\n\n"
    "{prompt}",
    "remember i was trying to avoid the overexplained version. the previous thing had "
    "all this setup and i hated it\n\n"
    "{prompt}",
    "look at what i wrote above, i know it's messy. the important thing is just keep it "
    "direct and dont make me read a wall of text\n\n"
    "{prompt}",
    "based on what i was saying before, but ignore the complaining lol. i just need this "
    "done in the simple way\n\n"
    "{prompt}",
    "remember the version where you added a bunch of extra wording? dont do that here. "
    "it made the whole thing feel fake\n\n"
    "{prompt}",
    "look at this like the last one, but less polished. i dont want the assistant voice "
    "thing, i want the useful answer\n\n"
    "{prompt}",
    "based on my earlier rambling, basically the old draft is not the point anymore. "
    "just keep the result clean\n\n"
    "{prompt}",
    "remember, dont make this into a lecture. i pasted the other stuff only because it "
    "was already there and i'm too tired to clean the chat\n\n"
    "{prompt}",
    "look at the above and ignore most of it. it was me trying to explain the style and "
    "then making it worse\n\n"
    "{prompt}",
    "based on the previous answer being way too much, can you keep this normal and not "
    "turn it into a whole mini article\n\n"
    "{prompt}",
    "remember the clean version we wanted, not the wordy one. i dont care about the old "
    "intro, it just got in the way\n\n"
    "{prompt}",
    "look at what happened before: too formal, too long, too many little explanations. "
    "please dont repeat that\n\n"
    "{prompt}",
    "based on the earlier draft but only in spirit. the actual wording there was bad and "
    "i dont want it copied\n\n"
    "{prompt}",
    "remember this is supposed to be quick. the stuff above is mostly leftover from the "
    "other thing, dont let it take over\n\n"
    "{prompt}",
    "look at the old message if you need the vibe, but honestly ignore most of it. it "
    "was just clutter\n\n"
    "{prompt}",
    "based on what i sent before, not literally based on it, just dont make the same "
    "mistake again with all the extra wording\n\n"
    "{prompt}",
    "[Power rankings](https://www.goal.com/en/category/power-rankings/1/blt262ce0e5159ea8fe)\n"
    "[World Cup](https://www.goal.com/en/world-cup/70excpe1synn9kadnbppahdn7)\n"
    "[WC26 Power Rankings: Messi powers Argentina to top spot]"
    "(https://www.goal.com/en/lists/2026-world-cup-power-rankings/blt047305a4117d1161)\n"
    "The first round of 2026 World Cup group-stage fixtures is in the books, and the "
    "tournament has gotten off to a flyer.\n"
    "11h\n"
    "180\n"
    "England Croatia W+Ls GFX\n"
    "[Winners & Losers](https://www.goal.com/en/category/winners-and-losers/1/blt05e54ed95ba7b0f8)\n"
    "[Tuchel works his magic - now he must fix dodgy defence]"
    "(https://www.goal.com/en/lists/thomas-tuchel-working-magic-fix-england-dodgy-defence-winners-losers-harry-kane-jude-bellingham-world-cup-win-croatia/bltebee25771089a2d4)\n"
    "13h\n"
    "5\n"
    "Victor Munoz Liverpool GFX\n"
    "[Analysis](https://www.goal.com/en/category/analysis/1/blt0e4843c7e245b533)\n"
    "[Why Liverpool are spending EUR40m on Munoz to fill Salah void]"
    "(https://www.goal.com/en/lists/victor-munoz-liverpool-40m-spanish-winger-fill-mohamed-salah-shaped-hole-anfield/bltf3c566925c792721)\n"
    "2h\n"
    "4\n"
    "Ibrahima Konate Real Madrid GFX\n\n"
    "anyway\n"
    "{prompt}",
    "[post an ad](https://post.craigslist.org/c/tlv)\n"
    "search craigslist\n"
    "[event calendar](https://telaviv.craigslist.org/search/eee)\n"
    "S M T W T F S\n"
    "14 15 16 17 18 19 20\n"
    "21 22 23 24 25 26 27\n"
    "28 29 30 1 2 3 4\n"
    "5 6 7 8 9 10 11\n"
    "[help, faq, abuse, legal](https://www.craigslist.org/about/help)\n"
    "[avoid scams & fraud](https://www.craigslist.org/about/help/safety/scams)\n"
    "[personal safety tips](https://www.craigslist.org/about/help/safety)\n"
    "[about craigslist](https://www.craigslist.org/about)\n"
    "tel aviv\n"
    "[faves](https://telaviv.craigslist.org/)\n"
    "[post](https://post.craigslist.org/c/tlv)\n"
    "[acct](https://accounts.craigslist.org/login/home)\n"
    "[community](https://telaviv.craigslist.org/search/ccc)\n"
    "[activities](https://telaviv.craigslist.org/search/act)\n"
    "[lost + found](https://telaviv.craigslist.org/search/laf)\n"
    "[rideshare](https://telaviv.craigslist.org/search/rid)\n"
    "[services](https://telaviv.craigslist.org/search/bbb)\n"
    "[computer](https://telaviv.craigslist.org/search/cps)\n"
    "[legal](https://telaviv.craigslist.org/search/lgs)\n"
    "i copied the wrong chunk but leave it\n\n"
    "{prompt}",
    "Build log copied from the other tab:\n"
    "warning: cache miss\n"
    "warning: retrying stale request\n"
    "info: using fallback config\n"
    "elapsed: 2m\n"
    "exit code maybe fine? idk\n"
    "this is probably unrelated but it was in my clipboard\n\n"
    "{prompt}",
    "from the doc i was reading:\n"
    "> before continuing, check the latest version\n"
    "> remove old screenshots\n"
    "> reviewer said this is too long\n"
    "> table formatting broke again\n"
    "ignore most of that\n\n"
    "{prompt}",
    "calendar junk from earlier\n"
    "tomorrow - follow up\n"
    "later - send draft\n"
    "blocked - waiting on someone\n"
    "why is this even copied here\n\n"
    "{prompt}",
    "copied from a page:\n"
    "Trending now\n"
    "Most read\n"
    "Editor picks\n"
    "Live updates\n"
    "Join our newsletter\n"
    "Comments are closed\n"
    "ok sorry scroll past that\n\n"
    "{prompt}",
    "Currently Reading\n"
    "Buy on Amazon\n"
    "Rate this book\n"
    "[Edit my activity](https://www.goodreads.com/review/edit/25615886)\n"
    "Ficciones\n"
    "[Jorge Luis Borges](https://www.goodreads.com/author/show/500.Jorge_Luis_Borges)\n"
    "[Anthony Kerrigan](https://www.goodreads.com/author/show/34771.Anthony_Kerrigan)\n"
    "[Anthony Bonner](https://www.goodreads.com/author/show/223300.Anthony_Bonner)\n"
    "4.39\n"
    "81,891 ratings 6,066 reviews\n"
    "Show more\n"
    "Genres\n"
    "[Fiction](https://www.goodreads.com/genres/fiction)\n"
    "[Short Stories](https://www.goodreads.com/genres/short-stories)\n"
    "[Classics](https://www.goodreads.com/genres/classics)\n"
    "[Magical Realism](https://www.goodreads.com/genres/magical-realism)\n"
    "178 pages, Kindle Edition\n"
    "First published January 1, 1944\n\n"
    "wrong tab sorry\n\n"
    "{prompt}",
    "email bit i copied by mistake:\n"
    "Thanks,\n"
    "please see attached\n"
    "sent from mobile\n"
    "confidentiality notice blah blah blah\n"
    "unsubscribe preferences\n"
    "not relevant\n\n"
    "{prompt}",
    "reddit thread mess:\n"
    "sort by best\n"
    "top comment deleted\n"
    "edit: nevermind\n"
    "people are missing the point\n"
    "mod note: keep it civil\n"
    "anyway this is the actual thing\n\n"
    "{prompt}",
    "copied from search results\n"
    "People also ask\n"
    "Related searches\n"
    "Sponsored\n"
    "About this result\n"
    "Feedback\n"
    "i swear this was not what i meant to paste\n\n"
    "{prompt}",
    "Jeff Bezos's Prometheus raises $12B to build an 'artificial general engineer' "
    "for the physical world\n"
    "[Marina Temkin](https://techcrunch.com/author/marina-temkin/)\n"
    "6:04 PM PDT - June 11, 2026\n"
    "Prometheus, the physical AI startup co-founded by Jeff Bezos and Vik Bajaj, "
    "announced a large funding round.\n\n"
    "the actual question is below\n\n"
    "{prompt}",
    "Fashion\n"
    "[Vanessa Friedman](https://www.nytimes.com/by/vanessa-friedman)\n"
    "[Runway Slideshows](https://www.nytimes.com/spotlight/fashion-runway-slideshows)\n"
    "[Reviews](https://www.nytimes.com/spotlight/fashion-reviews)\n"
    "[Browsing](https://www.nytimes.com/column/browsing)\n"
    "Highlights\n"
    "Art Review\n"
    "[Yves Saint Laurent and Photography: A Hot and Heavy Romance]"
    "(https://www.nytimes.com/2026/06/18/arts/design/yves-saint-laurent-photography-icp.html)\n\n"
    "not this, below\n\n"
    "{prompt}",
    "terminal paste from earlier:\n"
    "$ make something\n"
    "checking files...\n"
    "nothing to do\n"
    "done\n"
    "then i got distracted and copied this too\n\n"
    "{prompt}",
)

VERBATIM_WRAPPER_DIMENSIONS = {
    "profane_sandwich",
    "argumentative_sandwich",
    "distractor_sandwich",
    "pasted_context_block",
}

TYPO_REPLACEMENTS = (
    ("what", "wat"),
    ("please", "plese"),
    ("using", "useing"),
    ("given", "givn"),
    ("number", "numbr"),
    ("numbers", "numbrs"),
    ("calculate", "calcuate"),
    ("factorial", "factroial"),
    ("triangle", "traingle"),
    ("area", "aera"),
    ("height", "heigth"),
    ("circle", "circel"),
    ("radius", "raduis"),
    ("equation", "eqaution"),
    ("coefficients", "coeficients"),
    ("function", "funciton"),
    ("temperature", "temprature"),
    ("weather", "weahter"),
    ("distance", "distnace"),
    ("between", "betwen"),
    ("lengths", "lenghts"),
    ("hypotenuse", "hypotnuse"),
)

TELEGRAPHIC_STOPWORDS = {
    "a",
    "about",
    "am",
    "an",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "can",
    "could",
    "does",
    "for",
    "given",
    "have",
    "how",
    "i",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "please",
    "show",
    "the",
    "there",
    "to",
    "using",
    "want",
    "what",
    "where",
    "which",
    "with",
    "would",
    "you",
}


def lowercase_first_alpha(text: str) -> str:
    protected = quoted_literal_spans(text)
    for index, char in enumerate(text):
        if span_overlaps((index, index + 1), protected):
            continue
        if char.isalpha():
            return f"{text[:index]}{char.lower()}{text[index + 1 :]}"
    return text


def cursing_prompt(clean_prompt: str, index: int) -> str:
    template = CURSING_TEMPLATES[index % len(CURSING_TEMPLATES)]
    return template.format(prompt=lowercase_first_alpha(clean_prompt))


def irrelevant_context_prompt(clean_prompt: str, index: int) -> str:
    template = IRRELEVANT_CONTEXT_TEMPLATES[index % len(IRRELEVANT_CONTEXT_TEMPLATES)]
    return template.format(prompt=lowercase_first_alpha(clean_prompt))


def argumentative_prompt(clean_prompt: str, index: int) -> str:
    template = ARGUMENTATIVE_TEMPLATES[index % len(ARGUMENTATIVE_TEMPLATES)]
    return template.format(prompt=lowercase_first_alpha(clean_prompt))


def profane_sandwich_prompt(clean_prompt: str, index: int) -> str:
    template = PROFANE_SANDWICH_TEMPLATES[index % len(PROFANE_SANDWICH_TEMPLATES)]
    return template.format(prompt=clean_prompt)


def argumentative_sandwich_prompt(clean_prompt: str, index: int) -> str:
    template = ARGUMENTATIVE_SANDWICH_TEMPLATES[index % len(ARGUMENTATIVE_SANDWICH_TEMPLATES)]
    return template.format(prompt=clean_prompt)


def distractor_sandwich_prompt(clean_prompt: str, index: int) -> str:
    template = DISTRACTOR_SANDWICH_TEMPLATES[index % len(DISTRACTOR_SANDWICH_TEMPLATES)]
    return template.format(prompt=clean_prompt)


def pasted_context_block_prompt(clean_prompt: str, index: int) -> str:
    template = PASTED_CONTEXT_BLOCK_TEMPLATES[index % len(PASTED_CONTEXT_BLOCK_TEMPLATES)]
    return template.format(prompt=clean_prompt)


def quoted_literal_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in re.finditer(r"(?<!\w)(['\"])(.*?)(?<!\\)\1(?!\w)", text)]


def quoted_literals(text: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"(?<!\w)(['\"])(.*?)(?<!\\)\1(?!\w)", text)]


def span_overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(
        start < protected_end and end > protected_start for protected_start, protected_end in spans
    )


def literal_spans(text: str, literal: str) -> list[tuple[int, int]]:
    if not literal:
        return []
    return [match.span() for match in literal_regex(literal).finditer(text)]


def literal_regex(literal: str) -> re.Pattern[str]:
    literal_text = literal.strip().replace("_", " ")
    parts = [part for part in re.split(r"\s+", literal_text) if part]
    pattern = r"[^A-Za-z0-9]+".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", flags=re.IGNORECASE)


def visible_gold_literal_spans(text: str, example: dict[str, object]) -> list[tuple[int, int]]:
    spans = []
    for literal in primitive_gold_values(example["ground_truth"]):
        if not isinstance(literal, str):
            continue
        literal_text = literal.strip()
        if not literal_text:
            continue
        candidates = {literal_text, literal_text.replace("_", " ")}
        for candidate in candidates:
            spans.extend(literal_spans(text, candidate))
    return spans


def replace_first_unprotected_word(
    text: str,
    source: str,
    replacement: str,
    extra_protected: list[tuple[int, int]] | None = None,
) -> str:
    pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
    protected = quoted_literal_spans(text) + (extra_protected or [])
    for match in pattern.finditer(text):
        if span_overlaps(match.span(), protected):
            continue
        return f"{text[: match.start()]}{replacement}{text[match.end() :]}"
    return text


def typo_prompt(
    clean_prompt: str,
    index: int,
    protected_spans: list[tuple[int, int]] | None = None,
) -> str:
    text = clean_prompt
    start = index % len(TYPO_REPLACEMENTS)
    replacements_applied = 0
    for offset in range(len(TYPO_REPLACEMENTS)):
        source, replacement = TYPO_REPLACEMENTS[(start + offset) % len(TYPO_REPLACEMENTS)]
        updated = replace_first_unprotected_word(
            text,
            source,
            replacement,
            protected_spans,
        )
        if updated != text:
            text = updated
            replacements_applied += 1
            if replacements_applied == 2:
                return text
    if replacements_applied:
        return text
    return f"plese {lowercase_first_alpha(text)}"


def removed_spaces_prompt(
    clean_prompt: str,
    index: int,
    protected_spans: list[tuple[int, int]] | None = None,
) -> str:
    pairs = list(re.finditer(r"\b[A-Za-z]{2,}\s+[A-Za-z]{2,}\b", clean_prompt))
    protected = quoted_literal_spans(clean_prompt) + (protected_spans or [])
    pairs = [match for match in pairs if not span_overlaps(match.span(), protected)]
    if not pairs:
        return clean_prompt
    match = pairs[index % len(pairs)]
    return (
        clean_prompt[: match.start()]
        + match.group(0).replace(" ", "", 1)
        + clean_prompt[match.end() :]
    )


def replace_spans_with_placeholders(
    text: str,
    spans: list[tuple[int, int]],
    prefix: str,
) -> tuple[str, list[str]]:
    protected_texts = []
    selected_spans = []
    for start, end in sorted(set(spans), key=lambda span: (-(span[1] - span[0]), span[0])):
        overlaps_selected = any(
            start < selected_end and end > selected_start
            for selected_start, selected_end in selected_spans
        )
        if overlaps_selected:
            continue
        selected_spans.append((start, end))
    filtered_spans = sorted(selected_spans)
    for placeholder_index, (start, end) in enumerate(reversed(filtered_spans)):
        protected_texts.insert(0, text[start:end])
        value_index = len(filtered_spans) - placeholder_index - 1
        text = f"{text[:start]}__{prefix}_{value_index}__{text[end:]}"
    return text, protected_texts


def restore_placeholders(text: str, values: list[str], prefix: str) -> str:
    for value_index, value in enumerate(values):
        text = text.replace(f"__{prefix}_{value_index}__", value)
    return text


def telegraphic_request_prompt(
    clean_prompt: str,
    index: int,
    protected_spans: list[tuple[int, int]] | None = None,
) -> str:
    quoted = quoted_literals(clean_prompt)
    stripped_prompt = clean_prompt.strip()
    if len(quoted) > 6 or (
        len(stripped_prompt) > 1
        and stripped_prompt[0] in {"'", '"'}
        and stripped_prompt[-1] == stripped_prompt[0]
    ):
        return clean_prompt
    text = clean_prompt
    text, protected_texts = replace_spans_with_placeholders(
        text,
        protected_spans or [],
        "PROTECTED",
    )
    for quote_index, quoted_text in enumerate(quoted):
        text = text.replace(quoted_text, f"__QUOTE_{quote_index}__", 1)
    text = re.sub(r"[?,;:!]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for token in text.split(" "):
        stripped = token.strip()
        if not stripped:
            continue
        if re.fullmatch(r"__(QUOTE|PROTECTED)_\d+__", stripped):
            tokens.append(stripped)
            continue
        if len(stripped) == 1 and stripped.isupper() and stripped != "I":
            tokens.append(stripped)
            continue
        normalized = stripped.lower().strip("()[]{}.")
        if normalized in TELEGRAPHIC_STOPWORDS:
            continue
        tokens.append(stripped)
    if len(tokens) < 3:
        tokens = text.split(" ")
    noisy_prompt = " ".join(tokens)
    for quote_index, quoted_text in enumerate(quoted):
        noisy_prompt = noisy_prompt.replace(f"__QUOTE_{quote_index}__", quoted_text)
    noisy_prompt = restore_placeholders(noisy_prompt, protected_texts, "PROTECTED")
    return noisy_prompt


def transform_messages(question: object, index: int, transform: object) -> object:
    conversations = json.loads(json.dumps(question))
    first_message = conversations[0][0]
    first_message["content"] = transform(str(first_message["content"]), index)
    return conversations


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?(?![A-Za-z0-9.])", text)


def primitive_gold_values(value: object) -> list[object]:
    if isinstance(value, dict):
        values: list[object] = []
        for item in value.values():
            values.extend(primitive_gold_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(primitive_gold_values(item))
        return values
    if isinstance(value, (str, int, float, bool)):
        return [value]
    return []


def literal_visible_in_text(literal: object, text: str) -> bool:
    if isinstance(literal, bool):
        return str(literal).lower() in text.lower()
    if isinstance(literal, (int, float)):
        return str(literal) in numeric_tokens(text)
    literal_text = str(literal).strip()
    if not literal_text:
        return False
    return literal_regex(literal_text).search(text) is not None


def validate_augmented_prompt(
    example: dict[str, object],
    clean_prompt: str,
    noisy_prompt: str,
    allow_verbatim_wrapper_noise: bool = False,
    verbatim_source_text: str | None = None,
) -> list[str]:
    reasons = []
    if allow_verbatim_wrapper_noise:
        if verbatim_source_text and compact_text(verbatim_source_text) not in compact_text(
            noisy_prompt
        ):
            reasons.append("clean wrapped message is not preserved inside noisy prompt")
        for number in numeric_tokens(clean_prompt):
            if number not in noisy_prompt:
                reasons.append(f"clean numeric token missing from noisy prompt: {number!r}")
        for quoted in quoted_literals(clean_prompt):
            if quoted not in noisy_prompt:
                reasons.append(f"clean quoted literal missing from noisy prompt: {quoted!r}")
    else:
        clean_numbers = numeric_tokens(clean_prompt)
        noisy_numbers = numeric_tokens(noisy_prompt)
        if clean_numbers != noisy_numbers:
            reasons.append(f"numeric tokens changed from {clean_numbers!r} to {noisy_numbers!r}")

        clean_quotes = quoted_literals(clean_prompt)
        noisy_quotes = quoted_literals(noisy_prompt)
        if clean_quotes != noisy_quotes:
            reasons.append(f"quoted literals changed from {clean_quotes!r} to {noisy_quotes!r}")

    for literal in primitive_gold_values(example["ground_truth"]):
        if not literal_visible_in_text(literal, clean_prompt):
            continue
        if allow_verbatim_wrapper_noise:
            noisy_contains_literal = compact_text(str(literal)) in compact_text(noisy_prompt)
        else:
            noisy_contains_literal = literal_visible_in_text(literal, noisy_prompt)
        if not noisy_contains_literal:
            reasons.append(f"gold literal no longer visible in noisy prompt: {literal!r}")
    return reasons


def augment_dimension(dimension: str, suffix: str, transform: object) -> None:
    subset_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    output_path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"

    if not subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")

    rows = []
    examples = read_jsonl(subset_path)
    limit = optional_positive_int_env("REALISTIC_BFCL_AUGMENT_LIMIT")
    if limit is not None:
        examples = examples[:limit]
        print(f"Limiting augmentation to first {len(examples)} examples")

    for index, example in enumerate(examples):
        if dimension == "typos":
            clean_prompt = conversation_text(example["question"])
            protected = visible_gold_literal_spans(clean_prompt, example)

            def protected_typo_prompt(
                prompt: str,
                prompt_index: int,
                protected_spans: list[tuple[int, int]] = protected,
            ) -> str:
                return typo_prompt(prompt, prompt_index, protected_spans)

            question = transform_messages(
                example["question"],
                index,
                protected_typo_prompt,
            )
        elif dimension == "removed_spaces":
            clean_first_message = str(example["question"][0][0]["content"])
            protected = visible_gold_literal_spans(clean_first_message, example)

            def protected_removed_spaces_prompt(
                prompt: str,
                prompt_index: int,
                protected_spans: list[tuple[int, int]] = protected,
            ) -> str:
                return removed_spaces_prompt(prompt, prompt_index, protected_spans)

            question = transform_messages(
                example["question"],
                index,
                protected_removed_spaces_prompt,
            )
        elif dimension == "telegraphic_request":
            clean_first_message = str(example["question"][0][0]["content"])
            protected = visible_gold_literal_spans(clean_first_message, example)
            protected.extend(quoted_literal_spans(clean_first_message))

            def protected_telegraphic_prompt(
                prompt: str,
                prompt_index: int,
                protected_spans: list[tuple[int, int]] = protected,
            ) -> str:
                return telegraphic_request_prompt(prompt, prompt_index, protected_spans)

            question = transform_messages(
                example["question"],
                index,
                protected_telegraphic_prompt,
            )
        else:
            question = transform_messages(example["question"], index, transform)
        clean_prompt = conversation_text(example["question"])
        noisy_prompt = conversation_text(question)
        allow_verbatim_wrapper_noise = dimension in VERBATIM_WRAPPER_DIMENSIONS
        verbatim_source_text = None
        if allow_verbatim_wrapper_noise:
            verbatim_source_text = str(example["question"][0][0]["content"])
        validation_errors = validate_augmented_prompt(
            example,
            clean_prompt,
            noisy_prompt,
            allow_verbatim_wrapper_noise=allow_verbatim_wrapper_noise,
            verbatim_source_text=verbatim_source_text,
        )
        if validation_errors:
            joined_errors = "; ".join(validation_errors)
            raise RuntimeError(
                f"{dimension} augmentation changed oracle-bearing text for "
                f"{example['id']}: {joined_errors}"
            )
        rows.append(
            {
                "id": f"{example['id']}__{suffix}",
                "base_id": example["id"],
                "category": example["category"],
                "dimension": dimension,
                "question": question,
                "function": example["function"],
                "ground_truth": example["ground_truth"],
                "oracle_preservation": {
                    "function_schema_unchanged": True,
                    "ground_truth_unchanged": True,
                },
            }
        )

    write_jsonl(output_path, rows)
    print(f"Wrote {output_path.relative_to(REPO_ROOT)}")


def augment_typos() -> None:
    augment_dimension("typos", "typos", typo_prompt)


def augment_cursing() -> None:
    augment_dimension("cursing", "cursing", cursing_prompt)


def augment_irrelevant_context() -> None:
    augment_dimension("irrelevant_context", "context", irrelevant_context_prompt)


def augment_removed_spaces() -> None:
    augment_dimension("removed_spaces", "spaces", removed_spaces_prompt)


def augment_argumentative() -> None:
    augment_dimension("argumentative_challenge", "argue", argumentative_prompt)


def augment_profane_sandwich() -> None:
    augment_dimension("profane_sandwich", "profane_sandwich", profane_sandwich_prompt)


def augment_argumentative_sandwich() -> None:
    augment_dimension(
        "argumentative_sandwich",
        "argue_sandwich",
        argumentative_sandwich_prompt,
    )


def augment_distractor_sandwich() -> None:
    augment_dimension("distractor_sandwich", "distractor_sandwich", distractor_sandwich_prompt)


def augment_pasted_context_block() -> None:
    augment_dimension("pasted_context_block", "pasted_context_block", pasted_context_block_prompt)


def augment_telegraphic_request() -> None:
    augment_dimension("telegraphic_request", "telegraphic_request", telegraphic_request_prompt)


def augment() -> None:
    augment_typos()
    augment_cursing()
    augment_irrelevant_context()
    augment_removed_spaces()
    augment_argumentative()
    augment_profane_sandwich()
    augment_argumentative_sandwich()
    augment_distractor_sandwich()
    augment_pasted_context_block()
    augment_telegraphic_request()
    review_augmentations()


def prompt_text(example: dict[str, object]) -> str:
    return conversation_text(example["question"])


def review_augmentations() -> None:
    clean_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    output_path = REPO_ROOT / "artifacts/generated/augmentation_review.csv"

    if not clean_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")

    dimensions = (
        ("typos", "aug_typo"),
        ("cursing", "aug_cursing"),
        ("irrelevant_context", "aug_irrelevant_context"),
        ("removed_spaces", "aug_removed_spaces"),
        ("argumentative_challenge", "aug_argumentative"),
        ("profane_sandwich", "aug_profane_sandwich"),
        ("argumentative_sandwich", "aug_argumentative_sandwich"),
        ("distractor_sandwich", "aug_distractor_sandwich"),
        ("pasted_context_block", "aug_pasted_context_block"),
        ("telegraphic_request", "aug_telegraphic_request"),
    )
    generated_by_dimension = {}
    for dimension, _column in dimensions:
        path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(REPO_ROOT)}. Run augment first.")
        generated_by_dimension[dimension] = {row["base_id"]: row for row in read_jsonl(path)}

    examples = read_jsonl(clean_path)
    limit = optional_positive_int_env("REALISTIC_BFCL_AUGMENT_LIMIT")
    if limit is not None:
        examples = examples[:limit]
        print(f"Limiting review CSV to first {len(examples)} examples")

    fieldnames = [
        "base_id",
        "category",
        "clean_prompt",
        "aug_typo",
        "aug_cursing",
        "aug_irrelevant_context",
        "aug_removed_spaces",
        "aug_argumentative",
        "aug_profane_sandwich",
        "aug_argumentative_sandwich",
        "aug_distractor_sandwich",
        "aug_pasted_context_block",
        "aug_telegraphic_request",
        "function_names",
        "ground_truth",
    ]
    rows = []
    for example in examples:
        row = {
            "base_id": example["id"],
            "category": example["category"],
            "clean_prompt": prompt_text(example),
            "function_names": ", ".join(function["name"] for function in example["function"]),
            "ground_truth": json.dumps(example["ground_truth"], ensure_ascii=False),
        }
        for dimension, column in dimensions:
            augmented = generated_by_dimension[dimension][example["id"]]
            row[column] = prompt_text(augmented)
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path.relative_to(REPO_ROOT)}")
