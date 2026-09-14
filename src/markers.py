"""Source provenance markers, the fix for defect D3.

A provenance marker is a token that identifies where a document came from rather than
what it says. The distinction is semantic and cannot be made statistically: on the
training split both "enron" and "beneficiary" appear in one class and nowhere else, but
only one of them would still hold on mail from a different organisation.

So the rule applied here is stated rather than fitted:

    Block a token if it names a specific organisation, person, place, phone fragment,
    mail infrastructure component, or corpus collection tool. Keep every token that
    describes the content, intent or rhetoric of the message.

Candidates were surfaced from the training split only, at document frequency 25 or more
with 95 percent or more of occurrences in one class, then classified by hand. The full
candidate list and the decision for each entry is in docs/phase4_findings.md.

The leak is not one sided. Normal leaks the company that produced it, Spam leaks the
tools that collected it, and Fraud leaked an encoding artifact we introduced ourselves.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------- the blocklist

# The Normal class is real internal Enron mail, so the company identifies the class.
ENRON_ORGANISATION = {
    "enron", "ect", "hou", "ees", "ebs", "eb", "lon", "enronxgate", "xgate", "vkamins",
}

# Named individuals in the Enron corpus. A person's name cannot indicate legitimacy.
ENRON_PEOPLE = {
    "vince", "kaminski", "shirley", "crenshaw", "stinson", "gibner", "zimin",
    "vasant", "tanya", "shanbhogue", "masson", "kevin", "jeff",
}

# Where the company was, including its phone fragments.
ENRON_PLACE = {"houston", "713", "853", "5290"}

# Mail client and header furniture, left over from how each corpus was captured.
#
# "am" was here for AM and PM timestamps and has been removed. It is also the English
# verb, and it opens almost every advance fee email as "I am Mr ...", so blocking it
# deleted genuine content. It had never appeared in the skew evidence either; it was
# added by assumption. "pm" carries no such collision and stays.
HEADER_FURNITURE = {"cc", "pm", "fyi", "doc", "edu"}

# Blocking a token that is also an ordinary English word deletes content, not
# provenance. Asserted by tests rather than left to review.
COMMON_ENGLISH_WORDS = frozenset({
    "am", "is", "are", "was", "be", "been", "the", "and", "or", "but", "if", "of",
    "to", "in", "on", "at", "for", "with", "from", "by", "as", "it", "he", "she",
    "they", "we", "you", "i", "me", "my", "your", "our", "his", "her", "this",
    "that", "these", "those", "not", "no", "yes", "all", "any", "can", "will",
    "have", "has", "had", "do", "does", "did", "so", "up", "out", "one", "two",
    "new", "now", "how", "who", "what", "when", "where", "why", "which", "than",
    "then", "there", "here", "also", "very", "more", "most", "some", "such", "own",
})

# The tools used to gather the spam corpus. Their names are a collection artifact and
# appear in 28 and 27 training documents respectively, all of them Spam.
SPAM_COLLECTION_TOOLS = {"spamassassin", "projecthoneypot"}

BLOCKLIST: frozenset[str] = frozenset(
    ENRON_ORGANISATION | ENRON_PEOPLE | ENRON_PLACE | HEADER_FURNITURE | SPAM_COLLECTION_TOOLS
)

BLOCKLIST_GROUPS: dict[str, frozenset[str]] = {
    "enron_organisation": frozenset(ENRON_ORGANISATION),
    "enron_people": frozenset(ENRON_PEOPLE),
    "enron_place": frozenset(ENRON_PLACE),
    "header_furniture": frozenset(HEADER_FURNITURE),
    "spam_collection_tools": frozenset(SPAM_COLLECTION_TOOLS),
}

# Tokens that are skewed just as hard but are kept, because they describe the message.
# Recorded so the curation is auditable rather than implicit.
KEPT_DESPITE_SKEW: dict[str, str] = {
    "beneficiary": "the rhetoric of an advance fee pitch",
    "kin": "next of kin, core advance fee framing",
    "deceased": "core advance fee framing",
    "foreigner": "core advance fee framing",
    "modalities": "register and vocabulary of the scam genre",
    "consignment": "register and vocabulary of the scam genre",
    "abidjan": "genuine geography of advance fee fraud, not a corpus artifact",
    "senegal": "genuine geography of advance fee fraud",
    "abacha": "a real figure central to the 419 genre",
    "viagra": "genuine spam product vocabulary",
    "oniine": "deliberate misspelling of online, a real spam evasion tactic",
    "shlpplng": "deliberate misspelling of shipping, a real spam evasion tactic",
    "mortgage": "genuine spam product vocabulary",
    "derivatives": "genuine subject matter of legitimate corporate mail",
    "valuation": "genuine subject matter of legitimate corporate mail",
    "interview": "genuine subject matter of legitimate corporate mail",
}

TOKEN_RE = re.compile(r"\b[a-z0-9]+\b")


def strip_markers(text: str, blocklist: frozenset[str] = BLOCKLIST) -> str:
    """Remove provenance markers from already normalised text."""
    return " ".join(t for t in TOKEN_RE.findall(text) if t not in blocklist)


def keep_only_markers(text: str, blocklist: frozenset[str] = BLOCKLIST) -> str:
    """Keep only the provenance markers, discarding everything else.

    Feeding this to a classifier measures the ceiling of the leak directly. Whatever
    score it reaches is achieved with no access to content at all.
    """
    return " ".join(t for t in TOKEN_RE.findall(text) if t in blocklist)
