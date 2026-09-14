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
    # Added at phase 5. The phase 4 sweep used document frequency 25 or more, which
    # missed individuals appearing in 20 to 24 documents.
    "anjam", "donna",
}

# Where the company was, including its phone fragments, and the institutions that
# recur in its mail. Added at phase 5: wharton, baylor and hsb are a business school,
# a university and an internal acronym, each at 96 to 100 percent Normal purity.
ENRON_PLACE = {"houston", "713", "853", "5290", "wharton", "baylor", "hsb"}

# Mail client and header furniture, left over from how each corpus was captured.
#
# "am" was here for AM and PM timestamps and has been removed. It is also the English
# verb, and it opens almost every advance fee email as "I am Mr ...", so blocking it
# deleted genuine content. It had never appeared in the skew evidence either; it was
# added by assumption. "pm" carries no such collision and stays.
# "forwarded" and "corp" added at phase 5. "forwarded" is the "Forwarded by ..."
# separator in 183 Normal documents at 89 percent purity, and "corp" is the tail of
# "Enron Corp" in 111 at 90 percent. Both are furniture, not content.
HEADER_FURNITURE = {"cc", "pm", "fyi", "doc", "edu", "forwarded", "corp"}

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
# "listinfo" added at phase 5: a Mailman URL fragment in 24 Spam documents at 92
# percent purity, so it records how the corpus was gathered rather than what it says.
SPAM_COLLECTION_TOOLS = {"spamassassin", "projecthoneypot", "listinfo"}

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

# Every high purity term surfaced by the phase 5 discriminative ranking, with the verdict
# recorded. The phase 5 check asserts that no such term is missing from here, so the
# review cannot pass by construction the way a "nothing blocked survived stripping" check
# would. Entries whose verdict starts with "blocked" must also appear in BLOCKLIST.
PHASE5_REVIEWED: dict[str, str] = {
    # blocked, provenance
    "anjam": "blocked: named Enron individual",
    "donna": "blocked: named Enron individual",
    "wharton": "blocked: business school that recurs in Enron recruiting mail",
    "baylor": "blocked: university local to the company",
    "hsb": "blocked: internal Enron acronym",
    "forwarded": "blocked: mail header separator",
    "corp": "blocked: tail of the company name",
    "listinfo": "blocked: Mailman URL fragment, records how the spam was collected",
    # kept, genuine content
    "senegal": "kept: genuine advance fee fraud geography",
    "dakar": "kept: genuine advance fee fraud geography",
    "ghana": "kept: genuine advance fee fraud geography",
    "ivory": "kept: genuine advance fee fraud geography",
    "ivoire": "kept: genuine advance fee fraud geography",
    "abidjan": "kept: genuine advance fee fraud geography",
    "liberia": "kept: genuine advance fee fraud geography",
    "burkina": "kept: genuine advance fee fraud geography",
    "faso": "kept: genuine advance fee fraud geography",
    "accra": "kept: genuine advance fee fraud geography",
    "dormant": "kept: the rhetoric of an advance fee pitch",
    "hospital": "kept: the narrative furniture of an advance fee pitch",
    "dearest": "kept: register of the scam genre",
    "beloved": "kept: register of the scam genre",
    "lord": "kept: register of the scam genre",
    "donate": "kept: the rhetoric of an advance fee pitch",
    "inheritance": "kept: core advance fee framing",
    "inherit": "kept: core advance fee framing",
    "chamber": "kept: chamber of commerce, scam narrative furniture",
    "frozen": "kept: frozen account, core advance fee framing",
    "seized": "kept: seized funds, core advance fee framing",
    "assuring": "kept: register of the scam genre",
    "21st": "kept: appears in scam date narratives, not a corpus marker",
    "5m": "kept: a sum of money, genuine content",
    "5million": "kept: a sum of money, genuine content",
    "andmanyother": "kept: spam template text",
    "viagra": "kept: genuine spam product vocabulary",
    "mailings": "kept: genuine spam vocabulary",
    "shlpplng": "kept: deliberate misspelling of shipping, a real evasion tactic",
    "oniine": "kept: deliberate misspelling of online, a real evasion tactic",
    "miiiion": "kept: deliberate misspelling of million, a real evasion tactic",
    "prlces": "kept: deliberate misspelling of prices, a real evasion tactic",
    "mortgage": "kept: genuine spam product vocabulary",
    "shops": "kept: genuine spam vocabulary",
    "opt": "kept: opt out language, genuine spam vocabulary",
    "sightings": "kept: spam template text",
    "va": "kept: ambiguous, a US state abbreviation in spam addresses",
    "fl": "kept: ambiguous, a US state abbreviation in spam addresses",
    "advertisement": "kept: genuine spam vocabulary",
    "solicitation": "kept: genuine spam vocabulary",
    "dinner": "kept: genuine subject matter of legitimate mail",
    "derivatives": "kept: genuine subject matter of legitimate mail",
    "speaker": "kept: genuine subject matter of legitimate mail",
    "agenda": "kept: genuine subject matter of legitimate mail",
    "lunch": "kept: genuine subject matter of legitimate mail",
    "valuation": "kept: genuine subject matter of legitimate mail",
    "interview": "kept: genuine subject matter of legitimate mail",
    "resume": "kept: genuine subject matter of legitimate mail",
    "rice": "kept: ambiguous, Rice University locally but also a common noun",
    "jul": "kept: ambiguous, a date abbreviation that number masking does not catch",
    "var": "kept: value at risk, genuine finance content",
    # Second review pass. Removing "forwarded" and "corp" shifted the ranking and
    # surfaced these. None is provenance; all describe what the mail is about.
    "conference": "kept: genuine subject matter of legitimate mail",
    "meeting": "kept: genuine subject matter of legitimate mail",
    "chair": "kept: meeting chair, genuine subject matter",
    "comments": "kept: genuine subject matter of legitimate mail",
    "suggested": "kept: genuine subject matter of legitimate mail",
    "model": "kept: quantitative modelling, genuine subject matter",
    "modeling": "kept: quantitative modelling, genuine subject matter",
    "weather": "kept: weather derivatives were real Enron business, genuine content",
    "professor": "kept: academic correspondence, genuine subject matter",
    "students": "kept: academic correspondence, genuine subject matter",
    "hr": "kept: human resources, generic business content not an Enron marker",
    "units": "kept: genuine subject matter of legitimate mail",
    "friday": "kept: ambiguous, a day name. Blocking day names risks content, see D12",
    "weekend": "kept: ambiguous, a calendar word rather than a corpus marker",
    "yesterday": "kept: ambiguous, a calendar word rather than a corpus marker",
    "cd": "kept: a product sold in spam, genuine content",
    "sex": "kept: genuine spam product vocabulary",
    "sexual": "kept: genuine spam product vocabulary",
    "proven": "kept: spam persuasion vocabulary",
    "thousands": "kept: spam persuasion vocabulary",
    "growing": "kept: spam persuasion vocabulary",
    "incredibly": "kept: spam persuasion vocabulary",
    "differ": "kept: spam template text",
    "successfull": "kept: deliberate or careless misspelling, real spam signal",
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
