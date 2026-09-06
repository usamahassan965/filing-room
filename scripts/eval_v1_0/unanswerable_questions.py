"""The ten unanswerable questions, and why each one is unanswerable.

A question is only useful here if a fluent system would *want* to answer it: it
has to be shaped exactly like the other 140, name a real company or a real
figure, and read as ordinary. What makes it unanswerable is never the wording --
it is that the corpus does not contain the answer, for one of four reasons:

* the company is not in the corpus at all (20 tickers, no others);
* the period is outside it (filings run 2020-01-26 to 2025-02-16);
* the fact lives in a document we did not collect (a proxy, a transcript);
* the fact is a forecast, which no filing states.

The reason is recorded per question, because "the system refused" is only a good
outcome if it refused for the right reason, and a reader has to be able to check
that the question really is unanswerable rather than merely hard.
"""

# (question, reason-code, note)
UNANSWERABLE: list[tuple[str, str, str]] = [
    (
        "What was Apple's total net sales for fiscal 2023?",
        "company-absent",
        "Apple is not one of the twenty tickers in the corpus; no Apple filing was ever fetched.",
    ),
    (
        "How much did Tesla spend on research and development in 2022?",
        "company-absent",
        "Tesla is not one of the twenty tickers in the corpus.",
    ),
    (
        "What net sales did Walmart report for fiscal 1998?",
        "period-absent",
        "The corpus begins with period_end 2020-01-26; nothing from the 1990s was fetched.",
    ),
    (
        "How many employees did Broadcom have at the end of fiscal 2016?",
        "period-absent",
        "Broadcom is in the corpus, but its earliest filing here covers fiscal 2019 onward.",
    ),
    (
        "What was NVIDIA's chief executive paid in total compensation for fiscal 2024?",
        "document-absent",
        "Executive compensation is disclosed in the DEF 14A proxy statement, which Item 11 of "
        "the 10-K incorporates by reference. The corpus holds 10-K and 10-Q filings only.",
    ),
    (
        "Which beneficial owners held more than five percent of Costco's common stock in 2024?",
        "document-absent",
        "Beneficial ownership is a proxy-statement disclosure incorporated by reference into "
        "Item 12; no proxy statement is in the corpus.",
    ),
    (
        "What did ExxonMobil's chief executive say about the energy transition at the company's "
        "2024 annual meeting of shareholders?",
        "document-absent",
        "Remarks at an annual meeting are not part of any periodic report.",
    ),
    (
        "What dividend per share does Pfizer expect to pay in 2027?",
        "forecast",
        "A forward dividend three years out is not stated in any filing; a system that answers "
        "this has invented it.",
    ),
    (
        "What crude oil price does Chevron assume for 2035 in its long-range planning?",
        "forecast",
        "Chevron discusses commodity price risk but does not publish a 2035 planning price.",
    ),
    (
        "How many paid members shopped at Costco's warehouses in Japan during fiscal 2023?",
        "granularity",
        "Costco reports membership counts in total and warehouse counts by country, but does "
        "not break paid membership out by country.",
    ),
]
