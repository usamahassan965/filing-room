"""The 60 narrative questions, written by hand against located spans.

Each entry is (candidate keys, question, note). A key is an int index into
narrative_cands.json or "E<n>" into narrative_extra.json. More than one key
means the same disclosure appears in more than one filing in the corpus and
every occurrence is gold -- without that, a system that returns the 2022 copy
of a passage repeated in 2021 and 2023 is scored as wrong for finding it.
"""

QUESTIONS: list[tuple[list, str, str]] = [
    # ---- Walmart ----------------------------------------------------------
    (
        [0],
        "How large was Walmart's working capital deficit at the end of fiscal 2024, "
        "and why was it smaller than the year before?",
        "",
    ),
    (
        [1],
        "What did Walmart say about how it responded to cost inflation from its "
        "suppliers during fiscal 2024?",
        "",
    ),
    (
        [35],
        "Why does Walmart report Return on Investment alongside Return on Assets, "
        "and what does it say the two measures are used for?",
        "",
    ),
    # ---- NVIDIA -----------------------------------------------------------
    ([2], "Which costs does NVIDIA include in cost of revenue when it computes gross profit?", ""),
    (
        [3, 16],
        "How does NVIDIA recognize revenue on arrangements that require it to "
        "significantly customize its intellectual property, and how is progress "
        "toward completion measured?",
        "the same accounting policy appears in two NVIDIA 10-Ks; both are gold",
    ),
    (
        [17],
        "What was NVIDIA to pay for Arm under the September 2020 purchase agreement, "
        "and how was that consideration split?",
        "",
    ),
    (
        ["E4"],
        "How much did NVIDIA return to shareholders in fiscal year 2020, and did it "
        "repurchase any shares that year?",
        "",
    ),
    # ---- Phillips 66 ------------------------------------------------------
    (
        [5],
        "What short-term uncommitted borrowing capacity did Phillips 66 have available "
        "in early 2025, and how much was drawn?",
        "",
    ),
    (
        [6],
        "How much was outstanding under Phillips 66's advance term loan agreements with "
        "WRB at the end of 2023, and what happened to that loan a year later?",
        "",
    ),
    (
        [14],
        "How much cash did DCP LP distribute to common unitholders other than Phillips 66 "
        "after the August 2022 merger, and how often must it distribute available cash?",
        "",
    ),
    (
        [15],
        "At what level does Phillips 66 group its long-lived assets to test them for "
        "impairment, and what triggers the test?",
        "",
    ),
    (
        ["E1"],
        "Who owns the Gray Oak Pipeline entity, and what interest did Phillips 66 "
        "Partners hold in it?",
        "",
    ),
    # ---- Exxon Mobil ------------------------------------------------------
    (
        [7, 28, 30],
        "What does ExxonMobil say could happen to its business if government "
        "climate policy or the pace of the energy transition shifts?",
        "ExxonMobil repeats this risk factor across three 10-Ks; all three are gold",
    ),
    ([29], "What does ExxonMobil say about the projects where it is not the operator?", ""),
    # ---- Qualcomm ---------------------------------------------------------
    (
        [8, 48, 58],
        "What risks does Qualcomm attach to entering new product areas and "
        "industries outside its established business?",
        "Qualcomm repeats this risk factor across three 10-Ks; all three are gold",
    ),
    (
        [49],
        "What preferential United States tax treatment does Qualcomm say a significant "
        "part of its income qualifies for, and at roughly what rate?",
        "",
    ),
    # ---- AMD --------------------------------------------------------------
    (
        [9],
        "What drove the change in AMD's cash and short-term investment balances during 2022?",
        "",
    ),
    (
        [73],
        "What effective tax rates did AMD report for 2021 and 2020, and what explains "
        "the difference between them?",
        "",
    ),
    (
        [74],
        "What did AMD point to in 2020 as evidence that it could sustain profitability "
        "for purposes of its deferred tax assets?",
        "",
    ),
    # ---- Lowe's -----------------------------------------------------------
    (
        [10],
        "What did Lowe's say about the size of its long-lived asset impairment charges "
        "in fiscal 2021 and 2020?",
        "",
    ),
    (
        [21],
        "By how much would a ten percent change in Lowe's self-insurance liability have "
        "moved net earnings in fiscal 2020?",
        "",
    ),
    # ---- Home Depot -------------------------------------------------------
    (
        [12],
        "How much cash did Home Depot hold at the end of fiscal 2023, and how much of it "
        "sat at foreign subsidiaries?",
        "",
    ),
    (
        [13],
        "What does Home Depot count as a purchase obligation, and how does it treat "
        "inventory purchase orders that can be cancelled?",
        "",
    ),
    (
        [54],
        "How did Home Depot change the size of its commercial paper programs in March "
        "2020, and what credit facilities backed them up?",
        "",
    ),
    (
        [55],
        "How often does Home Depot evaluate long-lived assets for impairment, and what "
        "does it treat as an indicator that a test is needed?",
        "",
    ),
    # ---- Target -----------------------------------------------------------
    (
        [20, 25, 43],
        "Why does Target present an adjusted earnings per share figure, and what "
        "does it exclude from it?",
        "Target repeats this non-GAAP explanation across three 10-Ks; all three are gold",
    ),
    (
        [44],
        "What were Target's period-end cash and cash equivalents in fiscal 2019 compared "
        "with the prior year, and how much of that was short-term investments?",
        "",
    ),
    # ---- Broadcom ---------------------------------------------------------
    (
        [23],
        "By how much did Broadcom's research and development expense rise in fiscal 2019, "
        "and what caused the increase?",
        "",
    ),
    (
        [24],
        "By how much did Broadcom's selling, general and administrative expense rise in "
        "fiscal 2019, in dollars and in percentage terms?",
        "",
    ),
    (
        [39],
        "Why did Broadcom's selling, general and administrative expense fall in fiscal 2021?",
        "",
    ),
    (
        [40],
        "Which Broadcom entities issued the notes under the 2017 Indentures, and in what "
        "principal amounts?",
        "",
    ),
    # ---- Eli Lilly --------------------------------------------------------
    (
        [26],
        "Which currencies is Eli Lilly most exposed to, and how does it manage that exposure?",
        "",
    ),
    (
        [27],
        "What was Eli Lilly's gross margin as a percentage of revenue in 2020, and what "
        "moved it against the prior year?",
        "",
    ),
    (
        [33],
        "What cost-containment measures does Eli Lilly say governments use to hold down "
        "pharmaceutical prices?",
        "",
    ),
    # ---- AbbVie -----------------------------------------------------------
    (
        [32],
        "Which actuarial assumptions does AbbVie say it reviews each year for its pension "
        "and other post-employment plans?",
        "",
    ),
    (
        [62],
        "How does AbbVie value the intangible assets it acquires in a business "
        "combination, and what does that model require it to estimate?",
        "",
    ),
    (
        [63],
        "What did AbbVie tell investors about the effect of the COVID-19 pandemic on its "
        "operations in its 2020 annual report?",
        "",
    ),
    # ---- Chevron ----------------------------------------------------------
    (
        [37],
        "What does Chevron identify as the single most significant factor affecting its "
        "results, and what does it say drives that factor?",
        "",
    ),
    (
        [38, 42, 59],
        "What does Chevron say exposes it to liability from litigation and "
        "government action, including climate-related claims?",
        "Chevron repeats this risk factor across three 10-Ks; all three are gold",
    ),
    # ---- SLB --------------------------------------------------------------
    (
        [45, 53],
        "How does SLB recognize revenue on long-term construction-type contracts, and "
        "how does it measure progress on them?",
        "the same accounting policy appears in two SLB 10-Ks; both are gold",
    ),
    ([52], "How does SLB decide how large its allowance for doubtful accounts should be?", ""),
    # ---- Merck ------------------------------------------------------------
    (
        [50, 18],
        "When does Merck record an accrual for a loss contingency, and what does it do "
        "when no single amount in a range is a better estimate?",
        "the same accounting policy appears in two Merck 10-Ks; both are gold",
    ),
    (
        [51],
        "Why does Merck report non-GAAP income and earnings per share alongside its GAAP figures?",
        "",
    ),
    (
        ["E5"],
        "What was Merck's net periodic benefit cost for its pension plans in 2022, 2021 and 2020?",
        "",
    ),
    # ---- Pfizer -----------------------------------------------------------
    (
        [56],
        "What does Pfizer say about the risk of infringing, or having to challenge, "
        "intellectual property rights held by others?",
        "",
    ),
    ([57], "What manufacturing and supply chain disruptions does Pfizer list among its risks?", ""),
    (
        [64],
        "What stake does Pfizer hold in the consumer healthcare joint venture with GSK, "
        "and why does Pfizer treat that position as a risk?",
        "",
    ),
    ([65], "What was the outcome of the Phase 3 JAVELIN Lung 100 trial of avelumab?", ""),
    (
        ["E3"],
        "How much gross cost savings did Pfizer report from restructuring its corporate "
        "enabling functions in 2022?",
        "",
    ),
    # ---- Johnson & Johnson ------------------------------------------------
    (
        [61],
        "What did Johnson & Johnson's Surgery franchise sell in 2023, and how did that "
        "compare with the prior year?",
        "",
    ),
    (
        [69],
        "What did Johnson & Johnson's Surgery franchise sell in 2024, and what was behind "
        "the fall in Advanced Surgery?",
        "",
    ),
    (
        ["E0"],
        "What was Johnson & Johnson's net debt position at its 2022 fiscal year end, and "
        "what caused it to grow so much against the prior year?",
        "",
    ),
    (
        ["E2"],
        "What net interest expense did Johnson & Johnson report for fiscal 2020, and why "
        "did it swing from the prior year?",
        "",
    ),
    # ---- ConocoPhillips ---------------------------------------------------
    ([66], "What are ConocoPhillips's largest asset removal obligations?", ""),
    (
        [67, 71],
        "Which ConocoPhillips entities cross-guarantee the company's publicly held debt "
        "securities, and how are they owned?",
        "the same disclosure appears in two ConocoPhillips 10-Ks; both are gold",
    ),
    (
        [70],
        "In how many countries did ConocoPhillips have operations at the end of 2024, and "
        "where is the company headquartered?",
        "",
    ),
    # ---- Costco -----------------------------------------------------------
    (
        [72],
        "What senior notes did Costco issue in April 2020, and what did it do with the proceeds?",
        "",
    ),
    ([41], "How much cash did Costco's operations provide in fiscal 2023?", ""),
    (
        [4, 41],
        "How much cash did Costco's operations provide in fiscal 2022?",
        "reported in the fiscal 2022 10-K and again as the comparative in the fiscal 2023 10-K",
    ),
    (
        [47, 4],
        "How much cash did Costco's operations provide in fiscal 2021 compared with fiscal 2020?",
        "reported in the fiscal 2021 10-K and again as the comparative in the fiscal 2022 10-K",
    ),
]
