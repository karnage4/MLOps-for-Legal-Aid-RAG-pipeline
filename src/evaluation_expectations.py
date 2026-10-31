"""Human-reviewed expectations for the held-out Legal-Aid RAG queries."""

from __future__ import annotations

from src.evaluation_queries import TEST_QUERIES


EXPECTATION_VERSION = "1"

EXPECTED_FACTS_BY_QUERY = {
    TEST_QUERIES[0]: [
        "The estate passes to legally eligible heirs rather than one universal recipient.",
        "In Pakistan, a Muslim's estate is distributed under applicable Islamic inheritance rules, while a non-Muslim's estate follows the applicable personal law.",
        "For a Muslim estate, heirs may include fixed-share heirs, residuaries, and, where applicable, distant kindred.",
        "The actual heirs and shares depend on the deceased's religion and surviving family members.",
    ],
    TEST_QUERIES[1]: [
        "One daughter with no son receives one-half of the estate.",
        "Two or more daughters with no son collectively receive two-thirds of the estate.",
        "When sons and daughters inherit together, each son generally receives twice each daughter's share.",
        "Final distribution depends on other surviving heirs and occurs after applicable debts and valid bequests.",
    ],
    TEST_QUERIES[2]: [
        "Under the Islamic inheritance rules represented in this project's corpus, a non-Muslim generally does not inherit from a Muslim, and vice versa.",
        "Marriage to a non-Muslim does not by itself necessarily establish that a Muslim heir has ceased to be Muslim.",
        "The determination can depend on the person's legal or religious status and applicable Pakistani law, so specific cases require qualified legal review.",
    ],
}

