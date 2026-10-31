"""Shared held-out query set for RAG evaluation workflows."""


# Fifteen queries covering key inheritance-law topics. The current evaluation
# and managed dataset both use the first three so their evidence stays aligned.
TEST_QUERIES = [
    "Who inherits property when a person dies without a will?",
    "What is the share of a daughter in Islamic inheritance?",
    "Can a non-Muslim inherit from a Muslim's estate?",
    "How is the estate divided when there are multiple heirs?",
    "What rights does a spouse have over inherited property?",
    "Can a person be disinherited under Pakistani inheritance law?",
    "What is the difference between a will and intestate succession?",
    "How are debts handled before distributing an inheritance?",
    "What happens to property if all direct heirs are deceased?",
    "Can a minor child be an heir?",
    "What is the role of an executor in inheritance proceedings?",
    "How is agricultural land treated differently in inheritance?",
    "What documents are required to claim inheritance?",
    "Can a gift given during lifetime affect inheritance share?",
    "What is the waiting period before inheritance can be claimed?",
]
