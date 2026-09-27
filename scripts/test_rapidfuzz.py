import rapidfuzz
from rapidfuzz import fuzz

print("RapidFuzz Version:", rapidfuzz.__version__)
print("Token Sort Ratio:", fuzz.token_sort_ratio("Prabhav Center Business", "Business Center Prabhav"))
print("Token Set Ratio:", fuzz.token_set_ratio("Near SBI ATM, 797 Lake Town Block A", "797 Lake Town Block A"))
print("Fuzz Ratio:", fuzz.ratio("Reliance Retail Ltd", "Reliance Retail Limited"))
