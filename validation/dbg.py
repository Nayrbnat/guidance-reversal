import re, sys
sys.path.insert(0, "src")
import gg_labels as L

cases = [
    ("Martin has been named President and Chief Executive Officer, following the resignation of William H. Osborne.", ["OSBORNE"]),
    ("He will succeed Joseph M. Zubretsky, who has accepted a career opportunity outside of the property and casualty industry.", ["ZUBRETSKY"]),
    ("Texas Pacific Land Trust Announces Resignation of David M. Peterson as General Agent and Principal Executive Officer, Effective November 4, 2016.", ["PETERSON"]),
    ("the Board of Directors of the company accepted the resignation of Michael G. Rhodes from his positions as Chief Executive Officer of the Company, effective April 1, 2024.", ["RHODES"]),
]
for s, lt in cases:
    nm = re.search(r"\b" + lt[0] + r"\b", s, re.I)
    before, after = s[:nm.start()], s[nm.end():]
    print("----", s[:70])
    print(" ds:", bool(L.departing_sentences(s, lt)))
    print(" succeeds_name:", bool(re.search(r"\b(succeed\w*|replac\w*)\s+(?:\w+\.?\s+){0,4}$", before[-45:], re.I)),
          "| poss:", bool(re.search(r"\b(?:resignation|departure|retirement|termination|separation)\s+of\s+(?:\w+\.?\s+){0,4}$", before[-60:], re.I)))
    print(" before tail:", repr(before[-60:]))
print(L.PAST_DEP.pattern[:120])
print(repr(L.FUTURE_DEP.pattern[:60]))
