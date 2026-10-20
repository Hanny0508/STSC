import json
import sys
from collections import defaultdict

F = sys.argv[1] if len(sys.argv) > 1 else "artifacts/text_domain/result_after_reveal.json"


with open(F, "r", encoding="utf-8") as f:
    data = json.load(f)


print("ROOT KEYS:")
print(data.keys())


cases = data["cases"]

print()
print("TOTAL CASES =", len(cases))


summary = defaultdict(lambda:{
    "attempts":0,
    "success":0,
    "fail":0,
    "encode":[],
    "decode":[]
})


for c in cases:

    payload = c["payload_bits"]

    summary[payload]["attempts"] += 1


    ok = (
        c.get("decoder_succeeded", False)
        and c.get("payload_exact_match", False)
    )


    if ok:
        summary[payload]["success"] += 1
    else:
        summary[payload]["fail"] += 1


    summary[payload]["encode"].append(
        c.get("encode_seconds",0)
    )

    summary[payload]["decode"].append(
        c.get("decode_seconds",0)
    )



print()
print("="*70)
print("PAYLOAD CAPACITY SUMMARY")
print("="*70)


for p in sorted(summary):

    x = summary[p]

    rate = x["success"]/x["attempts"]*100


    print(
        f"{p:4d} bits | "
        f"attempts={x['attempts']:3d} | "
        f"success={x['success']:3d} | "
        f"fail={x['fail']:3d} | "
        f"rate={rate:6.2f}% | "
        f"avg_encode={sum(x['encode'])/len(x['encode']):.3f}s | "
        f"avg_decode={sum(x['decode'])/len(x['decode']):.3f}s"
    )