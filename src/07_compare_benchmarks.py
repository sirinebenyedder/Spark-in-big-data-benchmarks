import json
from bench_common import RESULTS_DIR


def main() -> None:
    files = sorted(RESULTS_DIR.glob("b[0-4]_*.json"))
    if not files:
        print(f"Aucun fichier JSON trouvé dans {RESULTS_DIR}")
        return

    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    b0_time = next((r["duration_s"] for r in rows if r["code"] == "B0"), None)

    print(
        f"{'Code':<4} | {'Nom':<38} | {'p':>2} | {'P':>3} | {'AQE':<3} | "
        f"{'T (s)':>8} | {'S':>7} | {'E (%)':>7} | {'p*T (n-min)':>11} | {'D<=3m'}"
    )
    print("-" * 118)
    for r in rows:
        t_s = float(r["duration_s"])
        p = int(r["nodes_p"])
        s = (b0_time / t_s) if (b0_time and t_s > 0) else float(r.get("speedup_S", 1.0))
        e_pct = (s / p * 100.0) if p > 0 else 0.0
        cost_nmin = (p * t_s) / 60.0
        ok = "Oui" if t_s <= float(r.get("deadline_s", 180.0)) else "Non"
        aqe_str = "Oui" if r.get("aqe") else "Non"
        print(
            f"{r['code']:<4} | {r['nom'][:38]:<38} | {p:>2} | {r['partitions_P']:>3} | "
            f"{aqe_str:<3} | {t_s:>8.2f} | {s:>6.2f}x | {e_pct:>6.1f}% | {cost_nmin:>11.2f} | {ok}"
        )


if __name__ == "__main__":
    main()