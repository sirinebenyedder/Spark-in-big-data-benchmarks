import json
from bench_common import RESULTS_DIR


def main() -> None:
    files = sorted(RESULTS_DIR.glob("b[0-4]_*.json"))
    if not files:
        print(f"Aucun fichier JSON trouvé dans {RESULTS_DIR}")
        return

    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    b0_time = next((float(r["duration_s"]) for r in rows if r.get("code") == "B0"), None)
    b2_row = next((r for r in rows if r.get("code") == "B2"), None)
    b2_cost_nmin = (
        (int(b2_row["nodes_p"]) * float(b2_row["duration_s"])) / 60.0
        if b2_row
        else None
    )

    print(
        f"{'Code':<4} | {'Nom':<38} | {'p':>2} | {'P':>3} | {'AQE':<3} | "
        f"{'T (s)':>8} | {'S':>7} | {'E (%)':>7} | {'p*T (n-min)':>11} | {'G vs B2':>8} | {'D<=3m'}"
    )
    print("-" * 129)
    for r in rows:
        t_s = float(r["duration_s"])
        p = int(r["nodes_p"])
        s = (b0_time / t_s) if (b0_time and t_s > 0) else float(r.get("speedup_S", 1.0))
        e_pct = (s / p * 100.0) if p > 0 else 0.0
        cost_nmin = (p * t_s) / 60.0
        deadline = float(r.get("deadline_s", 180.0))
        ok = "Oui" if t_s <= deadline else "Non"
        aqe_val = bool(r.get("aqe", r.get("aqe_enabled", False)))
        aqe_str = "Oui" if aqe_val else "Non"

        if b2_cost_nmin and b2_cost_nmin > 0:
            gain_pct = ((b2_cost_nmin - cost_nmin) / b2_cost_nmin) * 100.0
            gain_str = f"{gain_pct:>+7.1f}%"
        else:
            gain_str = f"{'N/A':>8}"

        print(
            f"{r['code']:<4} | {str(r.get('nom', ''))[:38]:<38} | {p:>2} | {int(r['partitions_P']):>3} | "
            f"{aqe_str:<3} | {t_s:>8.2f} | {s:>6.2f}x | {e_pct:>6.1f}% | {cost_nmin:>11.4f} | {gain_str} | {ok}"
        )


if __name__ == "__main__":
    main()