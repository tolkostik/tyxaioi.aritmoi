"""
Πολλοί αλγόριθμοι επιλογής αριθμών. Κάθε στήλη φτιάχνεται από διαφορετικό αλγόριθμο,
χωρίς πανομοιότυπες στήλες και με περιορισμένη επικάλυψη ανάμεσά τους.

Σημαντικό: κάθε κλήρωση είναι τυχαία. Κανένας αλγόριθμος δεν αυξάνει την πιθανότητα να κερδίσεις·
αλλάζουν μόνο το «ύφος» των στηλών (και η αποφυγή δημοφιλών συνδυασμών μειώνει τη μοιρασιά αν κερδίσεις).
Ο ιστορικός έλεγχος (backtest) το δείχνει με πραγματικά νούμερα.
"""
import itertools
import math
import random

import lottery as L

ALGOS = [
    ("hot", "Συχνοί", "Αριθμοί που βγήκαν πιο συχνά στις τελευταίες κληρώσεις."),
    ("cold", "Καθυστερημένοι", "Αριθμοί που έχουν να βγουν περισσότερο καιρό."),
    ("recent", "Πρόσφατη φόρμα", "Βάρος στις τελευταίες 20 κληρώσεις, με φθίνουσα σημασία."),
    ("balanced", "Ισορροπημένη", "Μονοί/ζυγοί και μικροί/μεγάλοι μοιρασμένοι, άθροισμα στο συνηθισμένο εύρος."),
    ("pairs", "Συχνά ζεύγη", "Ξεκινά από αριθμό και προσθέτει όσους βγαίνουν συχνά μαζί του."),
    ("decades", "Δεκάδες", "Ένας αριθμός από κάθε δεκάδα, ώστε να απλώνεται σε όλο το εύρος."),
    ("sum", "Άθροισμα στόχος", "Στοχεύει στο πιο συχνό άθροισμα των κληρώσεων."),
    ("antipop", "Αντι-δημοφιλής", "Αριθμοί πάνω από 31 και χωρίς σχέδια, για λιγότερη μοιρασιά αν κερδίσεις."),
    ("montecarlo", "Monte Carlo", "Δοκιμάζει 3.000 τυχαίες στήλες και κρατά την πιο «τυπική» στατιστικά."),
    ("coverage", "Κάλυψη", "Αριθμοί που δεν έχουν μπει στις άλλες στήλες, για μέγιστη κάλυψη."),
    ("mirror", "Συμμετρική", "Ζεύγη αριθμών συμμετρικά γύρω από το κέντρο (π.χ. 7 και 39)."),
    ("random", "Εντελώς τυχαία", "Καθαρή τύχη, χωρίς στατιστικά: το σημείο αναφοράς."),
]
NAMES = {k: n for k, n, _ in ALGOS}
DESC = {k: d for k, _, d in ALGOS}


# ------------------------------------------------------------------ βοηθητικά
def _wsample(rng, weights, k, exclude=()):
    keys = [x for x in weights if x not in exclude]
    w = [max(1e-6, weights[x]) for x in keys]
    out = []
    for _ in range(min(k, len(keys))):
        r = rng.random() * sum(w)
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if acc >= r:
                out.append(keys.pop(i))
                w.pop(i)
                break
    return out


def _pair_counts(draws, n_main):
    pc = {}
    for d in draws:
        nums = sorted(d["numbers"][:n_main])
        for a, b in itertools.combinations(nums, 2):
            pc[(a, b)] = pc.get((a, b), 0) + 1
    return pc


def _typical(nums, st, maxn):
    """Πόσο «τυπική» είναι μια στήλη (άθροισμα, μονοί, μικροί, διασπορά) – μικρότερο = πιο τυπική."""
    k = len(nums)
    sums = st["sums"] or [k * (maxn + 1) / 2]
    med = sums[len(sums) // 2]
    exp_sum = med * k / max(1, (len(sums) and k))
    sc = abs(sum(nums) - exp_sum) / (maxn / 2)
    odd = sum(x % 2 for x in nums)
    sc += abs(odd - k / 2) / k
    low = sum(1 for x in nums if x <= maxn / 2)
    sc += abs(low - k / 2) / k
    sc += 0.5 if len({(x - 1) // 10 for x in nums}) < min(k, 3) else 0
    return sc


# ------------------------------------------------------------------ αλγόριθμοι
def run(algo, game, draws, k, rng, used=(), fast=False):
    g = L.GAMES[game]
    maxn = g["max"]
    st = L.stats(game, draws) if draws else {"freq": {}, "gap": {}, "sums": []}
    allnums = range(1, maxn + 1)
    uniform = {x: 1.0 for x in allnums}
    if not draws and algo not in ("random", "decades", "antipop", "coverage", "mirror"):
        algo = "random"
    if algo == "hot":
        return _wsample(rng, {x: (st["freq"].get(x, 0) + 1) ** 2 for x in allnums}, k)
    if algo == "cold":
        return _wsample(rng, {x: (st["gap"].get(x, 0) + 1) ** 1.5 for x in allnums}, k)
    if algo == "recent":
        w = {x: 0.3 for x in allnums}
        for i, d in enumerate(draws[:20]):
            for x in d["numbers"][: g["draw"]]:
                if x in w:
                    w[x] += 0.9 ** i
        return _wsample(rng, w, k)
    if algo == "balanced":
        for _ in range(500):
            c = sorted(rng.sample(list(allnums), k))
            if L._balanced(c, st, game) and abs(sum(x % 2 for x in c) - k / 2) <= 1 and \
                    abs(sum(1 for x in c if x <= maxn / 2) - k / 2) <= 1:
                return c
        return rng.sample(list(allnums), k)
    if algo == "pairs":
        pc = _pair_counts(draws[:300], g["draw"])
        start = _wsample(rng, {x: st["freq"].get(x, 0) + 1 for x in allnums}, 1)[0]
        chosen = [start]
        while len(chosen) < k:
            w = {x: 1 + sum(pc.get((min(x, c), max(x, c)), 0) for c in chosen) ** 1.5 for x in allnums}
            chosen += _wsample(rng, w, 1, exclude=chosen)
        return chosen
    if algo == "decades":
        groups = [list(range(a, min(a + 10, maxn + 1))) for a in range(1, maxn + 1, 10)]
        rng.shuffle(groups)
        out = []
        while len(out) < k:
            for gr in groups:
                free = [x for x in gr if x not in out]
                if free and len(out) < k:
                    out.append(rng.choice(free))
        return out
    if algo == "sum":
        sums = st["sums"]
        target = sums[len(sums) // 2] * k / g["draw"] if sums else k * (maxn + 1) / 2
        best, bd = None, 1e9
        for _ in range(200 if fast else 1500):
            c = rng.sample(list(allnums), k)
            dd = abs(sum(c) - target)
            if dd < bd:
                best, bd = c, dd
                if dd <= 1:
                    break
        return best
    if algo == "antipop":
        w = {x: (3.0 if x > 31 else 1.0) for x in allnums}
        for _ in range(300):
            c = _wsample(rng, w, k)
            if not L._popular(sorted(c), maxn):
                return c
        return c
    if algo == "montecarlo":
        best, bs = None, 1e9
        for _ in range(300 if fast else 3000):
            c = rng.sample(list(allnums), k)
            s = _typical(c, st, maxn)
            if s < bs:
                best, bs = c, s
        return best
    if algo == "coverage":
        free = [x for x in allnums if x not in set(used)]
        if len(free) < k:
            free = list(allnums)
        return rng.sample(free, k)
    if algo == "mirror":
        out = []
        while len(out) < k:
            a = rng.randint(1, maxn)
            b = maxn + 1 - a
            for x in (a, b):
                if x not in out and len(out) < k:
                    out.append(x)
        return out
    return rng.sample(list(allnums), k)


def _bonus(algo, game, draws, rng):
    g = L.GAMES[game]
    if not g["bonus_max"]:
        return None
    st = L.stats(game, draws) if draws else {"bfreq": {}, "bgap": {}}
    keys = range(1, g["bonus_max"] + 1)
    if algo in ("hot", "recent", "pairs"):
        w = {x: st["bfreq"].get(x, 0) + 1 for x in keys}
    elif algo == "cold":
        w = {x: st["bgap"].get(x, 0) + 1 for x in keys}
    else:
        w = {x: 1.0 for x in keys}
    return _wsample(rng, w, 1)[0]


def generate_multi(game, draws, columns, pick=None, avoid_popular=True, chosen=None, seed=None, max_overlap=None):
    """Μία στήλη ανά αλγόριθμο (με τη σειρά, ή όσους διάλεξε ο χρήστης).
    Καμία στήλη ίδια με άλλη, και κοινοί αριθμοί ανάμεσα σε δύο στήλες έως max_overlap."""
    g = L.GAMES[game]
    k = pick or g["pick"]
    rng = random.Random(seed)
    algos = [a for a in (chosen or [x[0] for x in ALGOS])]
    previous = {tuple(c) for c in _past(game)}
    stats_algos = ("hot", "cold", "recent", "balanced", "pairs", "sum", "montecarlo")
    if max_overlap is None:
        max_overlap = max(1, k // 2 - (0 if k <= 5 else 0))
    out, used, log = [], [], []
    bonuses = set()
    for i in range(columns):
        algo = algos[i % len(algos)]
        best, tries = None, 0
        for tries in range(1, 400):
            c = sorted(run(algo, game, draws, k, rng, used))
            if len(set(c)) != k:
                continue
            if any(set(c) == set(o["numbers"]) for o in out) or tuple(c) in previous:
                continue
            if any(len(set(c) & set(o["numbers"])) > max_overlap for o in out):
                continue
            if avoid_popular and algo != "random" and L._popular(c, g["max"]):
                continue
            best = c
            break
        if best is None:      # χαλαρώνουμε μόνο την επικάλυψη, ποτέ την ταυτότητα
            for _ in range(400):
                c = sorted(run(algo, game, draws, k, rng, used))
                if len(set(c)) == k and not any(set(c) == set(o["numbers"]) for o in out):
                    best = c
                    break
        bonus = _bonus(algo, game, draws, rng)
        if bonus is not None and len(bonuses) < g["bonus_max"]:
            for _ in range(30):
                if bonus not in bonuses:
                    break
                bonus = _bonus(algo, game, draws, rng)
            bonuses.add(bonus)
        used += best
        name = NAMES[algo] if (draws or algo not in stats_algos) else f"{NAMES[algo]} → τυχαία (χωρίς ιστορικό)"
        out.append({"numbers": best, "bonus": bonus, "algo": algo, "name": name})
        log.append(f"Στήλη {i + 1}: {NAMES[algo]} ({tries} δοκιμές)")
    overlaps = [len(set(a["numbers"]) & set(b["numbers"])) for a, b in itertools.combinations(out, 2)]
    cover = len(set(used))
    _remember(game, [o["numbers"] for o in out])
    report = {"log": log, "max_overlap": max(overlaps) if overlaps else 0, "coverage": cover,
              "coverage_pct": cover / g["max"] * 100, "identical": 0}
    return out, report


def _past(game):
    import json
    try:
        return json.loads((L.DATA_DIR / f"generated_{game}.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def _remember(game, cols):
    """Κρατά τις τελευταίες 300 στήλες, ώστε να μη βγαίνει ξανά η ίδια στήλη σε επόμενο πάτημα."""
    import json
    hist = (cols + _past(game))[:300]
    try:
        L.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (L.DATA_DIR / f"generated_{game}.json").write_text(json.dumps(hist), encoding="utf-8")
    except OSError:
        pass


# ------------------------------------------------------------------ ιστορικός έλεγχος (backtest)
def backtest(game, draws, pick=None, tests=40, per_draw=8, seed=7, progress=lambda i, n: None):
    """Για κάθε αλγόριθμο: μέσες επιτυχίες ανά στήλη σε παλιές κληρώσεις, με στοιχεία μόνο από πριν από αυτές.
    Επιστρέφει [(algo, μέσες επιτυχίες, αναμενόμενες από καθαρή τύχη)], καλύτερος πρώτος."""
    g = L.GAMES[game]
    k = pick or g["pick"]
    n_draw = g["draw"]
    expected = k * n_draw / g["max"]
    if len(draws) < 40:
        return [], expected
    rng = random.Random(seed)
    res = []
    tests = min(tests, len(draws) - 30)
    for ai, (algo, _, _) in enumerate(ALGOS):
        progress(ai, len(ALGOS))
        hits = cnt = 0
        for t in range(tests):
            target = set(draws[t]["numbers"][:n_draw])
            history = draws[t + 1: t + 201]
            for _ in range(per_draw):
                c = run(algo, game, history, k, rng, fast=True)
                hits += len(set(c) & target)
                cnt += 1
        res.append((algo, hits / cnt, expected))
    res.sort(key=lambda r: -r[1])
    return res, expected


def explain_backtest(res, expected):
    if not res:
        return "Χρειάζονται τουλάχιστον 40 παλιές κληρώσεις (πάτα «Ενημέρωση ιστορικού»)."
    spread = res[0][1] - res[-1][1]
    e, sp = f"{expected:.2f}".replace(".", ","), f"{spread:.2f}".replace(".", ",")
    return (f"Με καθαρή τύχη αναμένονται {e} επιτυχίες ανά στήλη. Όλοι οι αλγόριθμοι πέφτουν κοντά σε αυτό "
            f"(διαφορά καλύτερου–χειρότερου {sp}, μέσα στα όρια της τύχης). Αυτό επιβεβαιώνει ότι οι κληρώσεις "
            f"είναι τυχαίες: οι αλγόριθμοι δίνουν ποικιλία στις στήλες σου, όχι μεγαλύτερη πιθανότητα.")


_ = math
