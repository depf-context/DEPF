from functools import lru_cache


def normalize(text):
    return ' '.join(str(text).split())


def candidate_texts(cell, row):
    if cell.startswith('FEVER'):
        return [normalize(s) for s in row['candidates']]
    if cell == 'HoVer4B':
        return [normalize(c['title']) + ': ' + normalize(c['text']) for c in row['candidates']]
    if cell == 'WiCE4B':
        return [normalize(c['text']) for c in row['candidates']]
    raise ValueError(cell)


def tie_keys(cell, row):
    if cell == 'HoVer4B':
        keys = [int(c['raw_candidate_index']) for c in row['candidates']]
    else:
        keys = list(range(len(row['candidates'])))
    if len(set(keys)) != len(keys):
        raise ValueError('Nonunique candidate tie keys')
    return keys


class Evidence:
    def __init__(self, cell, row):
        self.cell = cell
        self.n = len(row['candidates'])
        self.texts = candidate_texts(cell, row)
        if cell.startswith('FEVER'):
            grouped = row['gold_evidence_raw']['gold snippets groups']
            self.routes = [frozenset(normalize(s) for s in r) for r in grouped if r]
            self.values = self.texts
        elif cell == 'HoVer4B':
            self.routes = [frozenset(int(s) for s in r) for r in row['sufficient_sets'] if r]
            self.values = list(range(self.n))
        elif cell == 'WiCE4B':
            self.routes = [frozenset(int(s) for s in r) for r in row['sufficient_sets'] if r]
            self.values = list(range(self.n))
        else:
            raise ValueError(cell)
        if any(not r <= set(self.values) for r in self.routes):
            raise ValueError('Evidence set absent from frozen candidate pool')

    @lru_cache(maxsize=None)
    def margin(self, retained):
        retained = tuple(sorted(retained))
        present = {self.values[j] for j in retained}
        alive = [route for route in self.routes if route <= present]
        if not alive:
            return 0
        coverage = {item: sum(1 << k for k, r in enumerate(alive) if item in r)
                    for item in set().union(*alive)}
        costs = {item: sum(self.values[j] == item for j in retained) for item in coverage}
        masks = list(coverage.values())
        vals = list(costs.values())
        full = (1 << len(alive)) - 1

        @lru_cache(None)
        def solve(uncovered):
            if not uncovered:
                return 0
            bit = uncovered & -uncovered
            return min(vals[j] + solve(uncovered & ~masks[j])
                       for j in range(len(masks)) if masks[j] & bit)

        return int(solve(full))

    def core_hit(self, retained):
        if not self.routes:
            return None
        core = set.intersection(*(set(s) for s in self.routes))
        current = {self.values[j] for j in retained}
        return int(bool(core - current))
