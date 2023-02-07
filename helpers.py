def get_best(file, d=768):
    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                if int(l.strip().split()[7]) <= d:
                    results.append(
                        (float(l.strip().split()[3]), float(l.strip().split()[4]), int(l.strip().split()[7]))
                    )
        return max(results)
    except (FileNotFoundError, ValueError):
        return None, None, None


def get_best_full(file, d=768):
    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                if int(l.strip().split()[7]) <= d:
                    results.append((
                        float(l.strip().split()[3]),
                        float(l.strip().split()[4]),
                        float(l.strip().split()[5]),
                        float(l.strip().split()[6]),
                        float(l.strip().split()[0]),
                        float(l.strip().split()[1]),
                        float(l.strip().split()[2]),
                        int(l.strip().split()[7])
                    ))
        return max(results)
    except (FileNotFoundError, ValueError):
        return None, None, None
