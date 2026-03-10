from pathlib import Path


def test_repository_has_no_git_conflict_markers():
    root = Path('.')
    offenders = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if '.git' in path.parts or '__pycache__' in path.parts:
            continue
        if path.suffix in {'.db', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pyc'}:
            continue

        try:
            lines = path.read_text(errors='ignore').splitlines()
        except Exception:
            continue

        for idx, line in enumerate(lines, start=1):
            if line.startswith('<<<<<<< '):
                offenders.append(f"{path}:{idx}:start")
            elif line == '=======':
                offenders.append(f"{path}:{idx}:mid")
            elif line.startswith('>>>>>>> '):
                offenders.append(f"{path}:{idx}:end")

    assert not offenders, "Unresolved merge conflict markers found:\n" + "\n".join(offenders)
