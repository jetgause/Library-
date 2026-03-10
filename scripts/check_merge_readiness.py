from pathlib import Path
import sys


def main() -> int:
    offenders = []
    for path in Path('.').rglob('*'):
        if not path.is_file():
            continue
        if '.git' in path.parts or '__pycache__' in path.parts:
            continue
        if path.suffix.lower() in {'.db', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pyc'}:
            continue

        try:
            lines = path.read_text(errors='ignore').splitlines()
        except Exception:
            continue

        for idx, line in enumerate(lines, start=1):
            if line.startswith('<<<<<<< '):
                offenders.append(f"{path}:{idx}:<<<<<<<")
            elif line == '=======':
                offenders.append(f"{path}:{idx}:=======")
            elif line.startswith('>>>>>>> '):
                offenders.append(f"{path}:{idx}:>>>>>>>")

    if offenders:
        print('Unresolved merge conflict markers found:')
        for item in offenders:
            print(item)
        return 1

    print('No unresolved merge conflict markers found.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
