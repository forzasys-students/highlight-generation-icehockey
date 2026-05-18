import argparse
import subprocess


def positive_float(value: str) -> float:
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be greater than 0")
    return value


def non_negative_float(value: str) -> float:
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Must be greater than or equal to 0")
    return value

def positive_int(value: str) -> int:
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be greater than 0")
    return value

def non_negative_int(value: str) -> int:
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Must be greater than or equal to 0")
    return value

def probability_float(value: str) -> float:
    value = float(value)
    if not(0 < value <= 1):
        raise argparse.ArgumentTypeError("Must be greater than 0 and less than or equal to 1")
    return value


def get_supported_xfade_transitions() -> set[str]:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-h", "filter=xfade"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return None
    
    transitions = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            transitions.add(parts[0])
    
    return transitions


def validate_cli_args(args, parser):
    if args.transition_type != "none":
        supported_transitions = get_supported_xfade_transitions()
        if supported_transitions is not None and args.transition_type not in supported_transitions:
            supported_text = "\n  ".join(sorted(supported_transitions))
            parser.error(
                f"Unsupported xfade transition_type '{args.transition_type}'.\n"
                f"Supported xfade transition types are:\n  {supported_text}"
            )
