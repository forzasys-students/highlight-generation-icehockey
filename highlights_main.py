import argparse
import pipeline
from validators import (
    positive_float,
    non_negative_float,
    probability_float,
    non_negative_int,
    validate_cli_args
)



def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--events_url", help="Events JSON URL")
    src.add_argument("--events_json", help="Local events JSON file")

    parser.add_argument("--model", required=True, help="Path to camera model checkpoint (e.g. ResNet, EfficientNet, MobileNet, ConvNeXt, ViT)")
    parser.add_argument("--out", default="recap.mp4")
    parser.add_argument("--workdir", default="work_recap")
    
    parser.add_argument("--export_xlsx", action="store_true", help="Export selected segments summary to XLSX")
    parser.add_argument("--keep_workdir", action="store_true", help="Keep temporary working directory")
    parser.add_argument("--transition_type", default="fade", help="Transition type: 'none' disables transitions, otherwise use a transition supported by ffmpeg xfade (see https://trac.ffmpeg.org/wiki/Xfade)")
    
    parser.add_argument("--highlight_sec", type=positive_float, default=None, help="Target duration for extra highlight material in seconds")
    parser.add_argument("--transition_sec", type=positive_float, default=0.5, help="Xfade duration in seconds")
    parser.add_argument("--min_valid_core_main_sec", type=positive_float, default=6.0, help="Minimum duration in seconds for a detected core_main run")
    parser.add_argument("--tolerate_nonmain_sec", type=non_negative_float, default=1.2, help="Allowed non-main interruption inside candidate core_main")
    parser.add_argument("--min_segment_sec", type=positive_float, default=1.0, help="Minimum segment duration in seconds")
    parser.add_argument("--pad_ms", type=non_negative_int, default=300, help="Padding around logo cuts in milliseconds")
    parser.add_argument("--sbd_threshold", type=probability_float, default=0.7, help="Shot boundary detection threshold")
    parser.add_argument("--min_gap_ms", type=non_negative_int, default=200, help="Minimum gap between detected boundaries in milliseconds")
    parser.add_argument("--edge_guard_ms", type=non_negative_int, default=160, help="Trim segment edges to remove 1-frame inserts / noisy boundaries")
    parser.add_argument("--core_main_nonmain_ms", type=non_negative_int, default=1500, help="End core/main after this much continuous non-main footage; later main shots go to core_after")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    validate_cli_args(args=args, parser=parser)
    
    pipeline.run(args)


if __name__ == "__main__":
    main()
