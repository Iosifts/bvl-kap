import argparse
import sys

from util import get_conf


def _get_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Generate image captions for the blind and visually limited (BVL) '
            'and grounding data for more accurate captioning.'
        ),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--base-path', metavar="PATH",
        help=(
            'store image file paths as relative to %(metavar)s in grounding '
            'data output. Allows for conflicting file names. If %(metavar)s '
            'is "auto", select the longest common path. (default: %(default)s)'
        ),
        dest='base_path', default="auto",
    )
    group.add_argument(
        '--no-base-path',
        help=(
            'only store image file name in grounding data output. Fails if '
            'multiple (non-identical) files with the same name are specified.'
        ),
        dest='base_path', action='store_const', const=None,
    )

    format_argument = parser.add_argument(
        '--format-json',
        help=('write output data in json format (default: %(default)s)'),
        dest='output_format', action='store_const', const='json', default='csv',
    )
    default_output_file = get_conf("DEFAULT_OUTPUT_FILE")
    parser.add_argument(
        '-o', '--output',
        help=(
            f'output file for grounding data. (default: {default_output_file})'
        ),
    )

    parser.add_argument('images', nargs='+', help=(
        'image folders or files. If argument is a folder, all images (detected '
        'by exif type) contained in the folder or its subfolders will be '
        'considered. '
    ))

    default_output_format = format_argument.default

    def fix_default_output_file(args):
        if args.output is None:
            # set the output file to default
            args.output = args.default_output
            # check whether we should adjust the file extension
            if (
                    args.output_format != default_output_format
                    and args.output.endswith("." + default_output_format)
            ):
                args.output = args.output.removesuffix(
                    "." + default_output_format)
                args.output = args.output + "." + args.output_format
        return args

    parser.set_defaults(postprocess=fix_default_output_file,
                        default_output=default_output_file)

    return parser


if __name__ == "__main__":
    parser = _get_argument_parser()

    subparsers = parser.add_subparsers(
        help=('whether to generate grounding data or a caption for each of the'
              ' images.'),
    )
    ground = subparsers.add_parser('ground')
    ground.set_defaults(action="ground", default_output="groundings.csv")

    caption = subparsers.add_parser('caption')
    caption.set_defaults(
        action="caption",
        default_output=("captions_grounded.csv"
                        if "--grounding-data" in sys.argv else "captions.csv"),
    )
    caption.add_argument(
        '--grounding-data',
        help='Additional grounding data to use for caption generation.',
    )
    caption.add_argument(
        '--fewshot',
        help='Path to BVL few-shot exemplar JSON file for style guidance.',
    )
    caption.add_argument(
        '--cache-timeout', type=int,
        help='Insert delay to help clear cache between runs.',
    )

    args = parser.parse_args() # parser.parse_intermixed_args()
    args = args.postprocess(args)

    if args.action == "ground":
        from grounding_data import interactive_generate_groundings
        interactive_generate_groundings(args)
    elif args.action == "caption":
        from captioning import interactive_generate_captions
        interactive_generate_captions(args)
    else:
        raise ValueError(f'unknown command verb: {args.action}')
