import os

import click

from detect import detect_main
from milling_pose import mill
from preview_pose import preview

DEFALT_TARGET = 'images/target.jpeg'
DEFALT_RERERENCE = 'images/reference.jpeg'


@click.group()
def cli():
    pass


@cli.command()
@click.option('--target', '-t', type=click.Path(exists=True), default=DEFALT_TARGET, help='Target file to process')
@click.option('--reference', '-f', type=click.Path(exists=True), default=DEFALT_RERERENCE, help='Reference file for comparison')
@click.option('--confidence', '-c', type=float, default=0.35, help='Confidence threshold for object detection')
def advise(target, reference, confidence):
    click.echo(f"Processing target file: {target}")
    click.echo(f"Using reference file: {reference}")
    click.echo(f"Confidence threshold: {confidence}")

    detect_main(target, reference, confidence)


@cli.command()
@click.option('--type', '-t', type=click.Choice(['preview', 'dataset']), required=True, help='Type of processing')
@click.option('--out', '-o', type=click.Path(), help='Output file path')
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input file path')
@click.option('--silent', '-s', is_flag=True, help='Silent mode')
@click.option('--weight', type=click.Path(exists=True), help='Customized weight file path')
@click.option('--stdio', is_flag=True, help='output to stdio')
@click.option('--show-origin', is_flag=True, help='Preview origin point')
@click.option('--show-box', is_flag=True, help='Prevew detetct box')
def milling(type, stdio, out, input, weight, show_origin, show_box, silent):
    if not weight:
        weight = 'weights/yolov11n-pose.pt'
    assert os.path.exists(weight), "Weight file not found"

    def get_out_file_name(out):
        """
        Generate the output file name based on the provided output path.
        If the provided output path is a directory, the function constructs the output file name
        based on the type of file ('preview' or 'dataset') and the input file name. If the output
        path is not a directory, it ensures that the directory exists.
        """

        out_file_name = None

        if os.path.isdir(out):
            input_file_name, ext = os.path.basename(input).split('.')
            if type == 'preview':
                out_file_name = os.path.join(out, f"p_{input_file_name}.{ext}")
            elif type == 'dataset':
                out_file_name = os.path.join(out, f"p_{input_file_name}.json")
        else:
            dirname = os.path.dirname(out)
            assert os.path.exists(dirname), f"Path not exists: {dirname}"

        assert out_file_name, "Output file name not generated"
        return out_file_name

    if not stdio:
        assert out, "Output file path is required"
        out_file_name = get_out_file_name(out)
    else:
        out_file_name = 'Standard Output'

    if not silent:
        click.echo(f"Processing input file: {input}")
        click.echo(f"Using weight file: {weight}")
        click.echo(
            f"Output to: {out_file_name}")

    if type == 'preview':
        preview(weight, input, out_file_name, origin=show_origin,
                box=show_box, verbose=not silent)
    elif type == 'dataset':
        dataset = mill(weight, input, verbose=not silent)
        if stdio:
            click.echo(dataset.model_dump_json())
        else:
            with open(out_file_name, 'w') as f:
                f.write(dataset.model_dump_json())


if __name__ == '__main__':
    cli()
