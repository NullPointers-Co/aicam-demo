import os

import click

from detect import *
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
    # check is exists
    if not os.path.exists(target):
        print(f"Target file not found: {target}")
        return
    if not os.path.exists(reference):
        print(f"Reference file not found: {reference}")
        return

    click.echo(f"Processing target file: {target}")
    click.echo(f"Using reference file: {reference}")
    click.echo(f"Confidence threshold: {confidence}")

    target_path = target
    reference_path = reference
    target_mod = get_img_mod(target_path, conf=confidence)
    reference_mod = get_img_mod(reference_path, conf=confidence)

    target_img_with_box = target_mod.plot()
    # reference_img_with_box = reference_mod.plot()

    target_width, target_height = target_mod.orig_shape
    reference_width, reference_height = reference_mod.orig_shape

    # 打印检测结果
    print("Target:")
    format_output(target_mod)
    print("Reference:")
    format_output(reference_mod)

    for target_obj in target_mod.boxes:
        for reference_obj in reference_mod.boxes:
            if target_obj.cls[0] == reference_obj.cls[0]:
                target_x1, target_y1, target_x2, target_y2 = normalize_coordinates(*target_obj.xyxy[0], target_width, target_height)
                reference_x1, reference_y1, reference_x2, reference_y2 = normalize_coordinates(*reference_obj.xyxy[0], reference_width, reference_height)

                print(f"Target Box: [{target_x1}, {target_y1}, {target_x2}, {target_y2}]")
                print(f"Reference Box: [{reference_x1}, {reference_y1}, {reference_x2}, {reference_y2}]")
                print(f"x1 move: {reference_x1 - target_x1}", f"y1 move: {reference_y1 - target_y1}",
                      f"x2 move: {reference_x2 - target_x2}", f"y2 move: {reference_y2 - target_y2}")

                retangle_x1, retangle_y1, retangle_x2, retangle_y2 = denormalize_coordinates(reference_x1, reference_y1, reference_x2, reference_y2, target_width, target_height)
                cv2.rectangle(target_img_with_box,
                              (retangle_x1, retangle_y1), (retangle_x2, retangle_y2),
                              (0, 255, 0), 2)  # 绿色框，线条宽度为2

                # 保存带绿色框的图像
                origin_image_name, origin_image_ext = os.path.splitext(os.path.basename(target_path))
                target_dir = os.path.dirname(target_path)
                output_image_with_box_path = os.path.join(target_dir, f'd_{origin_image_name}{origin_image_ext}')
                cv2.imwrite(output_image_with_box_path, target_img_with_box)

            else:
                print(f"Match loss: {target_obj.cls[0]}")

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
        if os.path.isdir(out):
            input_file_name, ext = os.path.basename(input).split('.')
            if type == 'preview':
                out_file_name = os.path.join(out, f"p_{input_file_name}.{ext}")
            elif type == 'dataset':
                out_file_name = os.path.join(out, f"p_{input_file_name}.json")
        else:
            dirname = os.path.dirname(out)
            assert os.path.exists(dirname), f"Path not exists: {dirname}"
            out_file_name = out
        return out_file_name

    if not stdio:
        assert out, "Output file path is required"
        out_file_name = get_out_file_name(out)

    if not silent:
        click.echo(f"Processing input file: {input}")
        click.echo(f"Using weight file: {weight}")
        click.echo(f"Output to: {out_file_name if not stdio else 'Stdandard Output'}")

    if type == 'preview':
        preview(weight, input, out_file_name, origin=show_origin, box=show_box)
    elif type == 'dataset':
        dataset = mill(weight, input, verbose=not silent)
        if stdio:
            click.echo(dataset.model_dump_json())
        else:
            with open(out_file_name, 'w') as f:
                f.write(dataset.model_dump_json())

if __name__ == '__main__':
    cli()
