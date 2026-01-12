import click
import orjson
from pathlib import Path
import cv2
from datasets.data_utils import get_bbox_from_mask,expand_bbox,box2squre,sobel

def crop_content(image,mask,ratio=1.2):
    box=get_bbox_from_mask(mask)
    box=box2squre(image,box)
    image_crop=image[box[0]:box[1],box[2]:box[3],:]
    mask_crop=mask[box[0]:box[1],box[2]:box[3]]
    return image_crop,mask_crop,box

def inference_single_image(normal_img,normal_mask,reference_img,reference_mask):
    reference_crop,reference_mask_crop,crop_box=crop_content(reference_img,reference_mask)
    reference_crop=cv2.resize(reference_crop,(224,224))
    reference_mask_crop=cv2.resize(reference_mask_crop,(224,224))
    reference_collage=sobel(reference_crop,reference_mask_crop/255,thresh=25)
    cv2.imwrite("./output/temp.jpg",reference_collage)
    print("done")


@click.command() 
@click.option("--config_path", type=str, help="Normal image path",default="/tmp/code/screw_pictures/loc1/merge.json")
def infer(config_path):
    with open(config_path,"rb") as f:
        config=orjson.loads(f.read())
    config_dir=Path(config_path).parent
    output_dir=config_dir.joinpath("merge")
    output_dir.mkdir(exist_ok=True,parents=True)
    prompt=config["prompt"]
    task_id=0
    crop_size=512
    for task in config["tasks"]:
        normal_img=cv2.imread(str(config_dir.joinpath(task["normal_img"])))
        normal_img=cv2.cvtColor(normal_img,cv2.COLOR_BGR2RGB)
        normal_mask=cv2.imread(str(config_dir.joinpath(task["normal_mask_img"])),cv2.IMREAD_GRAYSCALE)
        reference_img=cv2.imread(str(config_dir.joinpath(task["reference_img"])))
        reference_img=cv2.cvtColor(reference_img,cv2.COLOR_BGR2RGB)
        reference_mask=cv2.imread(str(config_dir.joinpath(task["reference_mask_img"])),cv2.IMREAD_GRAYSCALE)
        inference_single_image(normal_img,normal_mask,reference_img,reference_mask)
    
if __name__=="__main__":
    infer()