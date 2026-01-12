import click
import orjson
from pathlib import Path
import cv2
from datasets.data_utils import get_bbox_from_mask,expand_bbox,box2squre,sobel,box_in_box

def crop_content(image,mask,ratio=1.2):
    box=get_bbox_from_mask(mask)
    box=box2squre(image,box)
    image_crop=image[box[0]:box[1],box[2]:box[3],:]
    mask_crop=mask[box[0]:box[1],box[2]:box[3]]
    return image_crop,mask_crop,box

def get_normal_crop_box(normal_img,normal_box,crop_size=512):
    y1,y2,x1,x2=normal_box
    image_h,image_w=normal_img.shape[0],normal_img.shape[1]
    box_w=x2-x1
    box_h=y2-y1
    box_center_x=(x1+x2)//2
    box_center_y=(y1+y2)//2
    box_max_side=max(box_w,box_h)
    if box_max_side<=crop_size:
        x1=max(0,box_center_x-crop_size//2)
        x2=min(image_w,x1+crop_size)
        y1=max(0,box_center_y-crop_size//2)
        y2=min(image_h,y1+crop_size)
        return (y1,y2,x1,x2)
    else:
        max_side=2.0*box_max_side
        x1=max(0,box_center_x-max_side//2)
        x2=min(image_w,x1+max_side)
        y1=max(0,box_center_y-max_side//2)
        y2=min(image_h,y1+max_side)
        return (y1,y2,x1,x2)

def inference_single_image(normal_img,normal_mask,reference_img,reference_mask):
    reference_crop,reference_mask_crop,crop_box=crop_content(reference_img,reference_mask)
    reference_crop=cv2.resize(reference_crop,(224,224))
    reference_mask_crop=cv2.resize(reference_mask_crop,(224,224))
    reference_collage=sobel(reference_crop,reference_mask_crop/255,thresh=10)
    #正常图片
    normal_box=get_bbox_from_mask(normal_mask)
    reference_collage=cv2.resize(reference_collage,(normal_box[3]-normal_box[2],normal_box[1]-normal_box[0]))
    normal_crop_box=get_normal_crop_box(normal_img,normal_box)
    normal_crop_img=normal_img[normal_crop_box[0]:normal_crop_box[1],normal_crop_box[2]:normal_crop_box[3],:]
    collage=normal_crop_img.copy()
    collage_box = box_in_box(normal_box, normal_crop_box)
    collage[collage_box[0]:collage_box[1],collage_box[2]:collage_box[3],:]=reference_collage
    collage_mask = normal_crop_img.copy() * 0.0
    collage_mask[collage_box[0]:collage_box[1],collage_box[2]:collage_box[3],:] = 1.0
    cv2.imwrite("./output/temp.jpg",collage)
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