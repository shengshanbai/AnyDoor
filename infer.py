import click
import orjson
from pathlib import Path
import cv2
from datasets.data_utils import get_bbox_from_mask,expand_bbox,box2squre,sobel,box_in_box
import numpy as np
import torch
import einops
from omegaconf import OmegaConf
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

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
    #collage resize到512
    collage_resize=cv2.resize(collage,(512,512))
    collage_mask_resize=cv2.resize(collage_mask,(512,512))
    collage_resize=collage_resize.astype(np.float32)/127.5 - 1.0
    collage_mask_resize=(collage_mask_resize.astype(np.float32) > 0.5).astype(np.float32)
    collage_all = np.concatenate([collage_resize, collage_mask_resize[:,:,:1]  ] , -1)
    reference_crop=reference_crop.astype(np.float32)/255.0
    normal_crop_resize=cv2.resize(normal_crop_img,(512,512))
    normal_crop_resize = normal_crop_resize.astype(np.float32) / 127.5 - 1.0
    #开始推理
    num_samples = 1
    control = torch.from_numpy(collage_all.copy()).float().cuda()
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    
    clip_input = torch.from_numpy(reference_crop.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()
    
    guess_mode = False
    H,W = 512,512
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)
    
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = 9.0  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    pred=cv2.resize(pred,(normal_crop_resize.shape[1],normal_crop_resize.shape[0]))
    normal_img[normal_crop_box[0]:normal_crop_box[1],normal_crop_box[2]:normal_crop_box[3],:]=pred
    return normal_img

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
    for task in config["tasks"]:
        normal_img=cv2.imread(str(config_dir.joinpath(task["normal_img"])))
        normal_img=cv2.cvtColor(normal_img,cv2.COLOR_BGR2RGB)
        normal_mask=cv2.imread(str(config_dir.joinpath(task["normal_mask_img"])),cv2.IMREAD_GRAYSCALE)
        reference_img=cv2.imread(str(config_dir.joinpath(task["reference_img"])))
        reference_img=cv2.cvtColor(reference_img,cv2.COLOR_BGR2RGB)
        reference_mask=cv2.imread(str(config_dir.joinpath(task["reference_mask_img"])),cv2.IMREAD_GRAYSCALE)
        gen_image=inference_single_image(normal_img,normal_mask,reference_img,reference_mask)
        ouput_file=output_dir.joinpath(f"task_{task_id}.jpg")
        cv2.imwrite(str(ouput_file),cv2.cvtColor(gen_image,cv2.COLOR_RGB2BGR))
        task_id+=1
        
if __name__=="__main__":
    infer()