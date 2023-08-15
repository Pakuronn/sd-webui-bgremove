import base64
from io import BytesIO
from typing import TypedDict
import numpy as np
import gradio as gr
from fastapi import FastAPI, Body, Response
from PIL import ImageFilter, Image as ImageModule, ImageStat, ImageOps, ImageChops
import traceback
import sys
import platform

from modules.api.models import *
from modules.api import api, models
from modules.call_queue import queue_lock  # noqa: F401

# from scripts.logging import logger
from scripts.Remover import Remover
from scripts.bgutils import detect_faces, gamma, resize, crop, fit_to_size
from scripts.face_detect import FaceDetectConfig, FaceMode, findFaces, findFaces2

HOSTNAME_PENS = 'DESKTOP-0PN01Q5'
IS_PROD = platform.node() != HOSTNAME_PENS

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__ #type: ignore
    __delattr__ = dict.__delitem__ #type: ignore

def nparray_to_base64(image: np.ndarray):
    return pil_to_base64(ImageModule.fromarray(image))

def pil_to_base64(image: ImageModule.Image):
    buffered = BytesIO()
    format = 'PNG' if image.mode == 'RGBA' else 'JPEG'
    print('[pil_to_base64] format', format)
    image.save(buffered, format=format)
    return str(base64.b64encode(buffered.getvalue()), 'UTF-8')

def base64_to_pil(b64: str):
    return api.decode_base64_to_image(b64)

def base64_to_nparray(b64: str):
    # Convert a base64 image into the image type the extension uses
    return np.array(api.decode_base64_to_image(b64)).astype('uint8')

# txt2img_request = {
#     "batch_size": 1,
#     "cfg_scale": 7,
#     "denoising_strength": 0,
#     "enable_hr": False,
#     "eta": 0,
#     "firstphase_height": 0,
#     "firstphase_width": 0,
#     "height": 64,
#     "n_iter": 1,
#     "negative_prompt": "",
#     "prompt": "example prompt",
#     "restore_faces": False,
#     "s_churn": 0,
#     "s_noise": 1,
#     "s_tmax": 0,
#     "s_tmin": 0,
#     "sampler_index": "Euler a",
#     "seed": -1,
#     "seed_resize_from_h": -1,
#     "seed_resize_from_w": -1,
#     "steps": 3,
#     "styles": [],
#     "subseed": -1,
#     "subseed_strength": 0,
#     "tiling": False,
#     "width": 64,
#     "script_name": "", #!
#     prompt: prompt
# }

# models.StableDiffusionTxt2ImgProcessingAPI
# request: StableDiffusionProcessingTxt2Img & [
#     {"key": "sampler_index", "type": str, "default": "Euler"},
#     {"key": "script_name", "type": str, "default": None},
#     {"key": "script_args", "type": list, "default": []},
#     {"key": "send_images", "type": bool, "default": True},
#     {"key": "save_images", "type": bool, "default": False},
#     {"key": "alwayson_scripts", "type": dict, "default": {}},
# ]

def smoothclamp(x: float, mi: float, mx: float): 
    return mi + (mx-mi) * (lambda t: np.where(t<0, 0, np.where(t<=1, 3*t**2-2*t**3, 1)))((x-mi)/(mx-mi))

def is_grayscale_masked(input: ImageModule.Image, mask: ImageModule.Image):
    treshold: float = 5
    stat = ImageStat.Stat(input, mask)
    mask_stat = ImageStat.Stat(mask)
    area = mask_stat.sum[0] / 255
    r: float = stat.sum[0] / area
    g: float = stat.sum[1] / area
    b: float = stat.sum[2] / area
    diff = max(abs(r-b), abs(r-g), abs(g-b))
    # print('[is_grayscale_masked]', area)
    # print(round(r,2), round(g,2), round(b,2), round(diff,2), treshold)
    return True if diff < treshold else False

def bgremove_api(_: gr.Blocks, app: FastAPI):

    DETECT_IMAGE_SIZE = 1280
    STICKER_IMAGE_SIZE = 768
    PASS1_IMAGE_SIZE = 768
    PASS2_IMAGE_SIZE = 1024 if IS_PROD else 1024

    print('[bgremove_api.init] IS_PROD = ', IS_PROD, PASS1_IMAGE_SIZE, PASS2_IMAGE_SIZE)

    sdapi = api.Api(app, queue_lock)
    remover = Remover()

    @app.get("/bgremove/version")
    async def version():
        return {"version": '1.1.0'}

    class AvatarResponse(TypedDict, total=False):
        images: list[str]
        seed: int
        _req: Any
        _err: Any


    ##############


    def do_pass1(
        scale: float, 
        prompt: str, 
        negative_prompt: str, 
        image_sd: ImageModule.Image, 
        image_cn: ImageModule.Image,
        image_mask: ImageModule.Image|None,
        sd: dict[str, Any], 
        controlnet: dict[str, Any],
        face_height: int,
    ):
        ### make img2img call:
        # "inpaint_full_res": true,
        # "inpaint_full_res_padding": 0,
        print('[do_pass1]', image_sd.size, image_cn.size, '-' if image_mask == None else image_mask.size)
        width, height = image_sd.size
        req = models.StableDiffusionImg2ImgProcessingAPI()
        req.init_images = [pil_to_base64(image_sd)]
        req.cfg_scale = 7
        # req.width = width
        # req.height = height
        req.width = round(width * scale / 8) * 8
        req.height = round(height * scale / 8) * 8
        req.prompt = prompt #prompt_patched
        req.negative_prompt = negative_prompt
        req.steps = sd['steps'] if 'steps' in sd else 25
        req.sampler_index = sd['sampler_index'] if 'sampler_index' in sd else 'Euler a'
        req.denoising_strength = 1 if 'denoising_strength' not in sd else sd['denoising_strength']
        req.cfg_scale = 7 if 'cfg_scale' not in sd else sd['cfg_scale']
        if 'seed' in sd:
            req.seed = sd['seed']

        ### set inpaint mode if needed
        if image_mask != None:
            req.mask = pil_to_base64(image_mask)
            req.mask_blur = round(face_height * 0.15)
            req.inpainting_fill = 1 # !!! FIXED
            req.inpainting_mask_invert = 1

        req.alwayson_scripts = {
            "ControlNet": {
                "args": [{
                    "image": pil_to_base64(image_cn),
                    "module": "softedge_hed",
                    "model": "control_v11p_sd15_softedge [a8575a2a]", #"control_sd15_hed [fef5e48e]",
                    "weight": 0.5,
                    "control_mode": 0,
                    "processor_res": 512,
                } | controlnet]
            }
        }

        ### call img2img
        print('[/bgremove/avatar] pass1', req.prompt)
        result1: ImageToImageResponse = sdapi.img2imgapi(req)
        pass1_output = result1.images[0]    
        # print('[make_pass1] parameters', result1.parameters)
        seed = result1.parameters['seed']
        print('[make_pass1] seed', seed)
        # print('[make_pass1] info', result1.info)         
        return pass1_output, seed

    def do_pass2(
        scale: float, 
        prompt: str, 
        face_height: int, 
        input_image: ImageModule.Image, 
        pass2_input_b64: str,
        seed: int | None
    ):
        """
        8 k, sharp focus, cinematic lighting, concept art, comic style, digital painting, (intricate details:0.9), (hdr, hyperdetailed:1.2)
        Negative prompt: cyborg, robot eyes, crossed eyes, tattoos, cinematic, grayscale, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, disgusting, blurry, amputation, ugly
        Steps: 25, Sampler: Euler a, CFG scale: 7, Model hash: 9aba26abdf, Model: deliberate_v2, Denoising strength: 0.039, ControlNet: "preprocessor: softedge_hed, model: control_v11p_sd15_softedge [a8575a2a], weight: 1, control mode: ControlNet is more important, preprocessor params: (512, -1, -1)"
        """

        ### calculate denoising_strength
        #denoising_strength = 0.039 + (0.05-0.039) * smoothclamp(face_height / height, .25, .35)

        ### make img2img call:
        width, height = input_image.size
        req2 = models.StableDiffusionImg2ImgProcessingAPI()
        req2.prompt = prompt #'8 k, sharp focus, cinematic lighting, concept art, comic style, digital painting, (intricate details:0.9), (hdr, hyperdetailed:1.2)'
        req2.negative_prompt = 'cyborg, robot eyes, crossed eyes, tattoos, cinematic, grayscale, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, disgusting, blurry, amputation, ugly'
        req2.cfg_scale = 7
        req2.sampler_index = 'DPM++ 2M Karras' #pens: 'DPM++ 2M SDE Karras'
        req2.width = round(width * scale / 8) * 8
        req2.height = round(height * scale / 8) * 8
        is_small = max(req2.width, req2.height) < 800
        req2.denoising_strength = 0.2 if is_small else 0.25
        req2.steps = 10
        req2.init_images = [pass2_input_b64]
        # req2.restore_faces = True
        if seed:
            req2.seed = seed

        ### call img2img
        print('[/bgremove/avatar] pass2', round(100*(face_height / width)), req2.prompt)
        result2 = sdapi.img2imgapi(req2)
        pass2_output = result2.images[0]
        return pass2_output

    def make_faces_mask(input_image_pil: ImageModule.Image, size: int):
        print("[/bgremove/avatar] build face mask")
        width, height = input_image_pil.size
        facecfg = FaceDetectConfig(FaceMode.YUNET)
        npmasks, faces_bbox, _ = findFaces2(facecfg, input_image_pil, True, height * 0.1)
        faces_count = len(faces_bbox)
        if len(npmasks) > 0:
            mask_pil = resize(ImageModule.fromarray(npmasks[0]), size)
        else:
            mask_pil = None
        print("[/bgremove/avatar] faces_bbox", faces_bbox)
        if faces_count > 0:
            biggest_face = sorted(faces_bbox, key=lambda x: x[3], reverse=True)[0]
            face_height = biggest_face[3] / DETECT_IMAGE_SIZE * size
        else:
            face_height = 0
        return mask_pil, faces_count, face_height
    

    @app.post("/bgremove/avatar2")
    def avatar2(
        response: Response,
        input_image: str = Body("", title='Input Image'),
        prompt: str = Body("man, steampunk", title="Prompt"),
        prompt_pass2: str = Body("", title="Prompt for Pass-2"),
        negative_prompt: str = Body("", title="Prompt"),
        sd: dict[str, Any] = Body({}, title="Stable Diffusion text2img parameters"),
        controlnet: dict[str, Any] = Body({}, title="ControlNet parameters"),
        bg_remove: bool = Body(False, title="Inpaint faces"),
        bg_blur_radius: int = Body(0, title="Blur Radius"),
        faces_restore: bool = Body(False, title="Restore faces from source photo"),
        debug: bool = Body(True, title="Return intermediate images"),
        detect_gender: bool = Body(False, title="Fine-tune prompt based on detected gender and age"),
        output_size: int = Body(PASS2_IMAGE_SIZE, title="Output size (bigger side, default is 1024)")
    ) -> AvatarResponse:
        try:
            # input_image_pil = ImageModule.fromarray(base64_to_nparray(input_image))
            pass1_size = 768 if output_size < 1200 else 960
            print('[/bgremove/avatar] STARTED', pass1_size, output_size, prompt_pass2)
            input_image_pil = base64_to_pil(input_image)
            input_image_pil = resize(input_image_pil, DETECT_IMAGE_SIZE)

            if bg_remove: 
                # sticker:
                # remove background
                input_image_pil = gamma(input_image_pil, 0.8, 0.95)
                input_image_pil = ImageModule.fromarray(remover.process(input_image_pil))
                input_image_pil = crop(input_image_pil)
                input_image_pil = fit_to_size(input_image_pil, STICKER_IMAGE_SIZE, (255,255,255,0))
                bgmask_pil = input_image_pil.getchannel(3)
                input_image_pil = input_image_pil.convert('RGB')
            else:
                bgmask_pil = None

            ### build face mask for inpaint
            if faces_restore:
                mask_pil, faces_count, face_height = make_faces_mask(input_image_pil, pass1_size)
            else:
                mask_pil, faces_count, face_height = None, 0, 0

            pass2_needed = faces_restore and mask_pil and faces_count > 0

            ### fine-tune prompt
            if faces_count > 0 and detect_gender:
                print('[/bgremove/avatar] fine-tune prompt', faces_count)
                prompt_faces, _ = detect_faces(input_image_pil)
                prompt_patched = '('+prompt_faces+'), ' + prompt if prompt_faces else prompt
            else:
                prompt_patched = prompt

            if bg_remove:
                # sticker:
                sd_image_pil = input_image_pil
                cn_image_pil = sd_image_pil
            elif bg_blur_radius > 0: 
                # avatar:
                print("[/bgremove/avatar] blur background for sd and controlnet")
                sd_image_pil = resize(ImageModule.fromarray(remover.process(input_image_pil, 'blur', bg_blur_radius)), pass1_size)
                cn_image_pil = sd_image_pil
                # cn_image_pil = cn_image_pil.filter(ImageFilter.UnsharpMask(3, 150, 2))
                if mask_pil:
                    mask_pil = resize(mask_pil, pass1_size)
            else: 
                # avatar:
                sd_image_pil = resize(input_image_pil, pass1_size)
                cn_image_pil = sd_image_pil
                if mask_pil:
                    mask_pil = resize(mask_pil, pass1_size)
            cn_image_b64 = pil_to_base64(cn_image_pil)

            ### PASS 1

            scale = 1 if pass2_needed or bg_remove else output_size / pass1_size
            pass1_output, pass1_seed = do_pass1(
                scale,
                prompt = prompt_patched, 
                negative_prompt = negative_prompt, 
                image_sd = sd_image_pil,
                image_cn = cn_image_pil,
                image_mask = mask_pil if mask_pil else None,
                sd = sd,
                controlnet = controlnet,
                face_height = face_height
            )

            output_image = pass1_output
            debug_images = [
                cn_image_b64, 
                "" if mask_pil == None else pil_to_base64(mask_pil)
            ]

            if pass2_needed: 
                ### PASS 2
                scale = 1 if bg_remove else output_size / pass1_size
                pass2_output = do_pass2(
                    scale, 
                    prompt if prompt_pass2 == "" else prompt_pass2, 
                    face_height, 
                    sd_image_pil, 
                    pass1_output, 
                    pass1_seed
                )
                ### make image grayscale if needed
                # faces_image = sd_image_pil if not mask_pil else ImageChops.multiply(sd_image_pil, mask_pil.convert('RGB'))
                if mask_pil and is_grayscale_masked(sd_image_pil, mask_pil):
                    pass2_output = ImageOps.grayscale(base64_to_pil(pass2_output))
                    pass2_output = pil_to_base64(pass2_output)
                output_image = pass2_output
                debug_images = [pass1_output] + debug_images

            if bg_remove and bgmask_pil:
                output_image_pil = base64_to_pil(output_image).convert('RGBA')
                # output_image = nparray_to_base64(remover.process(output_image_pil))
                output_image_pil.putalpha(bgmask_pil)
                output_image = pil_to_base64(output_image_pil)
                debug_images = [pil_to_base64(bgmask_pil)] + debug_images

            return {
                "images": [output_image] + debug_images if debug else [output_image],
                "seed": pass1_seed
            }

        except Exception as err:
            response.status_code = 500
            exc_info = sys.exc_info()
            print('[/bgremove/avatar] catchedError', exc_info, traceback.format_exception(*exc_info))
            return {
                "_err": traceback.format_exception(*exc_info)
            }

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(bgremove_api)
except:
    pass
