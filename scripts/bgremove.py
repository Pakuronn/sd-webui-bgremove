import modules.scripts as scripts
import gradio as gr
import modules.processing as P
import scripts.bgutils as bgutils
from scripts.Remover import Remover
from PIL import ImageFilter, Image as ImageModule
import numpy as np

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.remover = Remover()
        print("[BgRemove.__init__]", self)

    def title(self):
        return "BgRemove"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        do_preprocess = gr.Checkbox(label='Background Remove: pre-process', value=False, elem_id=self.elem_id("do_preprocess"))
        do_postprocess = gr.Checkbox(label='Background Remove: post-process', value=False, elem_id=self.elem_id("do_postprocess"))
        do_detect_faces = gr.Checkbox(label='Background Remove: detect faces', value=False, elem_id=self.elem_id("do_detect_faces"))
        return [do_preprocess, do_postprocess, do_detect_faces]
    
    def process(self, p: P.StableDiffusionProcessingImg2Img, do_preprocess, do_postprocess, do_detect_faces):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """        
        print("[BgRemove.process]", do_preprocess, do_detect_faces, p)

        input: ImageModule.Image = p.init_images[0]
        # input_image = getattr(p, "init_images", [None])[0]
        if input is None:
            # if batch_hijack.instance.is_batch:
            #     shared.state.interrupted = True
            raise ValueError('[BgRemove] BgRemove is enabled but no input image is given')

        if do_preprocess:
            print("[BgRemove.process] preprocess")
            ### from sticker_process.run() ###
            output = bgutils.resize(input, 1280)
            output = bgutils.gamma(output, 0.8, 0.95)
            output = ImageModule.fromarray(self.remover.process(output))
            output = bgutils.crop(output)
            # output = resize(output, 800)
            # output = round_to_module(output, 8)
            # output = add_margin(output, padding)
            output = bgutils.fit_to_size(output, 768).convert('RGB')
            output = output.filter(ImageFilter.UnsharpMask(3, 150, 2))
            p.init_images[0] = output
        # elif bg_blur:
        #     print("[BgRemove.process] bg_blur")
        #     output = bgutils.resize(input, 1280)
        #     output = ImageModule.fromarray(self.remover.process(output, 'blur', 15))
        #     output = output.filter(ImageFilter.UnsharpMask(3, 150, 2))
        #     p.init_images[0] = output
        else:
            output = input

        if do_detect_faces:
            print("[BgRemove.process] detect_faces")
            # faces = [
            #   {'age': 26, 'region': {'x': 219, 'y': 253, 'w': 366, 'h': 366}, 'gender': {'Woman': 0.05471727345138788, 'Man': 99.94527697563171}, 'dominant_gender': 'Man'}, 
            #   {'age': 34, 'region': {'x': 17, 'y': 459, 'w': 105, 'h': 105}, 'gender': {'Woman': 12.601463496685028, 'Man': 87.39854097366333}, 'dominant_gender': 'Man'}
            # ]
            prompt_faces, faces = bgutils.detect_faces(output) #(input_image_pil)
            prompt = '('+prompt_faces+'), ' + p.prompt if prompt_faces else p.prompt
            p.prompt = prompt
            p.prompt_for_display = prompt

        return p

    def postprocess(self, p, res: P.Processed, do_preprocess, do_postprocess, do_detect_faces):
        print('[bgremove.postprocess] prompt: ', p.prompt)
        if do_postprocess:
            output = res.images[0]
            output = ImageModule.fromarray(self.remover.process(output))
            res.images = [output]
        return p
