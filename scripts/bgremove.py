import modules.scripts as scripts
import gradio as gr
import modules.processing as P
import scripts.bgutils as bgutils
from scripts.Remover import Remover
from PIL import ImageFilter, Image as ImageModule

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
        enabled = gr.Checkbox(label='Background Remove: enable', value=False, elem_id=self.elem_id("enabled"))
        return [enabled]
    
    def process(self, p: P.StableDiffusionProcessingImg2Img, enabled):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """        
        print("[BgRemove.process]", enabled,  p)
        if not enabled:
            return p

        input: ImageModule.Image = p.init_images[0]
        # input_image = getattr(p, "init_images", [None])[0]
        if input is None:
            # if batch_hijack.instance.is_batch:
            #     shared.state.interrupted = True
            raise ValueError('[BgRemove] BgRemove is enabled but no input image is given')

        ### from sticker_process.run() ###
        output = bgutils.resize(input, 1280)
        output = bgutils.gamma(output, 0.8, 0.95)
        # output = bgutils.remove_background_1(output)
        output = ImageModule.fromarray(self.remover.process(output))
        output = bgutils.crop(output)
        # output = resize(output, 800)
        # output = round_to_module(output, 8)
        # output = add_margin(output, padding)
        output = bgutils.fit_to_size(output, 768) #(255,0,255)
        # output = ImageModule.alpha_composite(ImageModule.new('RGBA', output.size, (255, 255, 255)), output)
        output = output.convert('RGB')
        output = output.filter(ImageFilter.UnsharpMask(3, 150, 2))
        
        p.init_images[0] = output
        return p