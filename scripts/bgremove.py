import modules.scripts as scripts
import gradio as gr
import modules.processing as P
import scripts.bgutils as bgutils
from scripts.Remover import Remover
from PIL import ImageFilter, Image as ImageModule
# from deepface import DeepFace
import numpy as np
import torch

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
        detect_faces = gr.Checkbox(label='Background Remove: detect faces', value=False, elem_id=self.elem_id("detect_faces"))
        bg_blur = gr.Checkbox(label='Background Remove: blur background', value=False, elem_id=self.elem_id("bg_blur")) 
        return [do_preprocess, do_postprocess, detect_faces, bg_blur]
    
    def process(self, p: P.StableDiffusionProcessingImg2Img, do_preprocess, do_postprocess, detect_faces, bg_blur):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """        
        print("[BgRemove.process]", do_preprocess, detect_faces, bg_blur, p)

        input: ImageModule.Image = p.init_images[0]
        # input_image = getattr(p, "init_images", [None])[0]
        if input is None:
            # if batch_hijack.instance.is_batch:
            #     shared.state.interrupted = True
            raise ValueError('[BgRemove] BgRemove is enabled but no input image is given')

        if do_preprocess:
            ### from sticker_process.run() ###
            output = bgutils.resize(input, 1280)
            output = bgutils.gamma(output, 0.8, 0.95)
            output = ImageModule.fromarray(self.remover.process(output))
            output = bgutils.crop(output)
            # output = resize(output, 800)
            # output = round_to_module(output, 8)
            # output = add_margin(output, padding)
            output = bgutils.fit_to_size(output, 768)
            output = output.convert('RGB')
            output = output.filter(ImageFilter.UnsharpMask(3, 150, 2))
            p.init_images[0] = output
        elif bg_blur:
            output = bgutils.resize(input, 1280)
            output = ImageModule.fromarray(self.remover.process(output, 'blur'))
            p.init_images[0] = output

        if detect_faces:
            """
            [{
                'age': 31, 
                'region': {'x': 1102, 'y': 346, 'w': 726, 'h': 726}, 
                'gender': {'Woman': 0.4312588833272457, 'Man': 99.56874251365662}, 
                'dominant_gender': 'Man', 
                'race': {'asian': 0.014595308454334495, 'indian': 0.006218922142182492, 'black': 0.0001256181196663502, 'white': 96.95498925179179, 'middle eastern': 1.7991946219917907, 'latino hispanic': 1.22487682699423}, 
                'dominant_race': 'white', 
                'emotion': {'angry': 0.26683781761676073, 'disgust': 0.004057566911797039, 'fear': 9.617278919904493e-05, 'happy': 1.4599894173443317, 'sad': 6.380016356706619, 'surprise': 0.05031790351495147, 'neutral': 91.83868765830994}, 
                'dominant_emotion': 'neutral'
            }]
            """
            print("[BgRemove.process] detect_faces", detect_faces)
            try:
                image = np.array(input)
                torch.cuda.empty_cache() # ???
                faces = DeepFace.analyze(img_path=image,  actions=['age', 'gender'], detector_backend="retinaface") # 'race', 'emotion'
                face = faces[0] #{}
                print(face)
                is_minor = face['age'] < 13
                male = 'kid boy' if is_minor else 'man'
                female = 'kid girl' if is_minor else 'woman'
                gender = male if face['dominant_gender'] == 'Man' else female
                # emotion = '' if face['dominant_emotion'] == 'neutral' else face['dominant_emotion']
                face_prompt = face['dominant_race']+' '+gender # emotion+' '+
                patched_prompt = '('+face_prompt+') ' + p.prompt
                p.prompt = patched_prompt
                p.prompt_for_display = patched_prompt
                print('[bgremove.deepface] face_prompt: ', p.prompt)
            except BaseException as e:
                print('[bgremove.deepface] Cannot detect face', e)

        return p

    def postprocess(self, p, res: P.Processed, do_preprocess, do_postprocess, detect_faces, bg_blur):
        print('[bgremove.postprocess] prompt: ', p.prompt)
        if do_postprocess:
            output = res.images[0]
            output = ImageModule.fromarray(self.remover.process(output))
            res.images = [output]
        return p
