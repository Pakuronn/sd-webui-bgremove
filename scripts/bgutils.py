import math
from typing import Tuple
from PIL import Image as ImageModule, ImageFilter, ImageDraw #, ImageFont, ImageOps
from PIL.Image import Image
from deepface import DeepFace
import numpy as np
# import cv2 as cv
# import numpy

def add_margin(input: Image, padding):
    w0, h0 = input.size
    w1 = w0 + 2*padding
    h1 = h0 + 2*padding
    result = ImageModule.new(input.mode, (w1, h1), (255,255,255,0))
    result.paste(input, (padding, padding))
    return result

def add_margin4(input: Image, t, r, b, l):
    w0, h0 = input.size
    w1 = w0 + r + l
    h1 = h0 + t + b
    result = ImageModule.new(input.mode, (w1, h1), (255,255,255,0))
    result.paste(input, (t, l))
    return result

def round_to_module(input: Image, mod = 32):
    w0, h0 = input.width, input.height
    w1 = math.ceil(w0/mod)*mod
    h1 = math.ceil(h0/mod)*mod
    pl = math.floor((w1-w0)/2)
    pt = math.floor((h1-h0)/2)
    pr = w1-w0-pl
    pb = h1-h0-pt
    # print('#', pt,pr,pb,pl)
    return add_margin4(input, pt,pr,pb,pl)

def add_stroke_with_shadow(input, stroke_radius = 15, treshold = 30, shadow_radius = 15, color = (255, 255, 255, 255)):
    stroke_image = ImageModule.new("RGBA", input.size, color)
    img_alpha = input.getchannel(3).filter(ImageFilter.BoxBlur(5)).point(lambda x: 255 if x>treshold else 0)
    stroke_alpha = img_alpha.filter(ImageFilter.MaxFilter(stroke_radius))
    shadow_alpha = stroke_alpha.copy().filter(ImageFilter.BoxBlur(shadow_radius)).point(lambda x: x * .75)
    shadow_image = ImageModule.new("RGBA", input.size, 0x000000)
    shadow_image.putalpha(shadow_alpha)
    stroke_alpha = stroke_alpha.filter(ImageFilter.SMOOTH)
    stroke_image.putalpha(stroke_alpha)
    output = ImageModule.alpha_composite(stroke_image, input)
    output = ImageModule.alpha_composite(shadow_image, output)
    return output

def smoothstep (a, b, value):
    x = max(0, min(1, (value-a)/(b-a)))
    return x * x * (3 - 2*x)

# def add_stroke(input: Image, stroke_radius = 12, treshold = 30):
#     stroke_image = ImageModule.new("RGBA", input.size, (255, 255, 255, 255))

#     img_alpha = input.getchannel(3).point(lambda x: 255 if x>treshold else 0)
#     stroke_alpha = img_alpha.filter(ImageFilter.MaxFilter(5))
#     stroke_alpha = stroke_alpha.filter(ImageFilter.GaussianBlur(6)).point(lambda x: 255*smoothstep(treshold-5,treshold+5,x))

#     # img_alpha = input.getchannel(3).filter(ImageFilter.GaussianBlur(stroke_radius))
#     # stroke_alpha = img_alpha.point(lambda x: 255*smoothstep(treshold-5,treshold+5,x)) #255 if x>treshold else 0

#     stroke_alpha = stroke_alpha.filter(ImageFilter.SMOOTH)
#     stroke_image.putalpha(stroke_alpha)
#     output = ImageModule.alpha_composite(stroke_image, input)
#     return output

def add_shadow(input: Image, shadow_radius = 15):
    shadow_alpha = input.getchannel(3).copy().filter(ImageFilter.GaussianBlur(shadow_radius)).point(lambda x: x * .75)
    shadow_image = ImageModule.new("RGBA", input.size, 0x000000)
    shadow_image.putalpha(shadow_alpha)
    output = ImageModule.alpha_composite(shadow_image, input)
    return output

# def remove_background_1(input: Image):
#     # https://github.com/plemeri/transparent-background
#     remover = Remover()
#     output = remover.process(input)
#     return ImageModule.fromarray(output)

# def remove_background_chromakey(input: Image):
#     chroma_l = numpy.array([250, 0, 250])
#     chroma_u = numpy.array([255, 10, 255])
#     im = numpy.array(input)
#     im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
#     im = cv.medianBlur(im, 5)
#     im_bw = cv.bitwise_not(cv.inRange(im, chroma_l, chroma_u))
#     #ImageModule.fromarray(im_bw).save('./temp/debug.png')
    
#     contours, hierarchy = cv.findContours(im_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

#     contours_filtered = contours if len(contours) < 2 else list(filter(lambda c: cv.contourArea(c) > 4000, contours))
#     im_out = numpy.ones(im_bw.shape, dtype = numpy.uint8)
#     cv.drawContours(im_out, contours_filtered, -1, (255), -1)

#     alpha = ImageModule.fromarray(im_out)
#     output = input
#     output.putalpha(alpha)
#     return output

# def remove_background_white(input: Image, stroke_width = 0):
#     im = numpy.array(input.convert('L'))
#     im_filtered = cv.medianBlur(im, 5)
    
#     white_treshold = 240
#     white_idx = white_treshold
#     white_cnt = 0
#     hist = cv.calcHist([im_filtered], [0], None, [256], [0,256])
#     for idx,cnt in enumerate(hist):
#         if idx > white_treshold and cnt > white_cnt:
#             white_cnt = cnt
#             white_idx = idx
#     print('# histogram', hist[white_treshold:])
#     print('# histogram', white_idx, white_cnt)

#     # im_filtered = cv.bilateralFilter(imgray, 10, 20, 5)
#     ret, im_bw = cv.threshold(im_filtered, white_idx-2, 255, cv.THRESH_BINARY_INV)
#     contours, hierarchy = cv.findContours(im_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

#     contours_filtered = contours if len(contours) < 2 else list(filter(lambda c: cv.contourArea(c) > 4000, contours))
#     im_out = numpy.ones(im_bw.shape, dtype = numpy.uint8)
#     cv.drawContours(im_out, contours_filtered, -1, (255), -1)
#     if stroke_width > 0:
#         cv.drawContours(im_out, contours_filtered, -1, (255), stroke_width)

#     # cv.drawContours(im, contours, -1, (0,0,0), -1)
#     alpha = ImageModule.fromarray(im_out)
#     output = input
#     output.putalpha(alpha) #ImageOps.invert(alpha))
#     return output

"""
def remove_background_2(input):
    my_session = new_session("u2net_human_seg")
    return remove(
        input, # data
        False, # alpha_matting
        250,   # alpha_matting_foreground_threshold
        50,     # alpha_matting_background_threshold
        10,    # alpha_matting_erode_size
        my_session,  # session
        False,  # only_mask
        False  # post_process_mask
    )

def remove_background_3(input):
    my_session = new_session("u2net")
    return remove(
        input, # data
        False, # alpha_matting
        250,   # alpha_matting_foreground_threshold
        50,     # alpha_matting_background_threshold
        10,    # alpha_matting_erode_size
        my_session,  # session
        False,  # only_mask
        False  # post_process_mask
    )
"""
 
def crop(input: Image):
    # imageSize = input.size

    # remove alpha channel
    # invert_im = input.convert("RGB")

    # invert image (so that white is 0)
    # invert_im = ImageOps.invert(invert_im)
    imageBox = input.getchannel(3).point(lambda x: x if x > 64 else 0).getbbox()
    print('#', input.width, input.height)
    print('#', imageBox)
    cropped = input.crop(imageBox)
    print('#', cropped.width, cropped.height)
    cropped = add_margin(cropped, 20)
    print('#', cropped.width, cropped.height)
    return cropped

def round_corners(input: Image, rad):
    circle = ImageModule.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2 - 1, rad * 2 - 1), fill=255)
    alpha = ImageModule.new('L', input.size, 255)
    w, h = input.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    # im.putalpha(alpha)

    output = ImageModule.new(input.mode, input.size, (0,0,0,0))
    output.paste(input, (0, 0), alpha)
    return output

def resize(input: Image, size: int):
    w0, h0 = input.size
    if size == max(w0, h0):
        return input
    else:
        k = size / max(w0, h0)
        w1, h1 = round(w0 * k), round(h0 * k)
        return input.resize((w1, h1), ImageModule.LANCZOS)

def fit_to_size(input: Image, size: int, bg_color = (255,255,255,255)):
    output = resize(input, size)
    result = ImageModule.new(input.mode, (size,size), bg_color)
    x = (size - output.width) // 2
    y = (size - output.height) // 2
    if output.mode == 'RGBA':
        result.paste(output, (x,y), output.getchannel(3).point(lambda x: x if x>64 else 0))
    else:
        result.paste(output, (x,y))
    return result

def gamma(input: Image, a = 1.1, b = 0.9):
    return input.point(lambda x: ((x/255)**a)*255*b)

def face2prompt(face):
    is_minor = face['age'] < 13
    male = 'kid boy' if is_minor else 'man'
    female = 'kid girl' if is_minor else 'woman'
    gender = male if face['dominant_gender'] == 'Man' else female
    # emotion = '' if face['dominant_emotion'] == 'neutral' else face['dominant_emotion']
    return gender # emotion+' '+ # face['dominant_race']+' '+

def detect_faces(input: Image) -> Tuple[str, list]: 
    #torch.cuda.empty_cache() # ???
    try:
        input_np = np.array(input)
        faces = DeepFace.analyze(
            img_path=input_np,  
            actions=['age', 'gender'], 
            detector_backend="mtcnn",
            enforce_detection=True #False
        ) # 'race', 'emotion' # , detector_backend="retinaface" # ssd
        # print('[detect_faces]', faces)
        faces2 = sorted(faces, key=lambda x: x['region']['w'], reverse=True)[0:2]
        faces2.sort(key=lambda x: x['region']['x'])
        print('[detect_faces] faces2', faces2)
        # face_prompt = ' and '.join(map(face2prompt, faces))
        if len(faces2) == 2:
            face_prompt = face2prompt(faces2[0]) + ' on left and ' + face2prompt(faces2[1]) + ' on right'
        elif len(faces2) == 1:
            face_prompt = face2prompt(faces2[0])
        else:
            face_prompt = ''
        return (face_prompt, faces)
    except Exception as err:
        print('[detect_faces] ERROR: Couldn`t detect face', err)
        return ('', [])

# def sharpen(input: Image):
    # cv2::GaussianBlur(frame, image, cv::Size(0, 0), 3);
    # cv::addWeighted(frame, 1.5, image, -0.5, 0, image);

# def remove_white_background0(input: Image):
#     # BG Remover 3
#     inputArr = np.array(input)
#     myimage_hsv = cv2.cvtColor(inputArr, cv2.COLOR_BGR2HSV)
     
#     #Take S and remove any value that is less than half
#     s = myimage_hsv[:,:,1]
#     s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded
 
#     # We increase the brightness of the image and then mod by 255
#     v = (myimage_hsv[:,:,2] + 127) % 255
#     v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask
 
#     # Combine our two masks based on S and V into a single "Foreground"
#     foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
 
#     background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
#     background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
#     foreground=cv2.bitwise_and(inputArr, inputArr, mask=foreground) # Apply our foreground map to original image
#     finalimage = background+foreground # Combine foreground and background
    # return ImageModule.fromarray(finalimage, input.mode)
