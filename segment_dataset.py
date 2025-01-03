
import sys, argparse, os, glob, json, math, gc, pdb, time
import numpy as np, cv2, moviepy.editor as mpy, torch, tqdm
from matplotlib import colors
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

ACTION_TYPE_SEGMENT = "seg"
ACTION_TYPE_LOAD    = "load"
VIDEO_DEBUG         = None

class Segment_MyVideo():
  def __init__(self, path, output, file_index = None, model_type = "vit_h", sam_checkpoint = "sam_vit_h_4b8939.pth"):
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    self.source_path = path
    self.output_path = output
    self.file_index = file_index
    self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    self.sam.to(device=self.device)
    self.predictor = SamPredictor(self.sam)
    self.video_log = {}

  def __process_frame(self, frame, annotation):
    boxes = []

    for lb in annotation["labels"]:
      boxes.append([
                    lb["box2d"]["x1"], 
                    lb["box2d"]["y1"], 
                    lb["box2d"]["x2"], 
                    lb["box2d"]["y2"]
                  ])

    if len(boxes) > 0:
      self.predictor.set_image(frame)
      boxes = torch.tensor(boxes, device = self.predictor.device)
      transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, frame.shape[:2])
      masks, _, _ = self.predictor.predict_torch(
          point_coords=None,
          point_labels=None,
          boxes=transformed_boxes,
          multimask_output=False,
      )

      return masks.squeeze(dim = 1).cpu().numpy().astype(int)
    
    return None

  def __process_video(self, idx, id, frame_ann):
    masks_path = os.path.join(self.output_path, "{}.npz".format(id))

    if os.path.isfile(masks_path):
      print("[{}] video was previously done".format(id))
      return

    video = mpy.VideoFileClip(os.path.join(self.source_path, "videos", "{}.webm".format(id)))
    available_idx = np.array([ int(os.path.basename(fr['name']).split('.')[0]) for fr in frame_ann ])
    total_frame = math.ceil(video.fps * video.duration)
    frame_ann_expand = []
    ## Put together frames with and without annotation
    for i in range(1, total_frame + 1):
      i_pos = np.where(i == available_idx)[0]

      if i_pos.shape[0] == 0:
        frame_ann_expand.append({"name": os.path.join(id, str(i)), "labels": []})

        if not id in self.video_log.keys():
          self.video_log[id] = []    

        self.video_log[id].append(i)
      else:
        frame_ann_expand.append(frame_ann[i_pos[0]])        

    self.do_process_video(idx, id, frame_ann_expand, total_frame, video, masks_path)

  def do_process_video(self, idx, id, frame_ann_expand, total_frame, video, masks_path):
    frame_masks = []

    for j, frame in tqdm.tqdm(enumerate(video.iter_frames()), total = total_frame, desc = "|-- ({}) Video {}".format(idx, id)):
      annotation = frame_ann_expand[j]
      masks      = self.__process_frame(frame, annotation)

      if masks is not None:
        frame_masks.append({"id": j + 1, "masks": masks})     

    if len(frame_masks) > 0:
      np.savez(masks_path, frame_masks=frame_masks)

  def run(self):
    bbox = glob.glob(os.path.join(self.source_path, "SomethingElse", "bounding_box*json"))
    bbox.sort()

    if self.file_index is not None:
      bbox = [bbox[self.file_index]]

    for bb in bbox:
      print("|- Loading {}".format(bb))
      annotations = json.load(open(bb,'r'))

      for idx, video_id in enumerate(annotations):
        video_id = video_id if VIDEO_DEBUG is None else VIDEO_DEBUG
        self.__process_video(idx, video_id, annotations[video_id])

        if VIDEO_DEBUG is not None:
          break

      if VIDEO_DEBUG is not None:
        break

    if len(self.video_log) > 0:
      log = open(os.path.join(self.output_path, "video_frames_without_label{}.json".format( "" if self.file_index is None else self.file_index  )), 'w')
      log.write(json.dumps(self.video_log))      

class Load_Segment():
  def __init__(self, path, output, mask):
    self.video_path  = path
    self.mask_path   = mask
    self.output_path = output

## https://stackoverflow.com/questions/66095686/apply-a-segmentation-mask-through-opencv
  def do_apply_masks(self, id, ext, video, masks):
    masks = masks["frame_masks"]
    total_frame = math.ceil(video.fps * video.duration)
    available_idx = np.array([ mk['id'] for mk in masks   ])
    new_frames = []
    color = np.array([0, 255, 0], dtype='uint8')
    alpha = 0.7

    for idx, frame in tqdm.tqdm(enumerate(video.iter_frames()), total = total_frame, desc = "|-- Video {}".format(id)):
      i_pos = np.where(idx + 1 == available_idx)[0]

      if i_pos.shape[0] > 0:
        for mk in masks[i_pos[0]]["masks"]:
          frame_mask = np.where(mk[..., None], color, frame)
          frame      = cv2.addWeighted(frame, alpha, frame_mask, 1.0 - alpha, 0.0)
          
        new_frames.append( mpy.ImageClip( frame ).set_duration(1/video.fps)  )

    new_video = mpy.concatenate_videoclips(new_frames, method='chain')
    del new_frames

    new_video.write_videofile(os.path.join(self.output_path, "{}_with_masks.{}".format(id, ext)), fps=video.fps)      
    gc.collect()     

  def load(self):
    videos = glob.glob(os.path.join(self.video_path, "videos", "*"))
    videos.sort()

    for video in videos:
      if VIDEO_DEBUG is not None:
        video = os.path.join(os.path.dirname(video), "{}.webm".format(VIDEO_DEBUG))

      video_aux = os.path.basename(video).split(".")
      video_id  = video_aux[0]
      video_ext = video_aux[1]
      mask_path = os.path.join(self.mask_path, "{}.npz".format(video_id))

      if os.path.isfile(mask_path):
        file  = mpy.VideoFileClip(video)
        masks = np.load(os.path.join(self.mask_path, "{}.npz".format(video_id)), allow_pickle=True)
        self.do_apply_masks(video_id, video_ext, file, masks)    

        if VIDEO_DEBUG is not None:
          break
      else:
        print("Video {} does not have masks".format(video_id))

class Segment_MyVideo_v2(Segment_MyVideo):
  def __prepare_image(self, image, transform):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=self.device)

    return image.permute(2, 0, 1).contiguous()
  
  def do_process_video(self, idx, id, frame_ann_expand, total_frame, video, masks_path):
    idxs = []
    obj_names = []
    contours = []
    frame_batch = []
    batch_size = 7
    resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)

    for j, frame in tqdm.tqdm(enumerate(video.iter_frames()), total = total_frame, desc = "|-- ({}) Video {}".format(idx, id)):
      annotation = frame_ann_expand[j]
      boxes = []
      categories = []

      for lb in annotation["labels"]:
        boxes.append([
                      lb["box2d"]["x1"], 
                      lb["box2d"]["y1"], 
                      lb["box2d"]["x2"], 
                      lb["box2d"]["y2"]
                    ])
        categories.append(lb["category"])

      if len(boxes) > 0:
        idxs.append(j)
        boxes = torch.tensor(boxes, device=self.device)
        frame_batch.append({
          "image": self.__prepare_image(frame, resize_transform),
          "boxes": resize_transform.apply_boxes_torch(boxes, frame.shape[:2]),
          "original_size": frame.shape[:2]
        })
        obj_names.append(categories)

      ##doing one batch at a time
      if (len(frame_batch) > 0) and (len(frame_batch) % batch_size == 0 or j == (total_frame - 1)):
        frame_batch_masks = self.sam(frame_batch, multimask_output=False)
        frame_batch = []

        for frame_masks in frame_batch_masks:
          contours_aux = []

          for mask in frame_masks["masks"]:
            con, _ = cv2.findContours(mask.squeeze().cpu().numpy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_aux.append([c for c in con]) #one mask can return more than one contourn, e.g., video 125757 

          contours.append(contours_aux)  

    idxs = np.array(idxs)
    obj_names = np.array(obj_names, dtype=object)  ## n objects per frame
    contours = np.array(contours, dtype=object)    ## m countours per frame. m >= n
    np.savez(masks_path, frame_idxs = idxs, obj_names = obj_names, contours = contours)

class Load_Segment_v2(Load_Segment):
  def __name_to_color(self, obj_names):
    np.random.seed(1749)

    color_names_aux = []

    for obj in obj_names:
      color_names_aux.extend(obj)

    color_names_aux = np.unique(np.array(color_names_aux))

    return {name:np.array(colors.hsv_to_rgb([np.random.random(), 1.0, 1.0]) * 255, dtype='uint8') for name in color_names_aux}      

## https://stackoverflow.com/questions/66095686/apply-a-segmentation-mask-through-opencv
## https://www.immersivelimit.com/tutorials/create-bounding-box-from-segmentation
## https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
  def do_apply_masks(self, id, ext, video, masks):
    available_idx = masks["frame_idxs"]
    contours = masks["contours"]
    obj_names = masks["obj_names"]
    color = self.__name_to_color(obj_names)
    total_frame = math.ceil(video.fps * video.duration)
    new_frames = []
    alpha = 0.7

    for idx, frame in tqdm.tqdm(enumerate(video.iter_frames()), total = total_frame, desc = "|-- Video {}".format(id)):
      i_pos = np.where(idx == available_idx)[0]

      if i_pos.shape[0] > 0:
        for cnt, name in zip(contours[i_pos[0]], obj_names[i_pos[0]]):
          mask = np.zeros(frame.shape[:2])
          mask = cv2.fillPoly(mask, pts =cnt, color=1)

          frame_mask = np.where(mask[..., None], color[name], frame)
          frame      = cv2.addWeighted(frame, alpha, frame_mask, 1.0 - alpha, 0.0)          

          ##bounding box around segmentation and label
          rows, cols = np.where(mask == 1)
          x1 = np.min(cols)
          y1 = np.max(rows)
          x2 = np.max(cols)
          y2 = np.min(rows)          
          cv2.rectangle(frame, (x1, y1), (x2, y2), color[name].tolist(), 1)

          (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
          cv2.rectangle(frame, (x1, y2 - h), (x1 + w, y2), color[name].tolist(), -1)
          cv2.putText(frame, name, (x1, y2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)          
          
        new_frames.append( mpy.ImageClip( frame ).set_duration(1/video.fps)  )

    new_video = mpy.concatenate_videoclips(new_frames, method='chain')
    del new_frames

    new_video.write_videofile(os.path.join(self.output_path, "{}_with_masks.{}".format(id, ext)), fps=video.fps)      
    gc.collect()  

def run(args, parser):
  param = ""

  if args.action_type == ACTION_TYPE_SEGMENT and args.bb_file is None:  
    param = '-f/--file'    
  if args.action_type == ACTION_TYPE_LOAD and args.mask_path is None:  
    param = '-m/--masks'    
    
  if param != "": 
    parser.error('the following arguments are required when --action={}: {}'.format(args.action_type, param))

  if args.action_type == ACTION_TYPE_SEGMENT:
    seg = Segment_MyVideo_v2(args.source_path, args.output_path, file_index = args.bb_file)
    seg.run()
  else:  
    loader = Load_Segment_v2(args.source_path, args.output_path, args.mask_path)
    loader.load()    

def main(*args):
  parser = argparse.ArgumentParser(description="")

  parser.add_argument("-a", "--action", help = "Action type", dest = "action_type", choices = [ACTION_TYPE_SEGMENT, ACTION_TYPE_LOAD], default=ACTION_TYPE_SEGMENT, required = True)
  parser.add_argument("-s", "--source", help = "Path to load the videos", dest = "source_path", required = True)
  parser.add_argument("-o", "--output", help = "Path to save results", dest = "output_path", required = True)
  parser.add_argument("-m", "--masks", help = "Path to load masks", dest = "mask_path", required = False)
  parser.add_argument("-f", "--file", help = "Index of a specific bounding box file annotation", dest = "bb_file", required = False, default = None, type=int)

  parser.set_defaults(func = run)
  
  args = parser.parse_args()
  args.func(args, parser) 

#This only executes when this file is executed rather than imported
if __name__ == '__main__':
  main(*sys.argv[1:])
