
import os, math, glob, json, pdb, cv2, numpy as np, torch
from torch.utils.data import Dataset
from pddl import parse_domain, logic
from enum import IntEnum
from sklearn.preprocessing import LabelEncoder

RETURN_ALL_FRAMES = "all"    ##you must pass collate_fn to the DataLoader
RETURN_ONLY_FIRST_AND_LAST = "first_last"

MAX_BOX_NUMBER = 4
NUMBER_BOX_COORDINATES = 4
NON_LABELED_BOX = ""

class PREDICATE_STATES(IntEnum): 
  NONAPPLICABLE  = -1
  NEGATIVE       = 0
  AFFIRMATIVE    = 1

class PARAMS(IntEnum): 
  HAND = 0
  OBJ1 = 1
  OBJ2 = 2

""""To be used with RETURN_ALL_FRAMES. Videos in one batch could have different frame number"""
def collate_fn(data):
  video_id, samples, boxes, surroundes, categories, precondition_obj, precondition_rel, effect_obj, effect_rel = zip(*data)

  max_frames = max([ sp.shape[0]  for sp in samples ])
  new_samples = []
  new_boxes = []
  new_surroundes = []
  new_categories = []
  new_precondition_obj = []
  new_precondition_rel = []
  new_effect_obj = []
  new_effect_rel = []  

  for aux in effect_obj:
    print(aux.shape)

  for i in range(len(data)):
    sp = samples[i]
    bx = boxes[i]
    sr = surroundes[i]
    ct = categories[i]
    pro = precondition_obj[i]
    prr = precondition_rel[i]
    efo = effect_obj[i]
    efr = effect_rel[i]    
    npad = max(max_frames - sp.shape[0], 0)

    if npad > 0:
      _pad = torch.zeros( (npad, sp.shape[1] * sp.shape[2] * sp.shape[3])   ).reshape( (npad, sp.shape[1], sp.shape[2], sp.shape[3]))
      sp = torch.cat([sp, _pad])

      _pad = torch.zeros( (npad, bx.shape[1] * bx.shape[2])   ).reshape( (npad, bx.shape[1], bx.shape[2]))
      bx = torch.cat([bx, _pad])      

      _pad = torch.zeros( (npad, sr.shape[1] * sr.shape[2])   ).reshape( (npad, sr.shape[1], sr.shape[2]))
      sr = torch.cat([sr, _pad])            

      _pad = torch.zeros( (npad, ct.shape[1])   ).reshape( (npad, ct.shape[1]))
      ct = torch.cat([ct, _pad])

      _pad = torch.zeros( (npad, pro.shape[1] * pro.shape[2])   ).reshape( (npad, pro.shape[1], pro.shape[2]))
      pro = torch.cat([pro, _pad])      

      _pad = torch.zeros( (npad, prr.shape[1])   ).reshape( (npad, prr.shape[1]))
      prr = torch.cat([prr, _pad])            

      _pad = torch.zeros( (npad, efo.shape[1] * efo.shape[2])   ).reshape( (npad, efo.shape[1], efo.shape[2]))
      efo = torch.cat([efo, _pad])      

      _pad = torch.zeros( (npad, efr.shape[1])   ).reshape( (npad, efr.shape[1]))
      efr = torch.cat([efr, _pad])      

    new_samples.append( sp )
    new_boxes.append( bx )
    new_surroundes.append( sr )
    new_categories.append( ct )
    new_precondition_obj.append( pro )
    new_precondition_rel.append( prr )
    new_effect_obj.append( efo )
    new_effect_rel.append( efr )    

  ## see __getitem__
  return ( np.array(video_id),
           torch.stack(new_samples),             #batch, max_frames, widght, height, 3x channel
           torch.stack(new_boxes),               #batch, max_frames, MAX_BOX_NUMBER, NUMBER_BOX_COORDINATES
           torch.stack(new_surroundes),          #batch, max_frames, MAX_BOX_NUMBER * (MAX_BOX_NUMBER - 1), NUMBER_BOX_COORDINATES
           torch.stack(new_categories),          #batch, max_frames, MAX_BOX_NUMBER
           torch.stack(new_precondition_obj),    #batch, max_frames, 3x something objects, 2x single-param predicates
           torch.stack(new_precondition_rel),    #batch, max_frames, 2x double-param predicates
           torch.stack(new_effect_obj),          #batch, max_frames, 3x something objects, 2x single-param predicates
           torch.stack(new_effect_rel) )         #batch, max_frames, 2x double-param predicates

class CustomVideoDataset(Dataset):
  def __init__(self, main_video_dir, annotations_file, labels_dir, pddl_file, frame_type = ["*.jpg", "*.jpeg", "*.png"], return_frame = RETURN_ONLY_FIRST_AND_LAST, resize_shape = None):
    self.return_frame = RETURN_ONLY_FIRST_AND_LAST if return_frame is None else return_frame
    self.frame_type   = frame_type
    self.resize_shape = resize_shape

    self.video_list  = glob.glob(os.path.join(main_video_dir, "*"))

    """File with bound boxes annotations
       video/frame id, boxes position, and categories"""
    self.annotations = open(annotations_file, "r")
    self.annotations = json.load(self.annotations)

    self.__encode_category()

    """Combines label files (train, validation, and test)
       video id, label, template, place_holders"""
    aux_files = glob.glob(os.path.join(labels_dir, "*.json"))
    self.labels  = []

    for lbf in aux_files:
      label = open(lbf, "r")
      label = json.load(label)
      self.labels.extend(label)

    """PDDL files have video labels as comments of the actions"""
    self.pddl_domain = parse_domain(pddl_file)
    self.pddl_plain_text = open(pddl_file, "r")
    self.pddl_plain_text = self.pddl_plain_text.read()

    self.pddl_domain_predicates_single = []
    self.pddl_domain_predicates_double = []

    for pred in self.pddl_domain.predicates:
      if len(pred.terms) == 1:
        self.pddl_domain_predicates_single.append(pred.name)
      else:  
        self.pddl_domain_predicates_double.append(pred.name)

    self.pddl_domain_predicates_double.append("=")
    self.pddl_domain_predicates_double.sort()    
    self.pddl_domain_predicates_single.sort()

  def __encode_category(self):
    category_list = []
    
    for video_id in self.annotations:
      for frame in self.annotations[video_id]:
        for label in frame["labels"]:
          category_list.append(label["category"])

    category_list = np.unique(category_list)
    category_list = np.append(category_list, NON_LABELED_BOX)

    self.category_encoder = LabelEncoder()
    self.category_encoder.fit(category_list)

  def __get_video_label(self, video_id):
    for video in self.labels:
      if video["id"] == video_id:
        return video["template"].replace("[something]", "something"), video["placeholders"]
      
    return NON_LABELED_BOX, []
  
  def __get_video_pddl_action(self, video_id):
    label, placeholders = self.__get_video_label(video_id)
    label = label.lower()

    """Matches the self.labels with self.pddl_domain action name by parsing this type of structure
    	 ; 0 Approaching something with your camera               
       (:action approach
         :parameters (?a - sth)
         :precondition (and
           (not (close ?a))
           (visible ?a)
           (not (visible hand))
         )
         :effect (close ?a)
       )"""
    plain_text = self.pddl_plain_text.lower()
    part1 = plain_text.split(label)[1]
    part2 = part1.split(":action")[1]
    part3 = part2.split(":parameters")[0].strip()

    for action in self.pddl_domain.actions:
      if action.name == part3:
        return action, placeholders
      
    return None, []
    
  def __get_pddl_info(self, operation, label_placeholders, box_categories):
    pred_single_param = self.pddl_domain_predicates_single
    pred_double_param = self.pddl_domain_predicates_double

    """pddl_obj has one row per each [something] with positions related to the ordered predicate names (with single param) and their antonyms
       Values representes the PREDICATE_STATES

       pddl_relations is a vector with positions related to the ordered predicate names (with double param) and their antonyms

       In both cases, ser values only when there are bounding boxes defined in box_categories"""
    pddl_obj = np.full((3, 2 * len(pred_single_param)), PREDICATE_STATES.NONAPPLICABLE) 
    pddl_relations = np.full(2 * len(pred_double_param), PREDICATE_STATES.NONAPPLICABLE)
    
    operands = [operation]

    if not isinstance(operation, logic.predicates.Predicate):
      operands = operation.operands

    param_idx = 0

    #==========================================================================================================#    
    """Filters PDDL predicates based on bounding boxes
    Builds a vector with values that match PDDL file parameters
    Assumptions: label_placeholders could have 'hand' and/or other two params
                 there is relation between labels (label_placeholders) and annotations (box_categories)
    TODO: improve match between labels (label_placeholders) and annotations (box_categories)"""
    box_categories_aux = box_categories.copy()
    
    for ph in label_placeholders: 
      ph  = ph.lower()
      obj = [  (box != NON_LABELED_BOX) and ((ph in box) or (box in ph)) for box in box_categories  ]

      if "hand" not in ph:
        param_idx = param_idx + 1
      if np.any(obj):
        box_categories_aux[ np.where(obj)[0][0] ] = "hand" if "hand" in ph else "?a" if param_idx == 1 else "?b"
    #==========================================================================================================#      

    box_category_par = []

    for op in operands:
      hand = "hand" in str(op) and "hand" in box_categories_aux
      obj1 = "?a" in str(op) and "?a" in box_categories_aux
      obj2 = "?b" in str(op) and "?b" in box_categories_aux
      affirmative = True

      if isinstance(op, logic.base.Not):
        op = op.argument
        affirmative = False

      op_name = "=" if isinstance(op, logic.predicates.EqualTo) else op.name

      if op_name == "=" or len(op.terms) > 1:
        idx = pred_double_param.index(op_name)

        ##For touching predicate in PDDL, hand is a constante but has no bounding-box in Something-Else
        if op_name == "touching" and not hand:
          pddl_relations[idx] = PREDICATE_STATES.AFFIRMATIVE if affirmative else PREDICATE_STATES.NEGATIVE
          pddl_relations[idx + len(pred_double_param)] = not pddl_relations[idx]
        if np.where([hand, obj1, obj2])[0].shape[0] > 1:
          pddl_relations[idx] = PREDICATE_STATES.AFFIRMATIVE if affirmative else PREDICATE_STATES.NEGATIVE
          pddl_relations[idx + len(pred_double_param)] = not pddl_relations[idx]

          box_cat_aux = ()

          if hand:
            box_cat_aux = box_cat_aux + (box_categories[box_categories_aux.index("hand")], )
          if obj1:
            box_cat_aux = box_cat_aux + (box_categories[box_categories_aux.index("?a")], )
          if obj2:
            box_cat_aux = box_cat_aux + (box_categories[box_categories_aux.index("?b")], )

          if box_cat_aux not in box_category_par:
            box_category_par.append( box_cat_aux )  
      else:  
        idx = pred_single_param.index(op_name)

        if hand:
          pddl_obj[PARAMS.HAND][idx]  = PREDICATE_STATES.AFFIRMATIVE if affirmative else PREDICATE_STATES.NEGATIVE
          pddl_obj[PARAMS.HAND][idx + len(pred_single_param)] = not pddl_obj[PARAMS.HAND][idx]
        if obj1:
          pddl_obj[PARAMS.OBJ1][idx]  = PREDICATE_STATES.AFFIRMATIVE if affirmative else PREDICATE_STATES.NEGATIVE
          pddl_obj[PARAMS.OBJ1][idx + len(pred_single_param)] = not pddl_obj[PARAMS.OBJ1][idx]
        if obj2:
          pddl_obj[PARAMS.OBJ2][idx]  = PREDICATE_STATES.AFFIRMATIVE if affirmative else PREDICATE_STATES.NEGATIVE
          pddl_obj[PARAMS.OBJ2][idx + len(pred_single_param)] = not pddl_obj[PARAMS.OBJ2][idx]

    return pddl_obj, pddl_relations, box_category_par       

  def __frame_by_type(self, path, frame_type):
    files = []

    """Filter defined image types"""
    for tp in frame_type:
      files.extend(glob.glob(os.path.join(path, tp)))

    return files  
  
  """Bounding boxes with: x1, y1, x2, y2, encoded categories, and their descriptions
     There are at most one box per hand and two box for objects
     
     Always returns MAX_BOX_NUMBER to faciliate final shape match
     Rows with only zero coordinates have no boxes"""
  def __get_frame_box(self, frame_name, video_annotation):
    box_list = np.zeros((MAX_BOX_NUMBER, NUMBER_BOX_COORDINATES))
    cat_list = [ self.category_encoder.transform([NON_LABELED_BOX])[0] for _ in range(MAX_BOX_NUMBER)  ]
    cat_desc = [ NON_LABELED_BOX for _ in range(MAX_BOX_NUMBER)  ]

    """Returns only bound boxes and their categories (labels)"""
    for item in video_annotation:
      if frame_name == os.path.basename(item["name"]):
        for i, label in enumerate(item["labels"]):
          box_list[i] = [ label["box2d"]["x1"], label["box2d"]["y1"], label["box2d"]["x2"], label["box2d"]["y2"] ] 
          cat_list[i] = self.category_encoder.transform( [ label["category"] ] )[0]
          cat_desc[i] = label["category"]

    return box_list, cat_list, cat_desc 

  """Surrounding boxes (x1, y1, x2, y2) that contains pars of bouding boxes

     Always returns MAX_BOX_NUMBER * (MAX_BOX_NUMBER - 1) to faciliate final shape match
     Rows with only zero coordinates have no boxes"""  
  def __get_surround_box(self, boxes, categories, pars):
    surround_box = np.zeros((MAX_BOX_NUMBER * (MAX_BOX_NUMBER - 1), NUMBER_BOX_COORDINATES))

    for i, par in enumerate(pars):
      idx1 = categories.index(par[0])
      idx2 = categories.index(par[1])

      box1 = boxes[idx1]
      box2 = boxes[idx2]

      surround_box[i] = [ min(box1[0], box2[0]), min(box1[1], box2[1]),   #x1, y1
                          max(box1[2], box2[2]), max(box1[3], box2[3]) ]  #x2, y2
        
    return surround_box

  def __len__(self):
    return len(self.video_list)

  """It returns more than one frame per video, thus frame_list.shape  = (batch_size, #frames, ...)
     Use frame_list.view((-1, ) + frame_list.shape[2:]) to maintain frame dimensions and combine other ones. frame_list.shape = (batch_size X #frames, ...)
     Procede the same with the other lists .
     The frames of the same video stay together and ordered after invoking view."""
  def __getitem__(self, idx):
    video_id    = os.path.basename(self.video_list[idx])
    frame_names = self.__frame_by_type(self.video_list[idx], self.frame_type)
    frame_names.sort()

    if self.return_frame == RETURN_ONLY_FIRST_AND_LAST:
      frame_names = [frame_names[0], frame_names[-1]]
    elif isinstance(self.return_frame, int): #returns self.return_frame frames
      frame_names = [frame_names[i] for i in range(0, len(frame_names), math.ceil(len(frame_names) / self.return_frame))]

    frame_list = []
    box_list   = []
    surround_list = []
    category_list = []
    pre_obj_list  = []
    pre_relation_list = []
    effect_obj_list   = []
    effect_relation_list = []
    action, label_ph = self.__get_video_pddl_action(video_id)

    print(action, label_ph)

    for frame_path in frame_names:
      frame = cv2.imread(frame_path)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      if self.resize_shape is not None:
        frame = cv2.resize(frame, self.resize_shape)

      frame_list.append( frame / 255.0  )

      boxes, categories, cat_desc     = self.__get_frame_box(os.path.basename(frame_path), self.annotations[video_id])
      pre_obj, pre_relation, box_pars = self.__get_pddl_info(action.precondition, label_ph, cat_desc)
      eff_obj, eff_relation, _        = self.__get_pddl_info(action.effect, label_ph, cat_desc)
      surround_boxes                  = self.__get_surround_box(boxes, cat_desc, box_pars)

      box_list.append(boxes)
      category_list.append(categories)

      pre_obj_list.append(pre_obj)
      pre_relation_list.append(pre_relation)

      effect_obj_list.append(eff_obj)
      effect_relation_list.append(eff_relation)      

      surround_list.append(surround_boxes)

    ## see collate_fn
    return ( np.array(video_id), 
             torch.Tensor(np.array(frame_list)),            #frames, widght, height, 3x channel
             torch.Tensor(np.array(box_list)),              #frames, MAX_BOX_NUMBER, NUMBER_BOX_COORDINATES
             torch.Tensor(np.array(surround_list)),         #frames, MAX_BOX_NUMBER * (MAX_BOX_NUMBER - 1), NUMBER_BOX_COORDINATES
             torch.Tensor(np.array(category_list)),         #frames, MAX_BOX_NUMBER
             torch.Tensor(np.array(pre_obj_list)),          #frames, 3x something objects, 2x single-param predicates
             torch.Tensor(np.array(pre_relation_list)),     #frames, 2x double-param predicates
             torch.Tensor(np.array(effect_obj_list)),       #frames, 3x something objects, 2x single-param predicates
             torch.Tensor(np.array(effect_relation_list)) ) #frames, 2x double-param predicates
