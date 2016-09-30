import os
import cPickle
import scipy
import numpy as np
from datasets.imdb import imdb


class wider(imdb):
  def __init__(self, meth, path):
    imdb.__init__(self, 'wider')
    self._is_train = True if meth == 'train' else False
    self._data_path = "data/wider"
    self._classes = ['__background__', 'face']
    self._load_image_set(path)
    self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

  def _load_image_set(self, path):
    train, test = load_wider(path)
    selected = train if self._is_train is True else test
    self._image_index = []
    self._gt_roidb = []
    for img_path, bboxes in selected:
      self._image_index.append(img_path)
      num = len(bboxes)
      boxes = np.zeros((num, 4), dtype=np.int16)
      gt_classes = np.zeros((num), dtype=np.int32)
      overlaps = np.zeros((num, self.num_classes), dtype=np.float32)
      seg_areas = np.zeros((num), dtype=np.float32)
      for ix in xrange(num):
        box = bboxes[ix]
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2] - 1
        y2 = box[1] + box[3] - 1
        boxes[ix, :] = [int(x1), int(y1), int(x2), int(y2)]
        cls = 1
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
      overlaps = scipy.sparse.csr_matrix(overlaps)
      self._gt_roidb.append({
        'boxes': boxes,
        'gt_classes': gt_classes,
        'gt_overlaps': overlaps,
        'flipped': False,
        'seg_areas': seg_areas})

  def image_path_at(self, i):
    return self._image_index[i]

  def gt_roidb(self):
    return self._gt_roidb

  def rpn_roidb(self):
    if self._is_train is True:
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)
    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print 'loading {}'.format(filename)
    assert os.path.exists(filename), \
           'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = cPickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)


# utils to load wider data

def get_dirmapper(dirpath):
  """return dir mapper for wider face
  """
  mapper = {}
  for d in os.listdir(dirpath):
    dir_id = d.split('--')[0]
    mapper[dir_id] = os.path.join(dirpath, d)
  return mapper


def load_wider(WIDER_FACE):
  """load wider face dataset
  """
  train_mapper = get_dirmapper(os.path.join(WIDER_FACE, 'WIDER_train', 'images'))
  val_mapper = get_dirmapper(os.path.join(WIDER_FACE, 'WIDER_val', 'images'))

  def gen(text, mapper):
    fin = open(text, 'r')

    result = []
    while True:
      line = fin.readline()
      if not line: break  # eof
      name = line.strip()
      dir_id = name.split('_')[0]
      img_path = os.path.join(mapper[dir_id], name + '.jpg')
      face_n = int(fin.readline().strip())

      bboxes = []
      for i in range(face_n):
        line = fin.readline().strip()
        components = line.split(' ')
        x, y, w, h = [float(_) for _ in components]

        size = min(w, h)
        # only large enough
        if size > 12:
          bbox = [x, y, w, h]
          bboxes.append(bbox)

      # # for debug
      # img = cv2.imread(img_path)
      # for bbox in bboxes:
      #   x, y, w, h = bbox
      #   cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 1)
      # cv2.imshow('img', img)
      # cv2.waitKey(0)

      if len(bboxes) > 0:
        result.append([img_path, bboxes])
    fin.close()
    return result

  txt_dir = os.path.join(WIDER_FACE, 'wider_face_split')
  train_data = gen(os.path.join(txt_dir, 'wider_face_train.txt'), train_mapper)
  val_data = gen(os.path.join(txt_dir, 'wider_face_val.txt'), val_mapper)
  return (train_data, val_data)
