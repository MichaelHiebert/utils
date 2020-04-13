from collections import defaultdict
import numpy as np
import random

class BoundingBox():
    """
        A Bounding Box for a particular frame.
    """

    def __init__(self, frame_id, object_label, top_left, bottom_right, confidence=None):
        """
            Parameters
            ----------
            frame_id : any hashable item :
                an object that can uniquely distinguish one frame from others, like an unique string
            object_label : str :
                the label of this bounding box
            top_left : tuple (int, int) or (float,f loat) :
                the top left (x,y) coordinate of the bounding box. Can be normalized or not, so long as this normalization is consistent across all bounding boxes
            bottom_right : tuple (int, int) or (float,f loat) :
                the bottom right (x,y) coordinate of the bounding box. Can be normalized or not, so long as this normalization is consistent across all bounding boxes
            confidence : float [0,1] :
                the confidence that this bounding box is correct
        """

        self.frame_id = frame_id
        self.label = object_label
        self.top_left = top_left
        self.tlx,self.tly = [float(x) for x in self.top_left]
        self.bottom_right = bottom_right
        self.brx,self.bry = [float(x) for x in self.bottom_right]

        self.width = self.brx - self.tlx
        self.height = self.bry - self.tly
        
        self.confidence = confidence

    def related_to(self, other_bounding_box):
        """
            Returns True if this bounding box "applies" to `other_bounding_box`, that is,
            there is overlap and they share a label.
        """

        obb = other_bounding_box

        return self.label == obb.label and self._overlap(obb)

    def _overlap(self, other_bounding_box):
        """
            Returns True if this bounding box overlaps `other_bounding_box`, False otherwise.
        """
        obb = other_bounding_box

        if self.tlx > obb.brx or self.brx < obb.tlx:
            return False

        if self.tly > obb.bry or self.bry < obb.tly:
            return False

        return True

    def iou(self, other_bounding_box):
        """
            Calculates the intersection over union of this bounding box
            and another bounding box

            other_bounding_box : BoundingBox :
                another bounding box
        """
        obb = other_bounding_box

        return self._intersection(obb) / self._union(obb)

    def _intersection(self, other_bounding_box):
        """Calculates the intersection area of this bounding box and another bounding box."""

        obb = other_bounding_box

        rightest_left = max(self.tlx, obb.tlx)
        leftest_right = min(self.brx, obb.brx)

        if leftest_right < rightest_left:
            return 0 # no overlap
        
        bottomest_top = max(self.tly, obb.tly)
        toppest_bottom = min(self.bry, obb.bry)

        if bottomest_top > toppest_bottom:
            return 0

        return (leftest_right - rightest_left) * (toppest_bottom - bottomest_top)

    def _union(self, other_bounding_box):
        """Calculates the union area of this bounding box and another bounding box."""
        obb = other_bounding_box

        total_area = self.width * self.height + obb.width * obb.height

        return total_area - self._intersection(obb)

    def matches(self, obb_iterable, threshold=0.5):
        """
            Returns a len(obb_iterable) list of booleans where the item corresponding to another bounding box is True if the IoU of this bounding box and that bounding box is >= threshold, and False otherwise
        """

        list_to_ret = []

        for obb in obb_iterable:
            if self.related_to(obb):
                iou = self.iou(obb)
                list_to_ret.append(iou >= threshold)
            else:
                list_to_ret.append(False)
        
        return list_to_ret
    
    def __eq__(self, value):
        return type(value) == BoundingBox and \
            self.frame_id == value.frame_id and \
            self.label == value.label and \
            self.bottom_right == value.bottom_right and \
            self.top_left == value.top_left and \
            self.confidence == value.confidence

class Detection():
    """
        An abstract Detection encompassing any number of frames,
        absolute labels, and/or predictions.
    """

    def __init__(self, labels=None, predictions=None, frames=None):
        """
            Initialize this `Detection` object.

            Paramters
            ---------
            labels : iterable of
            
            `(frame_id, object_label, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y))` 
            
            or
            
            `(frame_id, object_label, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), confidence)`
            
            or
            
            `BoundingBox`::

            All of the ground-truth labels for a given set of frames.

            predictions : iterable of

            `(frame_id, object_label, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y))` 
            
            or
            
            `(frame_id, object_label, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), confidence)`
            
            or
            
            `BoundingBox`::

            All of the predictions for a given set of frames.

            TODO frames : iterable of `Frame` or `channels`x`rows`x`columns` numpy array
        """

        if labels != None:
            self.labels = self._handle_new_bounding_boxes(labels)
        else:
            self.labels = None

        if predictions != None:
            self.predictions = self._handle_new_bounding_boxes(predictions)
        else:
            self.predictions = None

        if frames != None:
            pass # TODO

    def _handle_new_bounding_boxes(self, bb_iter):
        """
            Converts all possible bounding-box array formats into one format.

            Parameters
            ----------
            bb_array : iterable of bounding boxes in the format described in
            `__init__`
        """

        # (frame_id, object_label, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), confidence)

        dict_to_ret = dict()

        for item in bb_iter:
            if type(item) == BoundingBox:
                if item.frame_id not in dict_to_ret:
                    dict_to_ret[item.frame_id] = list()

                dict_to_ret[item.frame_id].append(item)
            else:
                pass # TODO
        
        return dict_to_ret

    def from_csv(self, label_filepath=None, pred_filepath=None, delimiter=',', line_delimiter='\n'):
        """
            Loads the data from the specified filepath(s) and formats them appropriately.

            The CSV must be formatted as follows:

            frame_id, object_label, top_left_x, top_left_y, bottom_right_x, bottom_right_y

            or 

            frame_id, object_label, top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence

            Parameters
            ----------
            label_filepath : str ::
                the relative filepath to the csv file containing the ground truth labels

            pred_filepath : str ::
                the relative filepath to the csv file containing the predicted labels

            delimiter : str ::
                the delimiting character or str between entries in a row

            line_delimiter : str ::
                the delimiting character or str between rows

            Returns
            -------
            None
        """

        if label_filepath != None:
            self.labels = self._digest_csv(label_filepath, delimiter, line_delimiter)
        
        if pred_filepath != None:
            self.predictions = self._digest_csv(pred_filepath, delimiter, line_delimiter)

    def _digest_csv(self, filepath, delimiter, line_delimiter):
        """Digest a csv file into a dictionary where frames are keys and lists of BoundingBox are values"""

        dict_to_ret = dict()

        with open(filepath) as f:
            lines = f.read()

        for line in lines.split(line_delimiter):
            splt = [item.strip() for item in line.split(delimiter)]

            if len(splt) <= 1: continue

            if len(splt) == 6:
                frame_id,label,tlx,tly,brx,bry = splt
                confidence = None
            elif len(splt) == 7:
                frame_id,label,tlx,tly,brx,bry,confidence= splt
            else:
                raise RuntimeError('Incorrect CSV row format!')

            if frame_id in dict_to_ret:
                dict_to_ret[frame_id].append(BoundingBox(frame_id, label, (tlx,tly), (brx,bry), confidence=confidence))
            else:
                dict_to_ret[frame_id] = [BoundingBox(frame_id, label, (tlx,tly), (brx,bry), confidence=confidence)]
                    
        return dict_to_ret

    def add_label(self, *args):
        """
            Add a prediction of the form BoundingBox or

            `frame_id, object_label, top_left_x, top_left_y, bottom_right_x, bottom_right_y(, confidence)`

            where confidence is optional
        """
        try:
            if self.labels == None: self.labels = dict()

            self._add_bounding_box(args, self.labels)
        except Exception as e:
            self.labels = None
            raise e

    def add_prediction(self, *args):
        """
            Add a prediction of the form BoundingBox or

            `frame_id, object_label, top_left_x, top_left_y, bottom_right_x, bottom_right_y{, confidence}` or

            `frame_id, object_label, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y){, confidence}`

            where confidence is optional
        """
        try:
            if self.predictions == None: self.predictions = dict()
            self._add_bounding_box(args, self.predictions)
        except Exception as e:
            self.predictions = None
            raise e
    
    def _add_bounding_box(self, bb_args, bb_dict):
        """Add a bounding box to the specified label/predictions dict"""
        bb = self._handle_bb_args(bb_args)

        if bb.frame_id not in bb_dict:
            bb_dict[bb.frame_id] = [bb]
        else:
            bb_dict[bb.frame_id].append(bb)

    def _handle_bb_args(self, args):
        """Convert the arglist in any format to a BoundingBox and return it"""
        if len(args) == 1: # BoundingBox
            if type(args[0]) == BoundingBox:
                return args[0]
        elif len(args) in [4,5]: # frame_id, object_label, top_left, bottom_right(, confidence)
            return BoundingBox(*args)
        elif len(args) == 6:
            return BoundingBox(*args[:2], (args[2], args[3]), (args[4], args[5]))
        elif len(args) == 7:
            return BoundingBox(*args[:2], (args[2], args[3]), (args[4], args[5]), args[6])
        else:
            raise RuntimeError('Unable to process BoundingBox arguments!')

    def metrics(self, confidence_threshold=0.5, iou_threshold=0.5):
        """
            Parameters
            ----------
            confidence_threshold : float ::
                the threshold under which predicted bounding boxes will be filtered out, as if they were not predicted at all
            iou_threshold : float ::
                the threshold for overlapping bounding boxes to determine a valid match

            Returns
            -------
            a tuple of floats for precision,recall,fscore
        """
        if self.labels == None:
            raise RuntimeError('There are no labels associated with this detection!')
        if self.predictions == None:
            raise RuntimeError('There are no predictions associated with this detection!')

        true_pos = 0
        false_pos = 0

        # true_neg = 0 # irrelevant
        false_neg = 0

        for frame in self.labels:
            labels = self.labels[frame]

            if frame not in self.predictions:
                false_neg += len(labels)
                continue
            
            preds = self.predictions[frame]

            f_neg = [False for item in labels]
            t_pos = []
            
            for pred in preds:
                if pred.confidence == None or pred.confidence >= confidence_threshold:
                    matches = pred.matches(labels, threshold=iou_threshold)
                    t_pos.append(True in matches)

                    f_neg = [f_n or matches[i] for i,f_n in enumerate(f_neg)]
            
            for item in f_neg:
                if item == False: false_neg += 1
            
            for item in t_pos:
                if item == True: true_pos += 1
                else: false_pos += 1

        if true_pos == 0.0: # we made no good predictions. sad!
            return 0.0,0.0,0.0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fscore = 2 * ( (precision * recall) / (precision + recall))

        return precision,recall,fscore

    def load_labels_from_annot_dict(self, annot_dict):
        """
            Only for use with annotation dict of form:
            {
            'frame_id': {
                'label': [
                    (x,y,w,h)
                    ]
                }
            }
        """

        internal_dict = dict()

        for frame in annot_dict:
            if frame not in internal_dict:
                internal_dict[frame] = []

            for label in annot_dict[frame]:
                for box in annot_dict[frame][label]:
                    x,y,w,h = box

                    tlx = x
                    tly = y
                    brx = x + w
                    bry = y + h

                    bb = BoundingBox(frame, label, (tlx,tly), (brx,bry))

                    internal_dict[frame].append(bb)

        self.labels = internal_dict

    def labels_to_annot_dict(self):
        annot_dict = dict()

        for frame in self.labels:

            if frame not in annot_dict:
                annot_dict[frame] = dict()

            for bb in self.labels[frame]:
                if bb.label not in annot_dict[frame]:
                    annot_dict[frame][bb.label] = []

                val = bb.tlx, bb.tly, bb.brx - bb.tlx, bb.bry - bb.tly
                annot_dict[frame][bb.label].append(val)

        return annot_dict

    def add_disjoint_boxes(self, num_boxes_to_add, label, max_width, max_height, verbose=False):
        for frame in self.labels:
            bba = BoundingBoxArray(self.labels[frame], max_width=max_width, max_height=max_height)
            to_add = bba.add_disjoint_boxes(num_boxes_to_add, label=label)
            bb_to_add = [BoundingBox(frame, label, tl, br) for tl,br in to_add]

            for bb in bb_to_add:
                if verbose: print('Added new bounding box of label {} to frame {} with coords {} and {}'.format(label, frame, bb.top_left, bb.bottom_right))
                self.labels[frame].append(bb)

class BoundingBoxArray():
    """A two-dimensional array of bounding boxes."""

    def __init__(self, bounding_box_list, max_width=None, max_height=None):
        self.list = bounding_box_list
        
        if max_width == None:
            self.width = int(max([bb.brx for bb in bounding_box_list]))
        else:
            self.width = max_width

        if max_height == None:
            self.height = int(max([bb.bry for bb in bounding_box_list]))
        else:
            self.height = max_height

        if self.width <= 1 or self.height <= 1:
            self.width = int(self.width * 1000)
            self.height = int(self.height * 1000)
            self.normalized = True
        else:
            self.normalized = False
        
        self.array = np.zeros((self.height,self.width))

        for bb in bounding_box_list:
            self._add_bounding_box_to_array(bb)
    
    def _add_bounding_box_to_array(self, bb):
        tlx,tly = bb.top_left
        brx,bry = bb.bottom_right

        if self.normalized:
            tlx = int(tlx * 1000)
            tly = int(tly * 1000)
            brx = int(brx * 1000)
            bry = int(bry * 1000)

        self.array[tly:bry, tlx:brx] = 1

    def _add_disjoint_bounding_box(self, label='background', min_size=(10,10), decay=0, tries=10):
        """
            Tries to add a new non-overlapping bounding box. Returns True if successful, False if not.
        """

        if tries == 0: return False

        tlx = random.randint(0,self.width - 1)
        tly = random.randint(0,self.height - 1)

        # pick a random start point
        while self.array[tly,tlx] != 0:
            tlx = random.randint(0,self.width - 1)
            tly = random.randint(0,self.height - 1)

        should_grow = True

        cur_width = 0
        cur_height = 0

        cur_x = tlx
        cur_y = tly

        while should_grow and cur_x < self.width - 1 and cur_y < self.height - 1:
            if cur_width < min_size[0] and cur_height < min_size[1]:
                if random.randint(0,1) == 0:
                    if 1 in self.array[tly:cur_y,cur_x + 1]: # collision
                        return self._add_disjoint_bounding_box(label=label, min_size=min_size, decay=decay, tries=tries-1) # try again
                    else:
                        cur_x += 1
                        cur_width += 1
                else:
                    if 1 in self.array[cur_y + 1,tlx:cur_x]: # collision
                        return self._add_disjoint_bounding_box(label=label, min_size=min_size, decay=decay, tries=tries-1) # try again
                    else:
                        cur_y += 1
                        cur_height += 1
            elif cur_width < min_size[0]: # subminimal width
                if 1 in self.array[tly:cur_y,cur_x + 1]: # collision
                        return self._add_disjoint_bounding_box(label=label, min_size=min_size, decay=decay, tries=tries-1) # try again
                else:
                    cur_x += 1
                    cur_width += 1
            elif cur_height < min_size[1]: # subminimal height
                if 1 in self.array[cur_y + 1,tlx:cur_x]: # collision
                    return self._add_disjoint_bounding_box(label=label, min_size=min_size, decay=decay, tries=tries-1) # try again
                else:
                    cur_y += 1
                    cur_height += 1
            else: # above minimum thresholds
                while random.randint(0,1000) > decay and cur_x < self.width - 1 and cur_y < self.height - 1:

                    # randomly expand
                    if random.randint(0,1) == 0:
                        if 1 in self.array[tly:cur_y,cur_x + 1]: # collision
                            self.array[tly:cur_y, tlx:cur_x] = 2
                            return (tlx,tly),(cur_x,cur_y)
                        else:
                            cur_x += 1
                            cur_width += 1
                    else:
                        if 1 in self.array[cur_y + 1,tlx:cur_x]: # collision
                            self.array[tly:cur_y, tlx:cur_x] = 2
                            return (tlx,tly),(cur_x,cur_y)
                        else:
                            cur_y += 1
                            cur_height += 1
                    
                    decay += 25

                self.array[tly:cur_y, tlx:cur_x] = 2
                return (tlx,tly),(cur_x,cur_y)

        return self._add_disjoint_bounding_box(label=label, min_size=min_size, decay=decay, tries=tries-1) # try again
    
    def add_disjoint_boxes(self, num_boxes, label='background', min_width=10, min_height=10, tries=10):
        added = []
        for i in range(num_boxes):
            res = self._add_disjoint_bounding_box(label=label, min_size=(min_width,min_height), tries=tries)


            if res != False:
                if self.normalized:
                    tl,br = res
                    tl = tl[0] / 1000, tl[1] / 1000
                    br = br[0] / 1000, br[1] / 1000
                    res = tl,br
                added.append(res)
        
        return added

if __name__ == "__main__":
    a = BoundingBox('','',(0,0), (5,5))
    b = BoundingBox('','',(5,5), (10,10))
    c = BoundingBox('','',(15,15), (20,20))

    np.set_printoptions(threshold=np.inf)

    bba = BoundingBoxArray([a,b,c], max_width=20, max_height=20)
    res = bba.add_disjoint_boxes(5,min_size=5)
    print(res)
    print(bba.array)