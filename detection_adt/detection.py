from collections import defaultdict

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
        
        if confidence != None:
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
        if predictions != None:
            self.predictions = self._handle_new_bounding_boxes(predictions)
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

    def metrics(self):
        """
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
                matches = pred.matches(labels)
                t_pos.append(True in matches)

                f_neg = [f_n or matches[i] for i,f_n in enumerate(f_neg)]
            
            for item in f_neg:
                if item == False: false_neg += 1
            
            for item in t_pos:
                if item == True: true_pos += 1
                else: false_pos += 1

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fscore = 2 * ( (precision * recall) / (precision + recall))

        return precision,recall,fscore


                    






            