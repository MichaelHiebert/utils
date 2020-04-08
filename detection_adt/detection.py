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
        self.tlx,self.tly = self.top_left
        self.bottom_right = bottom_right
        self.brx,self.bry = self.bottom_right

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

        return self._overlap(obb) and self.label == obb.label

    def _overlap(self, other_bounding_box):
        """
            Returns True if this bounding box overlaps `other_bounding_box`, False otherwise.
        """
        obb = other_bounding_box

        if self.tlx > obb.brx or self.brx < obb.tlx:
            return False

        if self.tly < obb.bry or self.bry > obb.tly:
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
        """
            Calculates the intersection area of this bounding box and another bounding box.
        """

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
        """
            Calculates the union area of this bounding box and another bounding box.
        """
        obb = other_bounding_box

        total_area = self.width * self.height + obb.width * obb.height

        return total_area - self._intersection(obb)
    
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

        def _handle_new_bounding_boxes(self, old_bb_dict):
            """
                Converts all possible bounding-box array formats into one format.

                Parameters
                ----------
                bb_array : iterable of bounding boxes in the format described in
                `__init__`
            """

            # (frame_id, object_label, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), confidence)

            # TODO

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
                pass

        def _digest_csv(self, filepath, delimiter, line_delimiter):
            """Digest a csv file into a dictionary where frames are keys and lists of BoundingBox are values"""

            to_ret = dict()

            with open(filepath) as f:
                lines = f.read()

            for line in lines.split(line_delimiter):
                splt = [item.trim() for item in line.split(delimiter)]

                if len(splt) == 6:
                    frame_id,label,tlx,tly,brx,bry = splt
                    confidence = None
                elif len(splt) == 7:
                    frame_id,label,tlx,tly,brx,bry,confidence= splt
                else:
                    raise RuntimeError('Incorrect CSV row format!')

                if frame_id in to_ret:
                    to_ret[frame_id].append(BoundingBox(frame_id, label, (tlx,tly), (brx,bry), confidence=confidence))
                else:
                    to_ret[frame_id] = [BoundingBox(frame_id, label, (tlx,tly), (brx,bry), confidence=confidence)]
                        
            return to_ret

            