from utils import intersection_over_union

def apply_non_max_suppression(single_img_predictions, detection_threshold=0.5, overlap_threshold=0.4):
        boxes, scores, labels = single_img_predictions
    
        # Filter out non-detections, using the classification probability
        # Note that we want to keep the last dimension of boxes, which contains the bounding box cooordinates
        score_mask = (scores > detection_threshold)
        scores = scores[score_mask]
        labels = labels[score_mask]
        boxes = boxes[score_mask, :]

        ## Apply non-max suppression
        for iA,(boxA,scoreA) in enumerate(zip(boxes, scores)):
            if scoreA > 0.0:
                try:
                    # Search forward for overlapping boxes
                    suppression_candidates = [(iA, scoreA)]
                    for iB,(boxB,scoreB) in enumerate(zip(boxes[iA+1:, :], scores[iA+1:]), start=iA+1):
                        if intersection_over_union(boxA, boxB) > overlap_threshold:
                            suppression_candidates.append((iB, scoreB))

                    # Non-max suppression by setting score to -1
                    if len(suppression_candidates) > 1:
                        suppression_candidates.sort(key=lambda x: x[-1])
                        for target, _ in suppression_candidates[:-1]:
                            scores[target] = -1.0

                except IndexError:
                    # catch iA+1 when it goes out of bound
                    # Allegedly try-except blocks are ubiquitous for control-flow in Python
                    pass
                
        return boxes, scores, labels