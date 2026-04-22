# Available osam models (pass as the `model` arg):
#   efficientsam:10m
#   efficientsam:latest
#   sam:100m
#   sam:300m
#   sam:latest
#   sam2:tiny
#   sam2:small
#   sam2:latest
#   sam2:large
#   sam3:latest
#   yoloworld:latest
import numpy as np
import osam


class OSAM:
    def __init__(self, model="sam3:latest"):
        self.model = model

    def predict(self, image, text, box=None, score_threshold=0.1):
        """Run text/box-prompted segmentation via the osam API.

        If ``box`` is None, defaults to the full image so the model gets a
        localization hint.

        Returns:
            boxes: (N, 4) xyxy float32.
            masks: (N, H, W) bool.
        """
        image = np.asarray(image)
        H, W = image.shape[:2]

        if box is None:
            box = [0, 0, W - 1, H - 1]
        # osam encodes a box as two points with labels [2, 3] (top-left, bottom-right).
        points = np.asarray(box, dtype=float).reshape(2, 2)
        point_labels = np.array([2, 3], dtype=int)

        texts = [text] if isinstance(text, str) else text
        request = osam.types.GenerateRequest(
            model=self.model,
            image=image,
            prompt=osam.types.Prompt(
                texts=texts,
                points=points,
                point_labels=point_labels,
                score_threshold=score_threshold,
            ),
        )
        response = osam.apis.generate(request=request)
        annotations = response.annotations

        N = len(annotations)
        boxes = np.zeros((N, 4), dtype=np.float32)
        masks = np.zeros((N, H, W), dtype=bool)
        for i, ann in enumerate(annotations):
            bb = ann.bounding_box
            if bb is not None:
                boxes[i] = [bb.xmin, bb.ymin, bb.xmax, bb.ymax]
            if ann.mask is not None and bb is not None:
                # osam returns masks cropped to the bbox; paste into full frame.
                x0 = max(0, bb.xmin)
                y0 = max(0, bb.ymin)
                x1 = min(W, bb.xmax + 1)
                y1 = min(H, bb.ymax + 1)
                m = ann.mask.astype(bool)
                masks[i, y0:y1, x0:x1] = m[: y1 - y0, : x1 - x0]

        return boxes, masks


if __name__ == "__main__":
    import cv2

    image_path = "deeplabsam/images/example.png"
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read: {image_path}")

    model = OSAM(model="sam3:latest")
    boxes, masks = model.predict(image, text="mice", box=[607, 450, 770, 726])

    print("Boxes:", boxes.shape, boxes.dtype)
    print("Masks:", masks.shape, masks.dtype)
    print("Num detections:", len(boxes))

    overlay = image.copy()
    for m in masks:
        overlay[m] = [0, 0, 255]
    vis = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    for b in boxes.astype(int):
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

    cv2.imshow("osam", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
