import numpy as np
import osam

AVAILABLE_MODELS = tuple(t.name for t in osam.apis.registered_model_types)


class OSAM:
    def __init__(self, model="sam3:latest", device=None):
        if model not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown osam model {model!r}. "
                f"Available: {', '.join(AVAILABLE_MODELS)}"
            )
        import onnxruntime as ort
        if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA requested but CUDAExecutionProvider not available. Install onnxruntime-gpu.")
        self.model = model

    def predict(
        self,
        image,
        text,
        *,
        boxes=None,
        points=None,
        point_labels=None,
        iou_threshold=0.5,
        score_threshold=0.1,
        max_annotations=100,
    ):
        """Run text/box/point-prompted segmentation via the osam API.

        Exposes the full osam ``Prompt`` surface. ``boxes`` is a convenience
        for encoding xyxy rows as point pairs with labels ``[2, 3]`` (top-left,
        bottom-right). If neither ``boxes`` nor ``points`` is given, defaults
        to a full-image box so the model gets a localization hint.

        Returns:
            out_boxes: (N, 4) xyxy float32.
            masks: (N, H, W) bool.
        """
        image = np.asarray(image)
        H, W = image.shape[:2]

        if boxes is None and points is None:
            boxes = np.array([[0, 0, W - 1, H - 1]], dtype=float)
        if boxes is not None:
            if points is not None:
                raise ValueError("pass either `boxes` or `points`, not both")
            boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
            points = boxes.reshape(-1, 2)
            point_labels = np.tile([2, 3], len(boxes))

        prompt_kwargs = {
            "texts": [text] if isinstance(text, str) else text,
            "iou_threshold": iou_threshold,
            "score_threshold": score_threshold,
            "max_annotations": max_annotations,
        }
        if points is not None:
            prompt_kwargs["points"] = np.asarray(points, dtype=float)
        if point_labels is not None:
            prompt_kwargs["point_labels"] = np.asarray(point_labels, dtype=int)

        request = osam.types.GenerateRequest(
            model=self.model,
            image=image,
            prompt=osam.types.Prompt(**prompt_kwargs),
        )
        response = osam.apis.generate(request=request)
        annotations = response.annotations

        N = len(annotations)
        out_boxes = np.zeros((N, 4), dtype=np.float32)
        masks = np.zeros((N, H, W), dtype=bool)
        for i, ann in enumerate(annotations):
            bb = ann.bounding_box
            if bb is not None:
                out_boxes[i] = [bb.xmin, bb.ymin, bb.xmax, bb.ymax]
            if ann.mask is not None and bb is not None:
                # osam returns masks cropped to the bbox; paste into full frame.
                x0 = max(0, bb.xmin)
                y0 = max(0, bb.ymin)
                x1 = min(W, bb.xmax + 1)
                y1 = min(H, bb.ymax + 1)
                m = ann.mask.astype(bool)
                masks[i, y0:y1, x0:x1] = m[: y1 - y0, : x1 - x0]

        return out_boxes, masks


if __name__ == "__main__":
    import cv2

    image_path = "examples/images/example.png"
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read: {image_path}")

    model = OSAM(model="sam3:latest")
    boxes, masks = model.predict(image, text="mice", boxes=[607, 450, 770, 726])

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
