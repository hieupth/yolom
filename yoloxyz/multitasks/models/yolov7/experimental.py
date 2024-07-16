# This file contains experimental modules

import torch
import torch.nn as nn
import random

from yolov7.models.common import Conv, DWConv
from yolov7.models.experimental import Ensemble
from yolov7.utils.google_utils import attempt_download



class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
    

def attempt_load(weights, map_location=None, inplace=True):
    from yolov7.models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(
            ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval()
        )  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [
            nn.Hardswish,
            nn.LeakyReLU,
            nn.ReLU,
            nn.ReLU6,
            nn.SiLU,
            Detect,
            Model,
        ]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


class End2End(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(
        self,
        model,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        trt=False,
        device=None,
    ):
        super().__init__()
        device = device if device is not None else torch.device("cpu")
        assert isinstance(max_wh, (int)) or max_wh is None
        self.model = model.to(device)
        self.trt = trt
        # self.model.model[-1].end2end = True
        self.patch_model = ONNX_ORT if not trt else ONNX_TRT

        self.end2end = self.patch_model(
            max_obj=max_obj,
            iou_thres=iou_thres,
            score_thres=score_thres,
            max_wh=max_wh if not trt else None,
            device=device,
        )
        self.end2end.eval()
        self.end2end = self.end2end.to(device)

    def forward(self, x):
        x = self.model(x)
        # ONNX model
        if not self.trt:
            onnx_head, onnx_face, onnx_body = self.end2end(x)
            return onnx_head, onnx_face, onnx_body

        # TensorRT model
        (
            bnum_det,
            bboxes,
            bscores,
            bcategories,
            hnum_det,
            hboxes,
            hscores,
            hcategories,
            face,
        ) = self.end2end(x)
        # (1, 1), (1, 100, 4), (1, 100), (1, 100), (1, 1), (1, 100, 4), (1, 100), (1, 100), (1, 25200, 21)
        return (
            bnum_det,
            bboxes,
            bscores,
            bcategories,
            hnum_det,
            hboxes,
            hscores,
            hcategories,
            face,
        )


class TRT_NMS(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        out = g.op(
            "TRT::EfficientNMS_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4,
        )
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_TRT(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(
        self,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        device=None,
        n_classes=80,
    ):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device("cpu")
        self.background_class = (-1,)
        self.box_coding = (1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = "1"
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes = n_classes

    def forward(self, x):
        # body layer
        body = x["IDetectBody"][0]
        bnum_det, bboxes, bscores, bcategories = self._convert(body)

        # head layer
        head = x["IDetectHead"][0]
        hnum_det, hboxes, hscores, hcategories = self._convert(head)

        # face layer
        face = x["IKeypoint"][0]

        return (
            bnum_det,
            bboxes,
            bscores,
            bcategories,
            hnum_det,
            hboxes,
            hscores,
            hcategories,
            face,
        )

    def _convert(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        if self.n_classes == 1:
            scores = conf  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            scores *= conf  # conf = obj_conf * cls_conf
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(
            boxes,
            scores,
            self.background_class,
            self.box_coding,
            self.iou_threshold,
            self.max_obj,
            self.plugin_version,
            self.score_activation,
            self.score_threshold,
        )
        return num_det, det_boxes, det_scores, det_classes


class ONNX_ORT(nn.Module):
    """onnx module with ONNX-Runtime NMS operation."""

    def __init__(
        self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=640, device=None
    ):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.max_wh = max_wh  # if max_wh != 0 : non-agnostic else : agnostic
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,
            device=self.device,
        )

    def forward(self, x):
        # body layer
        body = x["IDetectBody"][0]
        bX, _, bboxes, bcategories, bscores, _ = self._convert(body)
        bX = bX.unsqueeze(1).float()
        onnx_body = torch.cat(
            [
                bX,
                bboxes,
                bcategories,
                bscores,
            ],
            1,
        )

        # head layer
        head = x["IDetectHead"][0]
        hX, _, hboxes, hcategories, hscores, _ = self._convert(head)
        hX = hX.unsqueeze(1).float()
        onnx_head = torch.cat(
            [
                hX,
                hboxes,
                hcategories,
                hscores,
            ],
            1,
        )

        # face layer
        face = x["IKeypoint"][0]
        get_kpt = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        fX, _, fboxes, fcategories, fscores, lmks = self._convert(face, get_kpt)
        fX = fX.unsqueeze(1).float()
        onnx_face = torch.cat([fX, fboxes, fcategories, fscores, lmks], 1)

        return onnx_head, onnx_face, onnx_body

    def _convert(self, x, lmks_ls: list = None):
        """specific predict output for onnx

        Args:
            x (numpy array): predict output model should be array
            lmks_ls (list, optional): index of landmark keypoints. Defaults to None.

        Example:
            # if have keypoints should specific get keypoint index
            get_kpts = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            X, Y, bboxes, label, score, kpts  = self._convert(model_output, get_kpts)

            # no key point
            X, Y, bboxes, label, score, lmks  = self._convert(model_output)

        """
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:6]

        if lmks_ls != None:
            assert len(lmks_ls) % 3 == 0
            lmks = x[:, :, lmks_ls]

        scores *= conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(
            nmsbox, max_score_tp, self.max_obj, self.iou_threshold, self.score_threshold
        )

        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        selected_lmks = None if lmks_ls == None else lmks[X, Y, :]

        return X, Y, selected_boxes, selected_categories, selected_scores, selected_lmks


class ORT_NMS(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        max_output_boxes_per_class=torch.tensor([100]),
        iou_threshold=torch.tensor([0.45]),
        score_threshold=torch.tensor([0.25]),
    ):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat(
            [batches[None], zeros[None], idxs[None]], 0
        ).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(
        g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )